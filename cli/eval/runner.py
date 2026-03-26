import json
import re
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

import mellea
from mellea.backends import ModelOption
from mellea.backends.backend import Backend
from mellea.core import ModelOutputThunk
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.components.unit_test_eval import AgenticTestBasedEval, TestBasedEval

console = Console()


class InputEvalResult:
    """Store results of a single input evaluation (within a unit test)."""

    def __init__(
        self,
        input_text: str,
        model_output: str,
        validation_passed: bool,
        score: int,
        validation_reason: str,  # add input_id
    ):
        self.input_text = input_text
        self.model_output = model_output
        self.validation_passed = validation_passed
        self.score = score
        self.validation_reason = validation_reason

    def to_dict(self):
        return {
            "input": self.input_text,
            "model_output": self.model_output,
            "passed": self.validation_passed,
            "score": self.score,
            "justification": self.validation_reason,
        }


class TestEvalResult:
    """Store results of a single test evaluation."""

    def __init__(self, test_eval: TestBasedEval, input_results: list[InputEvalResult]):
        self.test_eval = test_eval
        self.input_results = input_results

    def to_dict(self):
        return {
            "test_id": self.test_eval.test_id,
            "source": self.test_eval.source,
            "name": self.test_eval.name,
            "instructions": self.test_eval.instructions,
            "input_results": [r.to_dict() for r in self.input_results],
            "expected_targets": self.test_eval.targets,
            "passed": self.passed_count,
            "total_count": self.total_count,
            "pass_rate": self.pass_rate,
        }

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.input_results if r.validation_passed)

    @property
    def total_count(self) -> int:
        return len(self.input_results)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total_count if self.total_count > 0 else 0.0


def create_session(
    backend: str, model: str | None, max_tokens: int | None, base_url: str | None = None
) -> mellea.MelleaSession:
    """Create a mellea session with the specified backend and model."""
    model_id = None
    if model:
        if model.isupper() or "_" in model:
            if hasattr(mellea.model_ids, model):
                model_id = getattr(mellea.model_ids, model)
            else:
                model_id = model
        else:
            model_id = model
    else:
        model_id = mellea.model_ids.IBM_GRANITE_4_MICRO_3B

    try:
        backend_lower = backend.lower()
        backend_instance: Backend

        api_key = None
        if backend_lower == "vllm-server":
            backend_lower = "openai"
            api_key = "EMPTY"

        if backend_lower == "ollama":
            from mellea.backends.ollama import OllamaModelBackend

            backend_instance = OllamaModelBackend(
                model_id=model_id,
                model_options={ModelOption.MAX_NEW_TOKENS: max_tokens},
            )

        elif backend_lower == "openai":
            from mellea.backends.openai import OpenAIBackend

            backend_instance = OpenAIBackend(
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                model_options={ModelOption.MAX_NEW_TOKENS: max_tokens},
            )

        elif backend_lower in ["hf", "huggingface"]:
            from mellea.backends.huggingface import LocalHFBackend

            backend_instance = LocalHFBackend(
                model_id=model_id,
                model_options={ModelOption.MAX_NEW_TOKENS: max_tokens},
            )

        elif backend_lower == "watsonx":
            from mellea.backends.watsonx import WatsonxAIBackend

            backend_instance = WatsonxAIBackend(
                model_id=model_id,
                model_options={ModelOption.MAX_NEW_TOKENS: max_tokens},
            )

        elif backend_lower == "litellm":
            from mellea.backends.litellm import LiteLLMBackend

            backend_instance = LiteLLMBackend(
                model_id=str(model_id),
                model_options={ModelOption.MAX_NEW_TOKENS: max_tokens},
            )

        elif backend_lower == "vllm":
            from mellea.backends.vllm import LocalVLLMBackend

            backend_instance = LocalVLLMBackend(
                model_id=model_id,
                model_options={ModelOption.MAX_NEW_TOKENS: max_tokens},
            )

        else:
            raise ValueError(
                f"Unknown backend: {backend}. Supported: ollama, openai, hf, watsonx, litellm, vllm, vllm-server"
            )

        # create session with backend instance
        from mellea.stdlib.context import SimpleContext

        session = mellea.MelleaSession(backend=backend_instance, ctx=SimpleContext())
        return session

    except Exception as e:
        console.print(
            f"[red]Error creating session with backend={backend}, model={model_id}: {e}[/red]"
        )
        raise


def run_evaluations(
    test_files: list[str],
    backend: str,
    model: str | None,
    base_url: str | None,
    max_gen_tokens: int | None,
    judge_backend: str | None,
    judge_model: str | None,
    judge_base_url: str | None,
    max_judge_tokens: int | None,
    output_path: str,
    output_format: str,
    continue_on_error: bool,
):
    """Run all 'unit test' evaluations

    Each test file should be a json containing:
        "id": an id that is unique to this test file
        "source": the origin for the evaluation prompts, else "N/A"
        "name": an instruction-following attribute that the user intends to evaluate through this test
        "instructions": a set (in string form) of requirements which the generation should follow; the judge will evaluate if these are satisfied
        "examples": a list of entries containing an input_id, an input(prompt), and a list of targets. Each input may have multiple (or no) targets; inputs and targets are in messages format.
    """
    all_test_evals: list[TestBasedEval] = []

    for test_file in test_files:
        try:
            test_evals = TestBasedEval.from_json_file(test_file)
            all_test_evals.extend(test_evals)
            console.print(f"Loaded {len(test_evals)} test evaluations from {test_file}")
        except Exception:
            console.print(f"Error loading {test_file}")

    if not all_test_evals:
        console.print("Failed to load any test evaluations")
        return

    console.print(f"Total test evals to run: {len(all_test_evals)}")
    total_inputs = sum(len(test_eval.inputs) for test_eval in all_test_evals)
    console.print(f"Total inputs to run: {total_inputs}")

    console.print(f"Generation model: {model}")
    console.print(f"Judge model: {judge_model}")

    m = create_session(backend=backend, model=model, max_tokens=max_gen_tokens, base_url=base_url)
    # Use same backend as generator if judge_backend not specified
    judge_session = create_session(
        backend=judge_backend if judge_backend else backend,
        model=judge_model,
        max_tokens=max_judge_tokens,
        base_url=judge_base_url if judge_base_url else base_url,
    )

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Running evals", total=len(all_test_evals))
        for test_eval in all_test_evals:
            try:
                result = execute_test_eval(
                    test_eval=test_eval,
                    generation_session=m,
                    judge_session=judge_session,
                )
                all_results.append(result)
            except Exception as e:
                console.print(f"Error {e} on test {test_eval.test_id}")
                if not continue_on_error:
                    raise

            progress.advance(task)

    summary_stats(all_results)
    save_results(all_results, output_path, output_format, judge_model)

    m.cleanup()
    judge_session.cleanup()


def execute_test_eval(
    test_eval: TestBasedEval,
    generation_session: mellea.MelleaSession,
    judge_session: mellea.MelleaSession,
) -> TestEvalResult:
    """Execute a single test evaluation
    For each input in the test, generate a response using generation_session
    Then, after all inputs are processed, validate using judge_session.
    """
    input_results = []

    # for all inputs, generate responses with generator
    for idx, input_text in enumerate(test_eval.inputs):
        result: ModelOutputThunk = generation_session.act(
            SimpleComponent(instruction=input_text)
        )
        model_output = str(result)

        targets_for_input = (
            test_eval.targets[idx] if idx < len(test_eval.targets) else []
        )

        # query the judge
        test_eval.set_judge_context(
            input_text=input_text,
            prediction=model_output,
            targets_for_input=targets_for_input,
        )
        judge_output_thunk = judge_session.act(test_eval)
        judge_output = str(judge_output_thunk)
        score, justification = parse_judge_output(judge_output)
        passed = score == 1 if score is not None else False

        input_result = InputEvalResult(
            input_text=input_text,
            model_output=model_output,
            validation_passed=passed,
            score=score if score is not None else 0,
            validation_reason=justification,
        )
        input_results.append(input_result)

        # reset both generator and judge
        generation_session.reset()
        judge_session.reset()

    test_result = TestEvalResult(test_eval=test_eval, input_results=input_results)
    return test_result


def parse_judge_output(judge_output: str):
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", judge_output).strip()

    # Try parsing the entire output as JSON first
    try:
        data = json.loads(cleaned)
        return data.get("score"), data.get("justification")
    except (json.JSONDecodeError, AttributeError):
        pass
    
    for match in reversed(list(re.finditer(r'\{', cleaned))):
        candidate = cleaned[match.start():]
        try:
            data = json.loads(candidate)
            if "score" in data:
                return data.get("score"), data.get("justification")
        except (json.JSONDecodeError, ValueError):
            continue

    # if the above fails, search the text for the score
    score_match = re.search(r'score["\s:]+(\d+)', judge_output, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        return score, judge_output

    return None, judge_output


def save_results(results: list[TestEvalResult], output_path: str, output_format: str, judge_model: str | None = None):
    output_path_obj = Path(output_path)
    if output_path_obj.suffix != f".{output_format}":
        output_path_obj = Path(f"{output_path}.{output_format}")

    total_inputs = sum(r.total_count for r in results)
    passed_inputs = sum(r.passed_count for r in results)
    overall_pass_rate = passed_inputs / total_inputs if total_inputs > 0 else 0.0

    if output_format == "jsonl":
        with output_path_obj.open("w") as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + "\n")
    else:  # json
        summary = {
            "total_tests": len(results),
            "total_inputs": total_inputs,
            "passed_inputs": passed_inputs,
            "failed_inputs": total_inputs - passed_inputs,
            "overall_pass_rate": overall_pass_rate,
            "judge_model": judge_model,
        }

        with output_path_obj.open("w") as f:
            json.dump(
                {"summary": summary, "results": [r.to_dict() for r in results]},
                f,
                indent=2,
            )

    console.print(f"Results saved to {output_path}")


def summary_stats(results: list[TestEvalResult]):
    total_tests = len(results)
    tests_fully_passed = sum(1 for r in results if r.passed_count == r.total_count)
    test_pass_rate = tests_fully_passed / total_tests if total_tests > 0 else 0.0

    total_inputs = sum(r.total_count for r in results)
    passed_inputs = sum(r.passed_count for r in results)
    overall_pass_rate = passed_inputs / total_inputs if total_inputs > 0 else 0.0

    console.print(f"\nTotal Unit Tests: {total_tests}")
    console.print(
        f"Unit Test Pass Rate: {tests_fully_passed}/{total_tests} ({test_pass_rate * 100:.1f}%)"
    )
    console.print()
    console.print(f"Total number of inputs across tests: {total_inputs}")
    console.print(f"Number of inputs passed across tests: {passed_inputs}")
    console.print(f"Cumulative Pass Rate: {overall_pass_rate * 100:.1f}%")
    console.print()

    if len(results) > 1:
        console.print("Per-Test Breakdown:")
        for result in results:
            ut_score = "1/1" if result.passed_count == result.total_count else "0/1"
            console.print(
                f"\t{result.test_eval.name}: {ut_score} ({result.passed_count}/{result.total_count})"
            )
        console.print("\n\n")


def execute_agentic_test_eval(
    test_eval: AgenticTestBasedEval, judge_session: mellea.MelleaSession
) -> TestEvalResult:
    """Execute an agentic test evaluation using pre-computed generations (offline).

    Evaluates each turn in sequence. On pass, the user turn and model output are
    appended to the conversation history for the next turn's judge context. If
    early_stop is set and a turn fails, evaluation stops immediately.
    """
    input_results = []
    conversation_history: list[dict] = []
    final_idx = len(test_eval.inputs) - 1

    for idx, input_text in enumerate(test_eval.inputs):
        model_output = (
            test_eval.generations[idx] if idx < len(test_eval.generations) else ""
        )

        targets_for_input = (
            test_eval.targets[idx] if idx < len(test_eval.targets) else []
        )

        is_final_turn = idx == final_idx

        test_eval.set_judge_context(
            input_text=input_text,
            prediction=model_output,
            targets_for_input=targets_for_input,
            conversation_history=conversation_history,
        )
        judge_output_thunk = judge_session.act(test_eval)
        judge_output = str(judge_output_thunk)
        score, justification = parse_judge_output(judge_output)
        passed = score == 1 if score is not None else False

        input_result = InputEvalResult(
            input_text=input_text,
            model_output=model_output,
            validation_passed=passed,
            score=score if score is not None else 0,
            validation_reason=justification,
        )
        input_results.append(input_result)
        judge_session.reset()

        if test_eval.is_multi_turn and not is_final_turn:
            if test_eval.early_stop:
                # early stop: use model's response as context, stop if model fails
                conversation_history.append({"role": "user", "content": input_text})
                conversation_history.append({"role": "assistant", "content": model_output})
                if not passed:
                    console.print(
                        f"[yellow]Early stop: turn {idx + 1} failed for {test_eval.name}[/yellow]"
                    )
                    break
            else:
                # no early stop: use gold response as context
                gold_response = (targets_for_input[0] if targets_for_input else model_output)
                conversation_history.append({"role": "user", "content": input_text})
                conversation_history.append({"role": "assistant", "content": gold_response})

    # Pad skipped turns (due to early stop) as failed so they count in totals
    for skipped_idx in range(len(input_results), len(test_eval.inputs)):
        skipped_output = (
            test_eval.generations[skipped_idx]
            if skipped_idx < len(test_eval.generations)
            else ""
        )
        input_results.append(
            InputEvalResult(
                input_text=test_eval.inputs[skipped_idx],
                model_output=skipped_output,
                validation_passed=False,
                score=0,
                validation_reason="Skipped due to early stop on a prior turn.",
            )
        )

    return TestEvalResult(test_eval=test_eval, input_results=input_results)


### NOTE: PATH PREFIX/SUFFIX HARDCODED, MAKE MORE GENERAL
def find_agentic_test_pairs(test_dir: str) -> list[tuple[str, str]]:
    """Find (test_file, generations_file) pairs in a directory.

    Looks for *_mellea.json files paired with sim_*.json files in the same subdirectory.
    """
    test_dir_path = Path(test_dir)
    pairs = []

    for mellea_file in sorted(test_dir_path.rglob("*_mellea.json")):
        parent = mellea_file.parent
        sim_files = sorted(parent.glob("sim_*.json"))
        if sim_files:
            pairs.append((str(mellea_file), str(sim_files[0])))
        else:
            console.print(
                f"[yellow]No sim file found for {mellea_file}, skipping[/yellow]"
            )

    return pairs


def run_agentic_evaluations(
    test_dir: str,
    judge_backend: str,
    judge_model: str | None,
    judge_base_url: str | None,
    max_judge_tokens: int | None,
    output_path: str,
    output_format: str,
    early_stop: bool,
    continue_on_error: bool,
):
    """Run agentic (offline, multi-turn) evaluations."""
    pairs = find_agentic_test_pairs(test_dir)
    if not pairs:
        console.print("[red]No test/generation file pairs found[/red]")
        return

    all_test_evals: list[AgenticTestBasedEval] = []
    for test_file, gen_file in pairs:
        try:
            evals = AgenticTestBasedEval.from_agentic_json(
                test_file, gen_file, early_stop=early_stop
            )
            all_test_evals.extend(evals)
            console.print(f"Loaded {len(evals)} agentic test(s) from {test_file}")
        except Exception as e:
            console.print(f"[red]Error loading {test_file}: {e}[/red]")
            if not continue_on_error:
                raise

    if not all_test_evals:
        console.print("[red]Failed to load any agentic test evaluations[/red]")
        return

    console.print(f"Total agentic tests: {len(all_test_evals)}")
    total_turns = sum(len(t.inputs) for t in all_test_evals)
    console.print(f"Total turns to judge: {total_turns}")
    console.print(f"Judge model: {judge_model}")

    judge_session = create_session(
        backend=judge_backend,
        model=judge_model,
        max_tokens=max_judge_tokens,
        base_url=judge_base_url,
    )

    all_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Running agentic evals", total=len(all_test_evals))
        for test_eval in all_test_evals:
            try:
                result = execute_agentic_test_eval(
                    test_eval=test_eval, judge_session=judge_session
                )
                all_results.append(result)
            except Exception as e:
                console.print(f"[red]Error on test {test_eval.test_id}: {e}[/red]")
                if not continue_on_error:
                    raise
            progress.advance(task)

    summary_stats(all_results)
    save_results(all_results, output_path, output_format, judge_model)
    judge_session.cleanup()
