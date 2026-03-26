"""LLM Evaluation with Unit Tests in Mellea."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ...core import CBlock, Component, ModelOutputThunk, TemplateRepresentation


def extract_generations_from_trajectory(
    trajectory_path: str, num_turns: int
) -> list[str]:
    """Extract pre-computed generations from a trajectory file.

    Skips the first user turn (setup). For each subsequent user turn, the generation
    is the assistant turn immediately before the next user turn. For the last turn,
    it's the last assistant turn in the conversation. Empty string if none found.
    """
    path = Path(trajectory_path)
    with path.open("r") as f:
        data = json.load(f)

    conversation = data["conversation"]
    user_indices = [
        i for i, turn in enumerate(conversation) if turn.get("role") == "user"
    ]
    # skip setup turn
    task_user_indices = user_indices[1:]

    if len(task_user_indices) < num_turns:
        raise ValueError(
            f"Trajectory has {len(task_user_indices)} task user turns, "
            f"expected {num_turns}."
        )

    generations: list[str] = []
    for turn_idx in range(num_turns):
        if turn_idx < num_turns - 1:
            next_user_pos = task_user_indices[turn_idx + 1]
            prev_turn = conversation[next_user_pos - 1]
            if prev_turn.get("role") == "assistant":
                generations.append(_extract_content(prev_turn))
            else:
                generations.append("")
        else:
            # last turn: only use if conversation ends with an assistant turn
            last_turn = conversation[-1]
            if last_turn.get("role") == "assistant":
                generations.append(_extract_content(last_turn))
            else:
                generations.append("")

    return generations


def _extract_content(turn: dict) -> str:
    """Extract text content from a conversation turn."""
    content = turn.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


class Message(BaseModel):
    """Schema for a message in the test data."""

    role: str
    content: str


class Example(BaseModel):
    """Schema for an example in the test data."""

    input: list[Message]
    targets: list[Message] = Field(default_factory=list)
    input_id: str = ""


class TestData(BaseModel):
    """Schema for test data loaded from json."""

    source: str
    name: str
    instructions: str
    examples: list[Example] = Field(default_factory=list)
    id: str

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, v):
        """Ensure examples list is not empty."""
        if not v:
            raise ValueError("examples list cannot be empty")
        return v


class TestBasedEval(Component[str]):
    """Each TestBasedEval represents a single unit test."""

    def __init__(
        self,
        source: str,
        name: str,
        instructions: str,
        inputs: list[str],
        targets: list[list[str]] | None = None,  # can be optional
        test_id: str | None = None,
        input_ids: list[str] | None = None,
    ):
        """Initialize TestBasedEval (for a single unit test)."""
        self.source = source
        self.name = name
        self.instructions = instructions
        self.inputs = inputs
        self.targets = targets or []
        self.test_id = test_id
        self.input_ids = input_ids or []

    def parts(self) -> list[Component | CBlock]:
        """The set of constituent parts of the Component."""
        return []

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the test for judge evaluation."""
        return TemplateRepresentation(
            obj=self,
            args=self._judge_context if hasattr(self, "_judge_context") else {},
            template_order=["*"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""

    def set_judge_context(
        self, input_text: str, prediction: str, targets_for_input: list[str]
    ):
        """Set context for judge evaluation."""
        if len(targets_for_input) == 0:  # no reference
            target_text = "N/A"
        elif len(targets_for_input) == 1:
            target_text = targets_for_input[0]
        else:  # enumerate when there are multiple targets
            target_text = "\n".join(
                [f"{i}. {target}" for i, target in enumerate(targets_for_input, 1)]
            )

        self._judge_context: dict[str, Any] = {
            "input": input_text,
            "prediction": prediction,
            "target": target_text,
            "guidelines": self.instructions,
        }

    @classmethod
    def from_json_file(cls, filepath: str) -> list["TestBasedEval"]:
        """Load test evaluations from json/jsonl file, return list of TestBasedEval instances, one per 'unit test'."""
        path = Path(filepath)

        with path.open("r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        test_evals = []
        for test_data_dict in data:
            try:
                test_data = TestData(**test_data_dict)
            except Exception as e:
                raise ValueError(f"Invalid test data in {filepath}: {e}")

            inputs = []
            targets = []
            input_ids = []

            for example in test_data.examples:
                user_messages = [msg for msg in example.input if msg.role == "user"]
                if user_messages:
                    inputs.append(user_messages[-1].content)

                targets_for_input = [
                    msg.content for msg in example.targets if msg.role == "assistant"
                ]
                targets.append(targets_for_input)

                input_ids.append(example.input_id)

            test_eval = cls(
                source=test_data.source,
                name=test_data.name,
                instructions=test_data.instructions,
                inputs=inputs,
                targets=targets,
                test_id=test_data.id,
                input_ids=input_ids,
            )
            test_evals.append(test_eval)

        return test_evals


class AgenticTestBasedEval(TestBasedEval):
    """Multi-turn unit test with pre-computed generations (offline mode)."""

    def __init__(
        self,
        source: str,
        name: str,
        instructions: str,
        inputs: list[str],
        targets: list[list[str]] | None = None,
        test_id: str | None = None,
        input_ids: list[str] | None = None,
        generations: list[str] | None = None,
        early_stop: bool = False,
        is_multi_turn: bool = False,
    ):
        """Initialize an agentic test with pre-computed generations."""
        super().__init__(
            source=source,
            name=name,
            instructions=instructions,
            inputs=inputs,
            targets=targets,
            test_id=test_id,
            input_ids=input_ids,
        )
        self.generations = generations or []
        self.early_stop = early_stop
        self.is_multi_turn = is_multi_turn

    def set_judge_context(
        self,
        input_text: str,
        prediction: str,
        targets_for_input: list[str],
        conversation_history: list[dict] | None = None,
    ):
        """Set context for judge evaluation, including prior conversation history."""
        super().set_judge_context(input_text, prediction, targets_for_input)
        self._judge_context["conversation_history"] = conversation_history or []

    @classmethod
    def from_agentic_json(
        cls, test_filepath: str, generations_filepath: str, early_stop: bool = False
    ) -> list["AgenticTestBasedEval"]:
        """Load agentic test evals from a unit test file and a trajectory file.

        The test file defines multi-turn inputs/targets. The trajectory file provides
        pre-computed generations. Intermediate assistant turns in the input section
        serve as targets for earlier user turns; the targets section provides the
        target for the final user turn.
        """
        path = Path(test_filepath)
        with path.open("r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        test_evals = []
        for test_data_dict in data:
            try:
                test_data = TestData(**test_data_dict)
            except Exception as e:
                raise ValueError(f"Invalid test data in {test_filepath}: {e}")

            # Check if this is multi-turn (any example has >1 user turn)
            is_multi_turn = any(
                sum(1 for m in ex.input if m.role == "user") > 1
                for ex in test_data.examples
            )

            if is_multi_turn:
                # Each example is its own test with multiple turns
                for example in test_data.examples:
                    inputs, targets, input_ids = cls._parse_multi_turn_example(example)

                    generations = extract_generations_from_trajectory(
                        generations_filepath, len(inputs)
                    )

                    test_evals.append(
                        cls(
                            source=test_data.source,
                            name=test_data.name,
                            instructions=test_data.instructions,
                            inputs=inputs,
                            targets=targets,
                            test_id=test_data.id,
                            input_ids=input_ids,
                            generations=generations,
                            early_stop=early_stop,
                            is_multi_turn=True,
                        )
                    )
            else:
                # Single-turn examples: group all into one test (like from_json_file)
                all_inputs = []
                all_targets = []
                all_input_ids = []

                for example in test_data.examples:
                    user_messages = [msg for msg in example.input if msg.role == "user"]
                    if user_messages:
                        all_inputs.append(user_messages[-1].content)

                        targets_for_input = [
                            msg.content
                            for msg in example.targets
                            if msg.role == "assistant"
                        ]
                        all_targets.append(targets_for_input)
                        all_input_ids.append(example.input_id)

                generations = extract_generations_from_trajectory(
                    generations_filepath, len(all_inputs)
                )

                test_evals.append(
                    cls(
                        source=test_data.source,
                        name=test_data.name,
                        instructions=test_data.instructions,
                        inputs=all_inputs,
                        targets=all_targets,
                        test_id=test_data.id,
                        input_ids=all_input_ids,
                        generations=generations,
                        early_stop=early_stop,
                    )
                )

        return test_evals

    @classmethod
    def _parse_multi_turn_example(cls, example: "Example"):
        """Parse a single multi-turn example into inputs, targets, and input_ids."""
        inputs = []
        targets = []
        input_ids = []

        turns = example.input
        user_turn_count = 0
        for i, msg in enumerate(turns):
            if msg.role == "user":
                user_turn_count += 1
                intermediate_target = []
                for j in range(i + 1, len(turns)):
                    if turns[j].role == "assistant":
                        intermediate_target.append(turns[j].content)
                        break
                    elif turns[j].role == "user":
                        break

                inputs.append(msg.content)
                targets.append(intermediate_target)
                input_ids.append(f"{example.input_id}.turn_{user_turn_count}")

        # Replace the last target with the targets section
        if inputs and example.targets:
            final_targets = [
                msg.content for msg in example.targets if msg.role == "assistant"
            ]
            targets[-1] = final_targets

        return inputs, targets, input_ids
