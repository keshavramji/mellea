"""Pytest plugin that drives the test-based LLM evaluation pipeline.

`EvalPlugin` carries all runtime configuration (the loaded `TestBasedEval`
objects, the generation/judge backend settings, the pass threshold, and the
output path) into a pytest session. It parametrizes one test per
`TestBasedEval`, owns the generation and judge sessions for the run, collects
the per-test results, and writes the custom JSON/JSONL results file when the
session finishes.

The plugin is passed to ``pytest.main(..., plugins=[EvalPlugin(...)])`` from
``cli.eval.runner.run_evaluations`` -- it is never registered via
``pytest_plugins`` and the eval test module lives outside ``test/`` so the
normal test suite never collects it.
"""

from __future__ import annotations

import mellea
from cli.eval.runner import (
    TestEvalResult,
    console,
    create_session,
    save_results,
    summary_stats,
)
from mellea.stdlib.components.unit_test_eval import TestBasedEval


class EvalPlugin:
    """Pytest plugin holding the runtime state for one evaluation run.

    Args:
        test_evals: The unit tests to run, one parametrized pytest case each.
        backend: Generation backend name.
        model: Generation model id, or `None` for the default.
        max_gen_tokens: Max tokens for the generation model, or `None`.
        judge_backend: Judge backend name, or `None` to reuse `backend`.
        judge_model: Judge model id, or `None` for the default.
        max_judge_tokens: Max tokens for the judge model, or `None`.
        pass_threshold: Minimum aggregate pass rate for a test to pass.
        output_path: File path prefix for the results file.
        output_format: `"json"` or `"jsonl"`.
    """

    def __init__(
        self,
        test_evals: list[TestBasedEval],
        backend: str,
        model: str | None,
        max_gen_tokens: int | None,
        judge_backend: str | None,
        judge_model: str | None,
        max_judge_tokens: int | None,
        pass_threshold: float,
        output_path: str,
        output_format: str,
    ):
        self.test_evals = test_evals
        self.backend = backend
        self.model = model
        self.max_gen_tokens = max_gen_tokens
        self.judge_backend = judge_backend
        self.judge_model = judge_model
        self.max_judge_tokens = max_judge_tokens
        self.pass_threshold = pass_threshold
        self.output_path = output_path
        self.output_format = output_format

        self.results: list[TestEvalResult] = []
        self.error_tests: list[str] = []
        self.gen_session: mellea.MelleaSession | None = None
        self.judge_session: mellea.MelleaSession | None = None

    # --- hooks ---

    def pytest_configure(self, config):
        """Expose this plugin to the eval test module via the pytest config."""
        config._mellea_eval = self

    def pytest_sessionstart(self, session):
        """Create the generation and judge sessions for this run."""
        self.gen_session = create_session(
            backend=self.backend, model=self.model, max_tokens=self.max_gen_tokens
        )
        # Reuse the generation backend if no judge backend is specified.
        self.judge_session = create_session(
            backend=self.judge_backend if self.judge_backend else self.backend,
            model=self.judge_model,
            max_tokens=self.max_judge_tokens,
        )

    def pytest_generate_tests(self, metafunc):
        """Parametrize one pytest case per `TestBasedEval`."""
        if "test_eval" in metafunc.fixturenames:
            metafunc.parametrize(
                "test_eval",
                self.test_evals,
                ids=[t.test_id or t.name for t in self.test_evals],
            )

    def pytest_sessionfinish(self, session, exitstatus):
        """Summarise and persist results, then tear down the sessions."""
        if self.results:
            summary_stats(self.results)
            save_results(self.results, self.output_path, self.output_format)
        else:
            console.print("No evaluation results were produced.")

        if self.gen_session is not None:
            self.gen_session.cleanup()
        if self.judge_session is not None:
            self.judge_session.cleanup()

    # --- helpers used by the eval test module ---

    def record(self, result: TestEvalResult) -> None:
        """Record one test's result for the final summary and output file."""
        self.results.append(result)

    def mark_error(self, test_name: str) -> None:
        """Flag a test whose evaluation raised (the score could not be measured)."""
        self.error_tests.append(test_name)
