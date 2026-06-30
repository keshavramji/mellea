"""Parametrized pytest module backing ``m eval run``.

`pytest.main` is pointed at this file by ``cli.eval.runner.run_evaluations``,
with an `EvalPlugin` instance supplying the runtime configuration. The plugin's
`pytest_generate_tests` hook parametrizes `test_unit_test_eval` with one case
per `TestBasedEval`; each case generates responses, scores them with the judge,
and asserts the aggregate pass rate against the configured threshold.

This module lives under ``cli/eval/`` (not ``test/``) so the normal test suite
(`testpaths = ["test", "docs"]`) never collects it.
"""

import pytest

from cli.eval.runner import execute_test_eval


@pytest.fixture(scope="session")
def eval_runtime(request):
    """Return the `EvalPlugin` carrying this run's sessions and config."""
    return request.config._mellea_eval


def test_unit_test_eval(test_eval, eval_runtime):
    """Run one unit test and assert its aggregate pass rate meets the threshold."""
    try:
        result = execute_test_eval(
            test_eval=test_eval,
            generation_session=eval_runtime.gen_session,
            judge_session=eval_runtime.judge_session,
        )
    except Exception:
        # The score could not be measured (backend/OOM/parse failure). Flag it
        # as an eval error so the run is classified distinctly from a model that
        # simply scored below threshold, then let the test fail normally.
        eval_runtime.mark_error(test_eval.name)
        raise
    eval_runtime.record(result)

    assert result.pass_rate >= eval_runtime.pass_threshold, (
        f"{test_eval.name}: pass rate {result.pass_rate:.1%} "
        f"< threshold {eval_runtime.pass_threshold:.1%} "
        f"({result.passed_count}/{result.total_count} inputs passed)"
    )
