"""Unit tests for the pytest-driven eval pipeline — no backend, no model.

Covers EvalPlugin parametrization, the run_evaluations -> pytest.main wiring,
and the threshold assertion in the parametrized eval test module.
"""

import pytest

import cli.eval.runner as runner

# Aliased so pytest does not collect this imported function as a test here.
from cli.eval._eval_module import test_unit_test_eval as run_eval_case
from cli.eval.pytest_plugin import EvalPlugin
from cli.eval.runner import InputEvalResult, TestEvalResult
from mellea.stdlib.components.unit_test_eval import TestBasedEval

# --- helpers ---


def _make_eval(name: str, test_id: str | None = None) -> TestBasedEval:
    return TestBasedEval(
        source="src",
        name=name,
        instructions="Judge if correct",
        inputs=["q1", "q2"],
        test_id=test_id,
    )


def _make_plugin(test_evals, *, pass_threshold: float = 1.0) -> EvalPlugin:
    return EvalPlugin(
        test_evals=test_evals,
        backend="ollama",
        model=None,
        max_gen_tokens=None,
        judge_backend=None,
        judge_model=None,
        max_judge_tokens=None,
        pass_threshold=pass_threshold,
        output_path="out",
        output_format="json",
    )


def _result(passed: list[bool]) -> TestEvalResult:
    input_results = [
        InputEvalResult(f"q{i}", f"a{i}", p, 1 if p else 0, "reason")
        for i, p in enumerate(passed)
    ]
    return TestEvalResult(_make_eval("t"), input_results)


class _StubMetafunc:
    def __init__(self, fixturenames):
        self.fixturenames = fixturenames
        self.calls: list[tuple] = []

    def parametrize(self, argname, argvalues, ids=None):
        self.calls.append((argname, list(argvalues), ids))


# --- EvalPlugin.pytest_generate_tests ---


def test_parametrize_uses_test_id_then_name_for_ids():
    evals = [_make_eval("first", test_id="id-1"), _make_eval("second")]
    plugin = _make_plugin(evals)

    metafunc = _StubMetafunc(fixturenames=["test_eval", "eval_runtime"])
    plugin.pytest_generate_tests(metafunc)

    assert len(metafunc.calls) == 1
    argname, argvalues, ids = metafunc.calls[0]
    assert argname == "test_eval"
    assert argvalues == evals
    # test_id wins when present, otherwise fall back to name.
    assert ids == ["id-1", "second"]


def test_parametrize_skipped_when_fixture_absent():
    plugin = _make_plugin([_make_eval("only")])
    metafunc = _StubMetafunc(fixturenames=["something_else"])
    plugin.pytest_generate_tests(metafunc)
    assert metafunc.calls == []


# --- run_evaluations -> pytest.main wiring ---


@pytest.fixture
def captured_pytest_main(monkeypatch):
    """Capture the args/plugins passed to pytest.main and skip real execution."""
    captured: dict = {}

    def fake_main(args, plugins=None):
        captured["args"] = args
        captured["plugins"] = plugins
        return 0

    monkeypatch.setattr(runner.pytest, "main", fake_main)
    # Avoid touching the filesystem during loading.
    monkeypatch.setattr(
        TestBasedEval, "from_json_file", classmethod(lambda cls, fp: [_make_eval("t")])
    )
    return captured


def _run(continue_on_error: bool, captured):
    code = runner.run_evaluations(
        test_files=["whatever.json"],
        backend="ollama",
        model=None,
        max_gen_tokens=None,
        judge_backend=None,
        judge_model=None,
        max_judge_tokens=None,
        output_path="out",
        output_format="json",
        continue_on_error=continue_on_error,
        pass_threshold=0.5,
    )
    return code


def test_run_evaluations_invokes_pytest_with_eval_module(captured_pytest_main):
    code = _run(continue_on_error=True, captured=captured_pytest_main)
    assert code == 0

    args = captured_pytest_main["args"]
    assert any(a.endswith("_eval_module.py") for a in args)
    assert "--no-cov" in args
    assert "--timeout=0" in args
    # continue_on_error=True must NOT add -x.
    assert "-x" not in args

    plugins = captured_pytest_main["plugins"]
    assert len(plugins) == 1 and isinstance(plugins[0], EvalPlugin)
    assert plugins[0].pass_threshold == 0.5


def test_run_evaluations_adds_dash_x_when_not_continue_on_error(captured_pytest_main):
    _run(continue_on_error=False, captured=captured_pytest_main)
    assert "-x" in captured_pytest_main["args"]


def test_run_evaluations_returns_eval_error_when_no_tests_loaded(monkeypatch):
    monkeypatch.setattr(
        TestBasedEval, "from_json_file", classmethod(lambda cls, fp: [])
    )
    code = runner.run_evaluations(
        test_files=["whatever.json"],
        backend="ollama",
        model=None,
        max_gen_tokens=None,
        judge_backend=None,
        judge_model=None,
        max_judge_tokens=None,
        output_path="out",
        output_format="json",
        continue_on_error=True,
    )
    assert code == runner.EVAL_ERROR


# --- classify_exit_code ---


def test_classify_all_passed():
    assert runner.classify_exit_code(0, has_eval_errors=False) == runner.ALL_PASSED


def test_classify_below_threshold():
    assert runner.classify_exit_code(1, has_eval_errors=False) == runner.BELOW_THRESHOLD


def test_classify_eval_error_outranks_below_threshold():
    # A test that raised (eval error) outranks a below-threshold failure even
    # though both surface as pytest exit code 1.
    assert runner.classify_exit_code(1, has_eval_errors=True) == runner.EVAL_ERROR


@pytest.mark.parametrize("pytest_exit", [2, 3, 4, 5])
def test_classify_session_problems_are_eval_errors(pytest_exit):
    assert runner.classify_exit_code(pytest_exit, has_eval_errors=False) == (
        runner.EVAL_ERROR
    )


# --- threshold assertion in the eval test module ---


class _StubRuntime:
    def __init__(self, result: TestEvalResult | None, pass_threshold: float):
        self._result = result
        self.pass_threshold = pass_threshold
        self.gen_session = object()
        self.judge_session = object()
        self.recorded: list[TestEvalResult] = []
        self.errors: list[str] = []

    def record(self, result):
        self.recorded.append(result)

    def mark_error(self, test_name):
        self.errors.append(test_name)


def test_eval_module_passes_at_or_above_threshold(monkeypatch):
    result = _result([True, True])  # pass_rate 1.0
    monkeypatch.setattr("cli.eval._eval_module.execute_test_eval", lambda **kw: result)
    runtime = _StubRuntime(result, pass_threshold=1.0)
    run_eval_case(_make_eval("t"), runtime)  # should not raise
    assert runtime.recorded == [result]
    assert runtime.errors == []


def test_eval_module_fails_below_threshold(monkeypatch):
    result = _result([True, False])  # pass_rate 0.5
    monkeypatch.setattr("cli.eval._eval_module.execute_test_eval", lambda **kw: result)
    runtime = _StubRuntime(result, pass_threshold=1.0)
    with pytest.raises(AssertionError):
        run_eval_case(_make_eval("t"), runtime)
    # The result is still recorded before the assertion fails, and it is NOT an
    # eval error (the score was measured, it just fell short).
    assert runtime.recorded == [result]
    assert runtime.errors == []


def test_eval_module_marks_error_when_execution_raises(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr("cli.eval._eval_module.execute_test_eval", _boom)
    runtime = _StubRuntime(None, pass_threshold=1.0)
    with pytest.raises(RuntimeError):
        run_eval_case(_make_eval("boom-test"), runtime)
    # Flagged as an eval error, and nothing recorded since no score was produced.
    assert runtime.errors == ["boom-test"]
    assert runtime.recorded == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
