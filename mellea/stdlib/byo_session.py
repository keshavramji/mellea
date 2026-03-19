"""BYO (Bring Your Own) Generator Session for Mellea.

Provides a MelleaSession shell that keeps Mellea primitives but allows
generation to be done through an external API or pre-computed outputs,
while still supporting Mellea's evaluation and validation pipeline.
"""

from __future__ import annotations

import datetime
from copy import deepcopy
from typing import Any, Literal, overload

from ..backends.model_ids import ModelIdentifier
from ..core import (
    Backend,
    BaseModelSubclass,
    CBlock,
    Component,
    Context,
    FancyLogger,
    GenerateLog,
    ModelOutputThunk,
    Requirement,
    S,
    SamplingResult,
    SamplingStrategy,
    ValidationResult,
)
from ..stdlib import functional as mfuncs
from .callable_backend import CallableBackend, GeneratorFn
from .context import SimpleContext
from .sampling import RejectionSamplingStrategy
from .session import MelleaSession, backend_name_to_class


class BYOGeneratorSession(MelleaSession):
    """A MelleaSession that decouples generation from Mellea backends.

    Supports two modes:
    1. **Callable generation**: provide a `generator_fn` that produces responses.
    2. **Injection-only**: use `.inject()` to feed pre-computed outputs.

    Validation / judging can be performed by a separate `judge_backend`
    (a real Mellea Backend with an LLM), which is used for LLM-as-a-Judge
    requirements and sampling strategy validation loops.

    Examples:
        ```python
        # Callable generation with a judge
        def my_agent(action, ctx):
            return call_my_external_api(str(action))

        session = BYOGeneratorSession(
            generator_fn=my_agent,
            judge_backend=my_ollama_backend,
        )
        result = session.act(some_component)
        validations = session.validate(my_requirements)

        # Injection-only (pre-computed outputs)
        session = BYOGeneratorSession(judge_backend=my_judge)
        session.inject("The pre-computed agent response", action=input_component)
        validations = session.validate(my_requirements)
        ```
    """

    def __init__(
        self,
        generator_fn: GeneratorFn | None = None,
        *,
        judge_backend: Backend | None = None,
        ctx: Context | None = None,
    ):
        """Initialize a BYO Generator Session.

        Args:
            generator_fn: A callable that takes (action, context) and returns
                a string (sync or async). If None, the session is injection-only.
            judge_backend: A Mellea Backend used for LLM-as-a-Judge validation.
                Required if using LLMaJ Requirements or sampling strategies
                with LLMaJ validation.
            ctx: The context to use. Defaults to SimpleContext().
        """
        gen_backend: Backend
        if generator_fn is not None:
            gen_backend = CallableBackend(generator_fn)
        else:
            gen_backend = _InjectionOnlyBackend()

        super().__init__(
            backend=gen_backend, ctx=ctx if ctx is not None else SimpleContext()
        )
        self.judge_backend = judge_backend

    def inject(
        self, output: str, *, action: Component | CBlock | None = None
    ) -> ModelOutputThunk:
        """Inject a pre-computed output string into the session context.

        Creates a computed ModelOutputThunk from the output string and adds
        it to the context. If an action is provided, it is added to the
        context before the output.

        Args:
            output: The pre-computed output string.
            action: Optional action (Component or CBlock) that produced
                this output. Added to context before the output.

        Returns:
            The created ModelOutputThunk.
        """
        thunk: ModelOutputThunk = ModelOutputThunk(output)
        thunk._computed = True

        if action is not None:
            thunk._action = action
            if isinstance(action, Component):
                thunk.parsed_repr = action.parse(thunk)
            self.ctx = self.ctx.add(action)

        thunk._generate_log = GenerateLog(
            date=datetime.datetime.now(tz=datetime.timezone.utc),
            prompt=None,
            backend="BYOGeneratorSession.inject",
            model_output=output,
            action=action,
            result=thunk,
            is_final_result=True,
        )

        self.ctx = self.ctx.add(thunk)
        return thunk

    @overload
    def act(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[S]: ...

    @overload
    def act(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[True],
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult[S]: ...

    def act(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: bool = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[S] | SamplingResult:
        """Run an action using the BYO generator.

        If a sampling strategy is provided with requirements that need LLM-as-a-Judge,
        the judge_backend is used for validation while the callable backend handles
        generation.
        """
        has_reqs = requirements is not None and len(requirements) > 0
        has_llmaj_reqs = has_reqs and any(
            r.validation_fn is None
            for r in requirements  # type: ignore
        )

        if strategy is None or not has_llmaj_reqs:
            return super().act(  # type: ignore[call-overload]
                action,
                requirements=requirements,
                strategy=strategy,
                return_sampling_results=return_sampling_results,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )

        if self.judge_backend is None:
            raise RuntimeError(
                "A judge_backend is required when using a sampling strategy with "
                "LLM-as-a-Judge requirements. Provide a judge_backend when creating "
                "the BYOGeneratorSession, or use validation_fn-based Requirements."
            )

        from ..helpers import _run_async_in_thread

        result = _run_async_in_thread(
            self._dual_backend_sample(
                action,
                requirements=requirements,
                strategy=strategy,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )
        )

        if return_sampling_results:
            assert isinstance(result, SamplingResult)
            self.ctx = result.result_ctx
            return result
        else:
            assert isinstance(result, SamplingResult)
            self.ctx = result.result_ctx
            return result.result

    @overload
    async def aact(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[False] = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[S]: ...

    @overload
    async def aact(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: Literal[True],
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult[S]: ...

    async def aact(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None = None,
        strategy: SamplingStrategy | None = RejectionSamplingStrategy(loop_budget=2),
        return_sampling_results: bool = False,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[S] | SamplingResult:
        """Async version of .act."""
        has_reqs = requirements is not None and len(requirements) > 0
        has_llmaj_reqs = has_reqs and any(
            r.validation_fn is None
            for r in requirements  # type: ignore
        )

        if strategy is None or not has_llmaj_reqs:
            return await super().aact(  # type: ignore[call-overload]
                action,
                requirements=requirements,
                strategy=strategy,
                return_sampling_results=return_sampling_results,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )

        if self.judge_backend is None:
            raise RuntimeError(
                "A judge_backend is required when using a sampling strategy with "
                "LLM-as-a-Judge requirements. Provide a judge_backend when creating "
                "the BYOGeneratorSession, or use validation_fn-based Requirements."
            )

        result = await self._dual_backend_sample(
            action,
            requirements=requirements,
            strategy=strategy,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        if return_sampling_results:
            self.ctx = result.result_ctx
            return result
        else:
            self.ctx = result.result_ctx
            return result.result

    async def _dual_backend_sample(
        self,
        action: Component[S],
        *,
        requirements: list[Requirement] | None,
        strategy: SamplingStrategy,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult[S]:
        """Run a sampling loop using callable backend for gen, judge backend for validation.

        Same logic as BaseSamplingStrategy.sample()  but splits
        generation and validation across two backends.
        """
        assert self.judge_backend is not None

        flog = FancyLogger.get_logger()

        reqs = list(requirements) if requirements else []
        # Merge with strategy-level requirements if they exist
        if hasattr(strategy, "requirements") and strategy.requirements is not None:
            reqs = list(set(reqs + strategy.requirements))

        loop_budget = getattr(strategy, "loop_budget", 1)

        sampled_results: list[ModelOutputThunk] = []
        sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        sampled_actions: list[Component] = []
        sample_contexts: list[Context] = []

        next_action = deepcopy(action)
        next_context = self.ctx

        for loop_count in range(loop_budget):
            flog.info(f"BYO sampling loop {loop_count + 1} of {loop_budget}")

            # generation via callable backend
            result, result_ctx = await self.backend.generate_from_context(
                next_action,
                ctx=next_context,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )
            await result.avalue()

            # re-parse with original action's parser
            result.parsed_repr = action.parse(result)

            # validation using judge backend
            val_scores = await mfuncs.avalidate(
                reqs=reqs,
                context=result_ctx,
                backend=self.judge_backend,
                output=result,
                format=None,
                model_options=model_options,
            )

            constraint_scores = list(zip(reqs, val_scores))

            sampled_results.append(result)
            sampled_scores.append(constraint_scores)
            sampled_actions.append(next_action)
            sample_contexts.append(result_ctx)

            if all(bool(s[1]) for s in constraint_scores):
                flog.info("BYO sampling: SUCCESS")
                assert result._generate_log is not None
                result._generate_log.is_final_result = True

                return SamplingResult(
                    result_index=len(sampled_results) - 1,
                    success=True,
                    sample_generations=sampled_results,
                    sample_validations=sampled_scores,
                    sample_contexts=sample_contexts,
                    sample_actions=sampled_actions,
                )

            failed = [s for s in constraint_scores if not bool(s[1])]
            flog.info(
                f"BYO sampling: FAILED. Valid: {len(constraint_scores) - len(failed)}/{len(constraint_scores)}"
            )

            # attempt repair if the strategy supports it
            if hasattr(strategy, "repair"):
                next_action, next_context = strategy.repair(
                    next_context,
                    result_ctx,
                    sampled_actions,
                    sampled_results,
                    sampled_scores,
                )
            else:
                next_action = deepcopy(action)
                next_context = self.ctx

        # Select best from failures
        if hasattr(strategy, "select_from_failure"):
            best_idx = strategy.select_from_failure(
                sampled_actions, sampled_results, sampled_scores
            )
        else:
            best_idx = 0

        assert sampled_results[best_idx]._generate_log is not None
        sampled_results[best_idx]._generate_log.is_final_result = True

        return SamplingResult(
            result_index=best_idx,
            success=False,
            sample_generations=sampled_results,
            sample_validations=sampled_scores,
            sample_actions=sampled_actions,
            sample_contexts=sample_contexts,
        )

    def validate(
        self,
        reqs: Requirement | list[Requirement],
        *,
        output: CBlock | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
        input: CBlock | None = None,
    ) -> list[ValidationResult]:
        """Validate requirements using the judge backend for LLMaJ, or validation_fn for Python checks."""
        backend = self._resolve_validation_backend(reqs)
        return mfuncs.validate(
            reqs=reqs,
            context=self.ctx,
            backend=backend,
            output=output,
            format=format,
            model_options=model_options,
            generate_logs=generate_logs,
            input=input,
        )

    async def avalidate(
        self,
        reqs: Requirement | list[Requirement],
        *,
        output: CBlock | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        generate_logs: list[GenerateLog] | None = None,
        input: CBlock | None = None,
    ) -> list[ValidationResult]:
        """Async validate using the judge backend for LLMaJ."""
        backend = self._resolve_validation_backend(reqs)
        return await mfuncs.avalidate(
            reqs=reqs,
            context=self.ctx,
            backend=backend,
            output=output,
            format=format,
            model_options=model_options,
            generate_logs=generate_logs,
            input=input,
        )

    def _resolve_validation_backend(
        self, reqs: Requirement | list[Requirement]
    ) -> Backend:
        """Determine which backend to use for validation.

        Uses judge_backend for LLMaJ requirements. Falls back to self.backend
        if all requirements use validation_fn (Python-based checks).
        """
        reqs_list = [reqs] if isinstance(reqs, Requirement) else reqs
        has_llmaj = any(r.validation_fn is None for r in reqs_list)

        if has_llmaj:
            if self.judge_backend is None:
                raise RuntimeError(
                    "A judge_backend is required for LLM-as-a-Judge validation. "
                    "Provide a judge_backend when creating the BYOGeneratorSession, "
                    "or use Requirements with validation_fn for Python-based checks."
                )
            return self.judge_backend

        # All requirements use validation_fn; backend won't actually be called
        # for generation, but it's required by the function signature.
        return self.backend

    def cleanup(self) -> None:
        """Clean up session resources."""
        self.reset()
        if hasattr(self.backend, "close"):
            self.backend.close()  # type: ignore
        if self.judge_backend is not None and hasattr(self.judge_backend, "close"):
            self.judge_backend.close()  # type: ignore


class _InjectionOnlyBackend(Backend):
    """A placeholder backend for injection-only sessions.

    Raises an error if generation is attempted, guiding the user to
    use .inject() or provide a generator_fn.
    """

    async def generate_from_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk, Context]:
        raise RuntimeError(
            "This BYOGeneratorSession has no generator function. "
            "Use .inject() to add pre-computed outputs, or provide a "
            "generator_fn when creating the session."
        )

    async def generate_from_raw(
        self,
        actions: Any,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        raise RuntimeError(
            "This BYOGeneratorSession has no generator function. "
            "Use .inject() to add pre-computed outputs, or provide a "
            "generator_fn when creating the session."
        )


def start_byo_session(
    generator_fn: GeneratorFn | None = None,
    *,
    judge_backend_name: Literal["ollama", "hf", "openai", "watsonx", "litellm"]
    | None = None,
    judge_model_id: str | ModelIdentifier | None = None,
    judge_model_options: dict | None = None,
    ctx: Context | None = None,
    **judge_backend_kwargs,
) -> BYOGeneratorSession:
    """Start a BYO (Bring Your Own) Generator Session.

    Creates a session where generation is handled by a user-provided callable
    or pre-computed outputs, while validation uses a standard Mellea backend.

    Args:
        generator_fn: Callable (action, ctx) -> str. If None, injection-only.
        judge_backend_name: Backend for LLM-as-a-Judge validation.
        judge_model_id: Model identifier for the judge backend.
        judge_model_options: Model options for the judge backend.
        ctx: Context to use. Defaults to SimpleContext().
        **judge_backend_kwargs: Additional kwargs for the judge backend constructor.

    Returns:
        A BYOGeneratorSession instance.
    """
    judge_backend = None
    if judge_backend_name is not None:
        backend_class = backend_name_to_class(judge_backend_name)
        if backend_class is None:
            raise ValueError(
                f"Unknown judge backend: {judge_backend_name}. "
                "Options: ollama, hf, openai, watsonx, litellm"
            )
        judge_backend = backend_class(
            judge_model_id, model_options=judge_model_options, **judge_backend_kwargs
        )

    return BYOGeneratorSession(
        generator_fn=generator_fn, judge_backend=judge_backend, ctx=ctx
    )
