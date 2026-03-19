"""A Backend implementation that wraps a user-provided callable for generation."""

from __future__ import annotations

import asyncio
import datetime
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Union

from ..core import CBlock, Component, Context, GenerateLog, ModelOutputThunk, S
from ..core.backend import Backend, BaseModelSubclass, C

GeneratorFn = Callable[[Component | CBlock, Context], str | Awaitable[str]]
"""Type alias for a BYO generator function.

The function receives the action (Component / CBlock) and the current Context,
and returns a string (sync or async) representing the generated output.
"""


class CallableBackend(Backend):
    """A Backend for delegating generation to a user-provided callable.

    This backend wraps a function with the signature:

        (action: Component | CBlock, ctx: Context) -> str | Awaitable[str]

    and adapts it to the Mellea Backend interface. This allows external
    generation systems (agentic frameworks, API wrappers, etc.) to be used
    while preserving core Mellea primitives.
    """

    def __init__(self, generate_fn: GeneratorFn):
        """Initialize with a user-provided generator function."""
        self.generate_fn = generate_fn

    async def generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generate by calling the user-provided function.

        Args:
            action: The Component or CBlock to generate from.
            ctx: The current context.
            format: Ignored (not applicable for callable backends).
            model_options: Ignored (not applicable for callable backends).
            tool_calls: Ignored (not applicable for callable backends).

        Returns:
            A tuple of (ModelOutputThunk, new Context).
        """
        result = self.generate_fn(action, ctx)
        if inspect.isawaitable(result):
            output_str = await result
        else:
            output_str = result

        assert isinstance(output_str, str), (
            f"BYO generator function must return a string, got {type(output_str)}"
        )

        thunk: ModelOutputThunk[C] = ModelOutputThunk(output_str)
        thunk._computed = True
        thunk._action = action

        # Parse via the action's Component if applicable
        if isinstance(action, Component):
            thunk.parsed_repr = action.parse(thunk)

        thunk._generate_log = GenerateLog(
            date=datetime.datetime.now(tz=datetime.timezone.utc),
            prompt=None,
            backend="CallableBackend",
            model_options=model_options,
            model_output=output_str,
            action=action,
            result=thunk,
            is_final_result=False,
        )

        new_ctx = ctx.add(action).add(thunk)
        return thunk, new_ctx

    async def generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        """Not supported for CallableBackend.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "CallableBackend does not support generate_from_raw. "
            "Use generate_from_context instead."
        )
