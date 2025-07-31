"""OpenAI implementations for IBM's "starter pack" of Activated LoRAs."""

import openai

from mellea.backends.aloras import Alora
from mellea.backends.openai import OpenAIAlora, OpenAIBackend
from mellea.backends.types import ModelOption
from mellea.helpers.fancy_logger import FancyLogger


class OpenAIConstraintAlora(OpenAIAlora):
    """The [Requirement Checking ALora for Granite 3.2 8B](https://huggingface.co/ibm-granite/granite-3.2-8b-alora-requirement-check) checks if the specified requirement was satisfied by the most recent model generation. Only one requirement is checked at a time."""

    def __init__(
        self, name: str, path: str, generation_prompt: str, backend: OpenAIBackend
    ):
        """Initialize after checking that the backend is correct."""
        assert backend._hf_model_id == "ibm-granite/granite-3.2-8b-instruct"
        super().__init__(name, path, generation_prompt, backend)
        # We do a lot of logging for ALoras because this is an experimental feature. Maybe we should tag these log messages?
        self._logger = FancyLogger.get_logger()

    def generate_using_strings(
        self, input: str, response: str, constraint: str, force_yn: bool = True
    ) -> str:
        """Generates a constraint response from the ALora."""
        # Go ahead and do runtime type-checking because passing CBlocks into this function is a common error.
        assert type(input) is str
        assert type(response) is str
        assert type(constraint) is str

        # Params aren't needed when just getting the backend args.
        backend_model_opts = self._backend._simplify_and_merge(None, False)
        sys_prompt = backend_model_opts.get(ModelOption.SYSTEM_PROMPT, None)

        chat = [
            *([{"role": "system", "content": sys_prompt}] if sys_prompt else []),
            {"role": "user", "content": input},
            {"role": "assistant", "content": response},
        ]

        prompt = self._backend.apply_chat_template(chat)
        prompt += f"\nRequirement: {constraint}<|end_of_text|>\n"
        prompt += self._generation_prompt

        self._logger.debug(f"Prompt for non-cached aLoRA({self.name}):\n{prompt}")

        if force_yn:
            assert hasattr(self._backend, "_tokenizer")
            token_Y = self._backend._tokenizer("Y", add_special_tokens=False)[
                "input_ids"
            ][0]
            token_N = self._backend._tokenizer("N", add_special_tokens=False)[
                "input_ids"
            ][0]
            return (
                self._backend._client.completions.create(
                    model=self.name,
                    prompt=prompt,
                    max_tokens=1,
                    n=1,
                    logit_bias={str(token_Y): 100, str(token_N): 100},
                )
                .choices[0]
                .text
            )
        else:
            return (
                self._backend._client.completions.create(
                    model=self.name, prompt=prompt, max_tokens=1, n=1
                )
                .choices[0]
                .text
            )


def add_granite_aloras(backend: OpenAIBackend):
    """Adds the IBM Granite "starter pack" ALoras to a backend."""
    backend.add_alora(
        OpenAIConstraintAlora(
            name="constraint",
            path="ibm-granite/granite-3.2-8b-alora-requirement-check",
            generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>",
            backend=backend,
        )
    )
