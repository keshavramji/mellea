# Comributor Guide

This document is a guide to new contributors. Each section covers a different section of the codebase.

## Remember to Propagate model_options

`mellea` users should be able to override the default model options for a single model call, or for a sequence of model calls. Due to the architecture of `mellea`, the user almost never has access to an actual backend object (e.g., an OpenAI-compatible `client` or a transformers `PretrainedModel`).

To ensure that `mellea` users should be able to override the default model options for a single model call, every Backend generation function accepts a model_options dictionary as a named parameter. This is done like so:

```
def generate(my, args, go here, *, model_options: Optional[dict] = None)
```

The `*` forces the model_options to be passed by keyword.

Every function that calls a Backend generation function should *also* accept the `model_options: Optional[dict] = None` keyword argument (and should pass that argument along to the generation function). And every function that calls one of *those* functions should do the same. And so on.

So: whenever adding a public method that the user might call, look through all of the code in the method. If you call any functions which take `model_options`, then you too should be taking `model_options` and passing it along to that function.

## Adding Components to the Standard Library

The purpose of this section is to provide background and context for contributors of new `Component`s.

Anything that will eventually be provided as a model input is either a `CBlock` or a `Component`.

`Component` is used for composite types, such as an Instruction or a Requirement, while `CBlock` is used is a wrapper around simple strings. (A good rule of thumb is that if something might end up rendering as anything other than a single `Span` in the eventual `Span` backends, then it should be implemented using `Component`.)

Components are currently rendered using the `TemplateFormatter`. This means you need to do one of two things when adding a new component:
1. Add a template with that Component's name to the `templates/prompts/default` directory. This is the preferred method for contributions to stdlib.
2. Or, define the `Component.template` field with your `jinja2` template. In this case, that single string will be used for *all* models.

Eventually the TemplateFormatter will support for model-specific prompts and will also support non-`stdlib` template resolution.

Make sure to add smoke tests for at least the ollama backend.

After adding a new Component, consider whether it makes sense to add a corresponding "verb-style convienance wrapper" to the `m.stdlib.session.MelleaSession` class.

## Adding a FormatterBackend

Any non-Span backend should be a FormatterBackend.

When generating, you need to make sure that you parse the model options and handle m-specific things appropriately for your model backend.

Formatters both print Components and parse results into CBlocks/Components. The final thing that a FormatterBackend generate call should do is:

1. Create a ModelOutputThunk whose value is the **raw** output of the model -- the string corresponding to the exact set of tokens that were generated (or as close as you can get to that for remote backends).
2. Then call `self.formatter.parse()` on that `ModelOutputThunk` and return the result.
