# BYO Generator Session

The BYO (Bring Your Own) Generator Session allows you to evaluate externally-generated outputs using Mellea's primitives — without requiring generation to go through a Mellea backend.

This is useful for **agentic setups** where a complex agent (with tool calling, multi-step reasoning, etc.) generates responses outside Mellea, but you want to evaluate those responses using Mellea's validation, judging, and sampling infrastructure.

## When to use

- Your generation comes from an external agent framework (e.g., tool-calling agents, research agents)
- You have pre-computed outputs (simulation traces, API logs) to evaluate
- You want Mellea's LLM-as-a-Judge evaluation without Mellea controlling generation

## Quick start

### Callable generator

Wrap any generation function and use it like a normal `MelleaSession`:

```python
from mellea.stdlib.byo_session import BYOGeneratorSession
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.context import SimpleContext

def my_agent(action, ctx):
    # Call your external API, agent framework, etc.
    return call_my_api(str(action))

session = BYOGeneratorSession(generator_fn=my_agent, ctx=SimpleContext())
result = session.act(SimpleComponent(instruction="Summarize this paper"), strategy=None)
```

### Pre-computed injection

Feed existing outputs directly into the session:

```python
session = BYOGeneratorSession(ctx=SimpleContext())
session.inject("The pre-computed agent response", action=my_input_component)
```

### Evaluation with a judge

Combine injection or callable generation with a Mellea judge backend:

```python
import mellea
from mellea.core import Requirement

session = mellea.start_byo_session(
    judge_backend_name="ollama",
    judge_model_id="granite3.2:8b",
)

session.inject("Paris.", action=SimpleComponent(instruction="Capital of France?"))
results = session.validate(Requirement("Answer should be factually correct."))
```

Or use `validation_fn` Requirements for Python-based checks (no judge backend needed):

```python
from mellea.core import Requirement, ValidationResult

session = BYOGeneratorSession(ctx=SimpleContext())
session.inject("42", action=my_component)

results = session.validate(
    Requirement(validation_fn=lambda ctx: ValidationResult(result="42" in str(ctx.last_output())))
)
```

## Architecture

```
                    ┌─────────────────────────────────┐
                    │      BYOGeneratorSession         │
                    │  (extends MelleaSession)         │
                    ├─────────────────────────────────┤
 .act() / .inject() │  CallableBackend  ──> gen_fn()  │  generation
                    │         or                       │
                    │  _InjectionOnlyBackend           │
                    ├─────────────────────────────────┤
 .validate()        │  judge_backend  ──> LLM-as-Judge│  evaluation
                    └─────────────────────────────────┘
```

- **`CallableBackend`** wraps your generator function into the Mellea `Backend` interface
- **`judge_backend`** is a standard Mellea backend (Ollama, OpenAI, etc.) used for LLM-as-a-Judge validation
- All Mellea primitives (Context, Component, Requirement, SamplingResult, ModelOutputThunk) work as normal

## Key files

- `mellea/stdlib/byo_session.py` — `BYOGeneratorSession` class
- `mellea/stdlib/callable_backend.py` — `CallableBackend` adapter
- `mellea/stdlib/session.py` — `start_byo_session()` convenience function
- `docs/examples/test_based_eval/byo_demo.py` — runnable examples

## Use with agentic-uts evaluation data

The `kr_data/agentic-uts/` directory contains evaluation scenarios for agentic systems. To evaluate simulation traces:

```python
import json
from mellea.stdlib.byo_session import BYOGeneratorSession
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.context import SimpleContext

with open("kr_data/agentic-uts/table_generation/sim_....json") as f:
    sim = json.load(f)

# Extract the agent's output from the simulation trace
agent_output = next(
    m["content"] for m in reversed(sim["messages"]) if m["role"] == "assistant"
)

session = BYOGeneratorSession(ctx=SimpleContext())
session.inject(agent_output, action=SimpleComponent(instruction="..."))
# Now validate with a judge or Python requirements
```
