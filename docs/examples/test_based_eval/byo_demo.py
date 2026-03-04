"""
BYO Generator Session Demo

Shows two patterns for evaluating externally-generated outputs with Mellea:
1. Callable generator function
2. Pre-computed output injection from simulation data
"""

import json

import mellea
from mellea.stdlib.byo_session import BYOGeneratorSession
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.context import SimpleContext


def demo_callable():
    """Use a callable generator and evaluate with .act()."""
    print("=== Callable Generator ===\n")

    def my_agent(action, ctx):
        # In practice: call an external API, agent framework, etc.
        return f"Generated response for: {str(action)[:50]}"

    session = BYOGeneratorSession(generator_fn=my_agent, ctx=SimpleContext())

    component = SimpleComponent(instruction="What is the capital of France?")
    result = session.act(component, strategy=None)

    print(f"Output: {result}")
    print(f"Context entries: {len(session.ctx.as_list())}\n")
    session.cleanup()


def demo_injection():
    """Inject pre-computed outputs from a simulation file."""
    print("=== Pre-computed Injection ===\n")

    sim_path = "kr_data/agentic-uts/table_generation/sim_2026-02-26_09-08-53_Benchmark_Table_Generation.json"
    with open(sim_path) as f:
        sim_data = json.load(f)

    # Extract the last assistant response from the simulation trace
    messages = sim_data.get("messages", [])
    agent_output = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "assistant"), None
    )

    session = BYOGeneratorSession(ctx=SimpleContext())
    input_action = SimpleComponent(
        instruction="Create a table of paper titles and affiliations"
    )
    thunk = session.inject(agent_output or "No output found", action=input_action)

    print(f"Injected {len(thunk.value or '')} chars into session")
    print(f"Context entries: {len(session.ctx.as_list())}\n")
    session.cleanup()


def demo_with_judge():
    """Full flow: inject output, then evaluate with a judge backend."""
    print("=== Injection + Judge Evaluation ===\n")

    # Create BYO session with a judge (requires running Ollama)
    session = mellea.start_byo_session(
        judge_backend_name="ollama", judge_model_id="granite3.2:8b"
    )

    session.inject(
        "Paris is the capital of France.",
        action=SimpleComponent(instruction="What is the capital of France?"),
    )

    from mellea.core import Requirement

    results = session.validate(Requirement("The answer should mention Paris."))
    print(f"Validation passed: {all(bool(r) for r in results)}\n")
    session.cleanup()


if __name__ == "__main__":
    demo_callable()
    demo_injection()
    # demo_with_judge()  # Uncomment if Ollama is running
