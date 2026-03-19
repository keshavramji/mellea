"""
Simple TestBasedEval Demo - Inline Instantiation

This demo shows how to create and run TestBasedEval tests directly in code,
without using files or the CLI.
"""

from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext
from mellea.stdlib.test_based_eval import TestBasedEval

import mellea
from mellea.backends.huggingface import LocalHFBackend


def main():
    test = TestBasedEval(
        source="demo",
        name="conciseness_test",
        instructions="The response should be brief, no more than one sentence.",
        inputs=["What is the capital of France?", "What is 2 + 2?"],
        targets=[["Paris."], ["4"]],
        test_id="demo_001",
    )

    print(f"Created test: {test.name}")
    print(f"Number of inputs: {len(test.inputs)}\n")

    # Setup sessions
    gen_backend = LocalHFBackend(
        model_id="IBM_GRANITE_4_MICRO_3B",
        model_options={ModelOption.MAX_NEW_TOKENS: 4096},
    )
    judge_backend = LocalHFBackend(
        model_id="MS_PHI_4_14B", model_options={ModelOption.MAX_NEW_TOKENS: 512}
    )

    gen_session = mellea.MelleaSession(backend=gen_backend, ctx=SimpleContext())
    judge_session = mellea.MelleaSession(backend=judge_backend, ctx=SimpleContext())

    # Run test
    passed = 0
    for idx, input_text in enumerate(test.inputs):
        print(f"Input: {input_text}")

        # Generate response
        output = gen_session.act(input_text)
        print(f"Generated: {output}")

        # Evaluate with judge
        test.set_judge_context(
            input_text=input_text,
            prediction=str(output),
            targets_for_input=test.targets[idx],
        )

        judge_result = judge_session.act(test)
        print(f"Judge: {judge_result}")

        # Check if passed (simple check - you can parse the judge output more thoroughly)
        if "score" in str(judge_result).lower() and "1" in str(judge_result):
            passed += 1
            print("Result: PASS\n")
        else:
            print("Result: FAIL\n")

        gen_session.reset()
        judge_session.reset()

    print(f"Final: {passed}/{len(test.inputs)} passed")

    # Cleanup
    gen_session.cleanup()
    judge_session.cleanup()


if __name__ == "__main__":
    main()
