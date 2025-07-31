"""Simple cli runner for m decompose."""

import json
import os
from typing import Annotated

import typer


def decompose(
    query: Annotated[
        str | None,
        typer.Option(help="Path to file containing one or more task queries."),
    ] = None,
    out_dir: Annotated[
        str, typer.Option(help="Path to file containing one or more task queries.")
    ] = ".",
    dry_run: Annotated[
        bool, typer.Option(help="Only decompose the task, skip execution.")
    ] = False,
    print_only: Annotated[
        bool, typer.Option(help="Only print outputs to console, do not save any files.")
    ] = False,
    generate_py_files: Annotated[
        bool, typer.Option(help="Save M program files in the out_dir under m_programs/")
    ] = False,
    model_id: Annotated[
        str | None,
        typer.Option(
            help="If set, overrides both decomposor_model_id and executor_model_id."
        ),
    ] = None,
    decomposor_model_id: Annotated[
        str | None,
        typer.Option(
            "-dm",
            help="Model ID to use for decomposer backend session. Is overridden by `model_id` if set",
        ),
    ] = None,
    executor_model_id: Annotated[
        str | None,
        typer.Option(
            "-em",
            help="Model ID to use for executor backend session. Is overridden by `model_id` if set",
        ),
    ] = None,
    backend_type: Annotated[
        str | None,
        typer.Option(
            help="If set, overrides both decomposor_backend_type and executor_backend_type."
        ),
    ] = None,
    decomposor_backend_type: Annotated[
        str | None,
        typer.Option(
            help="Backend type for decomposor session (e.g., huggingface, ollama, vllm)."
        ),
    ] = "ollama",
    executor_backend_type: Annotated[
        str | None,
        typer.Option(
            help="Backend type for executor session (e.g., huggingface, ollama, vllm)."
        ),
    ] = "ollama",
):
    """Run the M prompt decomposition pipeline. Uses `mistral-small:latest` running on Ollama.

    If no `QUERY` value is provided, the command will prompt for input from stdin.
    """

    # Import here so that imports (especially torch) don't slow down other cli commands and during cli --help.
    from .utils import create_model, generate_python_template, run_pipeline

    # If model_id is set, override both decomposor_model_id and executor_model_id
    if model_id is not None:
        decomposor_model_id = model_id
        executor_model_id = model_id

    # If backend_type is set, override both decomposor_backend_type and executor_backend_type
    if backend_type is not None:
        decomposor_backend_type = backend_type
        executor_backend_type = backend_type

    decompose_session = create_model(
        model_id=decomposor_model_id,
        backend_type=decomposor_backend_type,  # type: ignore
    )
    execute_session = create_model(
        model_id=executor_model_id,
        backend_type=executor_backend_type,  # type: ignore
    )

    all_results = []

    if query:
        try:
            with open(query) as f:
                content = f.read()
            task_sections = content.split("# Task")[1:]
            tasks = [section.strip() for section in task_sections]
            for i, task_input in enumerate(tasks):
                result = run_pipeline(
                    task_input,
                    index=i,
                    decompose_session=decompose_session,
                    execute_session=execute_session,
                    out_dir=out_dir,
                    dry_run=dry_run,
                    print_only=print_only,
                )
                all_results.append(result)
                if generate_py_files:
                    generate_python_template(
                        subtask_data=result["executed_results"]["subtask_data"],
                        output_dir=out_dir,
                        index=i,
                    )
            if not print_only:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "combined_results.json"), "w") as f:
                    json.dump(all_results, f, indent=2)
                print(
                    f"\nSaved combined results to: {os.path.join(out_dir, 'combined_results.json')}"
                )

        except Exception as e:
            print(f"Error reading query file: {e}")
            exit(1)
    else:
        task_input = typer.prompt(
            "Hi Welcome to use the M - Task Decomposition Pipeline! What can I do for you? \nUser Request: "
        )
        result = run_pipeline(
            task_input,
            index=None,
            decompose_session=decompose_session,
            execute_session=execute_session,
            out_dir=out_dir,
            dry_run=dry_run,
            print_only=print_only,
        )
        if generate_py_files:
            generate_python_template(
                subtask_data=result["executed_results"]["subtask_data"],
                output_dir=out_dir,
            )
        if not print_only:
            with open(os.path.join(out_dir, "combined_results.json"), "w") as f:
                json.dump([result], f, indent=2)
            print(
                f"\nSaved combined result to: {os.path.join(out_dir, 'combined_results.json')}"
            )


# # Basic dry run, no file input
# m decompose --dry_run --print_only

# # Basic dry run, no file output
# m decompose --dry_run --print_only

# # Full run, only print to terminal
# m decompose --print_only

# # Normal full run with outputs
# m decompose --out_dir outputs/

# Run with generation of m programs based on the executed results
# m decompose --generate-py-file --out_dir output/
