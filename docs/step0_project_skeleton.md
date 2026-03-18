# Step 0: Project Skeleton

## What was built

The minimal installable package structure:

```
pyproject.toml                  # Package metadata, dependencies, entry point
co_scientist/
├── __init__.py                 # Package init, version string
├── cli.py                      # Typer CLI with `run` command
└── config.py                   # RunConfig (Pydantic model)
```

## Design decisions

**Why Typer for CLI?** The architecture spec calls for Typer + Rich. Typer gives us type-safe argument parsing with auto-generated help text, and it integrates natively with Rich for styled terminal output.

**Why Pydantic for config?** `RunConfig` validates all CLI arguments in one place. The `pattern` constraint on `mode` means invalid modes fail immediately with a clear error, not somewhere deep in the pipeline. The `task_output_dir` property centralizes the path convention (`outputs/RNA__translation_efficiency_muscle/`) so every later step uses the same directory without reimplementing the logic.

**Entry point:** `pyproject.toml` registers `co-scientist` as a console script pointing to `co_scientist.cli:app`. After `pip install -e .`, the command is available globally.

## CLI interface

```bash
co-scientist --version              # prints version
co-scientist --help                 # shows commands
co-scientist run --help             # shows all options
co-scientist run <dataset_path>     # runs the pipeline
```

Options mirror the architecture spec Section 11.1:
- `--mode auto|interactive` (interactive allows conversational Q&A at decision points)
- `--budget` (iteration steps)
- `--max-cost` (LLM dollar limit)
- `--no-search`, `--resume`, `--seed`, `--output-dir`

## Verification

```bash
pip install -e .
co-scientist run RNA/translation_efficiency_muscle
co-scientist run expression/cell_type_classification_segerstolpe
```

Both print the config summary and a placeholder message.
