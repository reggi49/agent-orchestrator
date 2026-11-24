# Agent Orchestrator

A Python-based multi-agent software orchestrator that turns a development request into a verified plan, implementation proposal, isolated code changes, and a reviewed integration. It also provides a portable project protocol for storing project context, policies, workflow settings, and run artifacts.

## What it does

- Inspects a repository and initializes its `.agent/` project configuration.
- Plans work and generates implementation proposals using a configured language model.
- Runs implementation through the OpenHands Software Agent SDK, with Git branch or worktree isolation.
- Supports an end-to-end orchestration lifecycle, checkpointing, status inspection, and resume.
- Provides human review and approval gates before integration.
- Offers a higher-level CEO program mode for managing objectives and multiple workers.

## Requirements

- Python 3.10 or later.
- Git, for branch/worktree isolation and integration.
- Credentials and access for a supported language-model endpoint.

## Set up

```bash
git clone https://github.com/reggi49/agent-orchestrator.git
cd agent-orchestrator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your model provider, API key, and model name. The checked-in example is configured for 9Router:

```dotenv
LLM_PROVIDER=9router
LLM_API_KEY=your-api-key
LLM_BASE_URL=http://localhost:20128/v1
LLM_MODEL=your-model-name
DEFAULT_PROJECT_PATH=/path/to/your/project
```

For an OpenAI-compatible setup, set `LLM_PROVIDER=openai` and provide the corresponding `LLM_API_KEY` and `LLM_MODEL`. `LLM_BASE_URL` is optional; for 9Router, it defaults to `http://localhost:20128/v1`. The application also accepts `NINE_ROUTER_API_KEY`, `OPENAI_API_KEY`, `OPENAI_MODEL`, and `NINE_ROUTER_BASE_URL` as fallback environment variables.

**Keep `.env` private.** It is excluded from Git by `.gitignore`. The CLI loads its settings at startup, so configure the required API key and model before running commands (including help and project setup commands).

## Quick start

Initialize the target project's portable protocol. Initialization is a dry run unless `--apply` is supplied:

```bash
python main.py init --project /path/to/your/project
python main.py init --project /path/to/your/project --apply
```

Inspect the generated configuration, then run an orchestration request:

```bash
python main.py inspect --project /path/to/your/project
python main.py run \
  --project /path/to/your/project \
  --request "Add a health-check endpoint and tests"
```

`run` does not apply changes by default. To run through the apply/integration path, explicitly pass `--apply`:

```bash
python main.py run \
  --project /path/to/your/project \
  --request "Add a health-check endpoint and tests" \
  --apply
```

The default isolation is a Git worktree. You can choose `--isolation branch`. The approval policy can be set to `safe-auto`, `always-human`, or `never-auto` with `--approval-policy`; review approval requirements can also depend on project policy and risk.

## Workflow and run management

The orchestration lifecycle is designed to separate planning, proposing, implementation, review, and integration. You can also run these stages individually:

```bash
python main.py plan --project /path/to/project --request "Describe the work"
python main.py propose --project /path/to/project --plan latest
python main.py implement --project /path/to/project --proposal latest --apply
python main.py execute --project /path/to/project --proposal latest --apply
python main.py review-execution --project /path/to/project --execution latest --approve
python main.py integrate --project /path/to/project --execution latest --apply
```

Check or resume an orchestration run with its run ID, or use `latest`:

```bash
python main.py status --project /path/to/project --run latest
python main.py resume --project /path/to/project --run latest
```

For longer objectives, `ceo` mode can run or resume a program:

```bash
python main.py ceo --project /path/to/project --request "Improve reliability and test coverage"
```

Run `python main.py <command> --help` for command-specific options. Applying implementation or integration can modify the target repository; review the proposal and execution artifacts before approving or applying changes.

## Project files and artifacts

`init --apply` creates versioned project configuration under `.agent/`, including:

- `manifest.yaml` — project identity and context sources.
- `policy.yaml` — access and safety policy.
- `workflow.yaml` — workflow stages, verification settings, limits, and approvals.
- `artifacts/` — plans, proposals, reports, decisions, and implementation records.

Runtime state, caches, temporary files, and logs are also kept under `.agent/`. The initializer prints recommended `.gitignore` entries for runtime artifacts; it does not edit the target project's root `.gitignore` automatically. Do not ignore the entire `.agent/` directory, because its configuration files are intended to be tracked.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | Model provider name | `openai` |
| `LLM_API_KEY` | Model API key | Required; also accepts provider-specific fallback variables |
| `LLM_MODEL` | Model identifier | Required; `OPENAI_MODEL` is also accepted |
| `LLM_BASE_URL` | Optional API endpoint | 9Router uses `http://localhost:20128/v1` |
| `DEFAULT_PROJECT_PATH` | Optional default target project path | Unset |
| `MAX_REPAIR_ATTEMPTS` | Maximum repair attempts | `3` |
| `MAX_FILE_CHARACTERS` | Maximum file content size handled | `30000` |
| `DEBUG` | Enable full traceback output | `false` |

## Development

Install the project dependencies and run the test suite:

```bash
pip install -r requirements.txt
pytest
```
