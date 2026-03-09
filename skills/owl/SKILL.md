---
name: owl
description: Use when working with the Parliament of Owls (owl) CLI tool — querying multiple LLMs in parallel, configuring councils, adding providers, debugging model discovery or API issues, or posting results to GitHub Issues.
---

# Parliament of Owls (owl)

CLI tool that queries multiple LLMs in parallel ("a council") and displays rich results. Built on Simon Willison's `llm` library for standard model access, with native deep research API integrations.

## CLI Commands

```
owl ask [PROMPT]           # Query all council members in parallel
  -f, --file FILE_PATH     # Read prompt from file
  --gh OWNER/REPO          # Post responses to GitHub Issues
  --issue NUMBER           # Post to existing issue (requires --gh)
  # Also accepts stdin: echo "question" | owl ask

owl council                # Interactive TUI to select council members
owl council-list           # Show current council members
owl models                 # Show all available models
owl --version
```

## Config

**Location:** `~/.owl/config.yaml` (override with `$OWL_CONFIG_DIR`)

```yaml
council:
  - name: gpt-5
    source: llm
  - name: claude-sonnet-4.6
    source: llm
  - name: o3-deep-research
    source: openai-deep
  - name: sonar-deep-research
    source: perplexity
```

No API keys in config. Standard models use `llm keys set`; deep research APIs use env vars.

## Providers

| Source | Models | Env Var |
|--------|--------|---------|
| `llm` | Any installed llm plugin model | `llm keys set` |
| `openai-deep` | `o3-deep-research`, `o4-mini-deep-research` | `OPENAI_API_KEY` |
| `perplexity` | `sonar-deep-research` | `PERPLEXITY_API_KEY` |
| `google-deep` | `gemini-deep-research` | `GOOGLE_API_KEY` |
| `deepseek` | `deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| `xai` | `grok-agentic` | `XAI_API_KEY` |

Deep research models only appear in `owl models` when their API key env var is set.

## Architecture

```
src/owl/
  cli/main.py        # Click CLI entry point
  config.py          # YAML config load/save
  council.py         # Async parallel dispatch (asyncio.gather, 0.3s stagger)
  models.py          # Model discovery (llm plugins + deep research)
  output.py          # Rich terminal formatting
  github.py          # GitHub Issues integration
  tui.py             # Interactive council selector
  providers/
    base.py          # Provider interface + OwlResponse dataclass
    registry.py      # Source name -> provider routing
    llm_provider.py  # Wraps llm library
    openai_deep.py   # OpenAI Responses API
    perplexity.py    # Perplexity Chat Completions
    google_deep.py   # Gemini Interactions API (async polling)
    deepseek.py      # DeepSeek Chat Completions
    xai.py           # xAI Chat Completions + Agent Tools
    retry.py         # Auto-retry on 429/502/503 (2 retries, 2s/5s delays)
```

## Key Patterns

- **Parallel queries:** `asyncio.gather()` with 0.3s stagger delay between launches
- **Graceful errors:** One failed provider doesn't block others; errors shown in red panels
- **OwlResponse:** Dataclass with `model_name`, `source`, `text`, `error`, `citations`, `reasoning`
- **New providers:** Extend `Provider` base class in `providers/`, register in `registry.py`, add model entry in `models.py`
- **GitHub posting:** Uses `GITHUB_TOKEN` env var or `gh auth token` from gh CLI

## GitHub Integration

```bash
owl ask "question" --gh owner/repo            # Create new issue
owl ask "question" --gh owner/repo --issue 42  # Post to existing issue
```

Each response posted as a separate comment with model name heading, optional reasoning in collapsed `<details>`, and citations as bullet list.

## Common Tasks

**Add a new provider:** Create `src/owl/providers/newprovider.py` extending `Provider`, add source mapping to `registry.py`, add model entry to `models.py`.

**Debug missing models:** Run `owl models`. Deep research models need env vars set. Standard models need llm plugins installed (e.g. `llm install llm-claude-4`).

**Increase timeout:** Default 300s for deep research, 30s polling interval for Gemini. Configured in individual provider files.

**Test a single provider:** Check the provider's env var is set, then add only that model to config and run `owl ask "test"`.
