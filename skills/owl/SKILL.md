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
  --format LEVEL           # brief | standard (default) | detailed
  -s, --synthesise         # Arbiter model reconciles the answers into one
  --arbiter MODEL          # Synthesise with a specific model (implies -s)
  --gh OWNER/REPO          # Post responses to GitHub Issues
  --issue NUMBER           # Post to existing issue (requires --gh)
  # Also accepts stdin: echo "question" | owl ask

owl council                # Interactive picker (/text filters, a/n act on shown rows)
owl council-list           # Show current council members
owl models                 # Show all available models
owl --version

# Group-level flag, goes before the subcommand:
owl -v ask "..."           # Provider stack traces and retries on stderr
```

`--format` sets a word-count target (brief 100-200, standard 250-400,
detailed 600-1000). It is ignored by `openai-deep` and `google-deep`, which
always use the fixed deep research report structure.

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

# Optional. Used by --synthesise; falls back to the first `llm` member.
arbiter:
  name: claude-sonnet-4.6
  source: llm
```

No API keys in config. Standard models use `llm keys set`; deep research APIs use env vars.

## Providers

| Source | Models | Env Var |
|--------|--------|---------|
| `llm` | Any installed llm plugin model | `llm keys set` |
| `openai-deep` | `o3-deep-research`, `o4-mini-deep-research` | `OPENAI_API_KEY` |
| `perplexity` | `sonar-deep-research` | `PERPLEXITY_API_KEY` |
| `google-deep` | `gemini-deep-research` | `GOOGLE_API_KEY` |
| `deepseek` | `deepseek-v4-flash`, `deepseek-v4-pro` | `DEEPSEEK_API_KEY` |
| `xai` | `grok-agentic` (maps to `grok-4.5`) | `XAI_API_KEY` |

Deep research models only appear in `owl models` when their API key env var is set.

## Architecture

```
src/owl/
  cli/main.py        # Click CLI entry point
  config.py          # YAML config load/save
  council.py         # Async parallel dispatch (asyncio.gather, 0.3s stagger)
  synthesis.py       # Arbiter pass that reconciles the council's answers
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
    xai.py           # xAI Responses API + server-side search tools
    retry.py         # Auto-retry on 429/502/503 (2 retries, 2s/5s delays)
    errors.py        # Credential-safe error text (redaction + response body)
```

## Key Patterns

- **Parallel queries:** `asyncio.gather()` with 0.3s stagger delay between launches
- **Graceful errors:** One failed provider doesn't block others; errors shown in red panels
- **OwlResponse:** Dataclass with `model_name`, `source`, `text`, `error`, `citations`, `reasoning`
- **New providers:** Extend `Provider` base class in `providers/`, register in `registry.py`, add model entry in `models.py`
- **GitHub posting:** Uses `GITHUB_TOKEN` env var or `gh auth token` from gh CLI
- **Credential safety:** API keys go in headers, never query strings. All error
  text passes through `errors.describe_error()`, which redacts credential-bearing
  query params. Errors are printed *and* posted to GitHub, so a leak here is public
- **Background jobs:** `openai-deep` and `google-deep` start a job and poll. Override
  `HttpProvider.follow_up()` for APIs whose first reply is a job handle
- **Quiet by default:** `owl/__init__.py` installs a NullHandler so `logger.exception`
  does not reach stderr via `logging.lastResort`. `owl -v` turns it back on
- **Synthesis:** `--synthesise` sends every answer in full to an arbiter model, which is
  told to weigh reasoning rather than count votes. Skipped when fewer than two members
  answered. The arbiter defaults to the first `llm` council member, never a deep research
  one, since those are slow and costly for a reconciling job

## GitHub Integration

```bash
owl ask "question" --gh owner/repo            # Create new issue
owl ask "question" --gh owner/repo --issue 42  # Post to existing issue
```

All responses go in a single consolidated comment, each collapsed into its own
`<details>` block with the model name, optional reasoning in a nested
`<details>`, and citations as a bullet list. Failed members are listed in an
Errors section in the footer. If the combined body would exceed GitHub's
65536-character limit it is split across several comments, and any single
response too large for one comment is truncated to fit.

## Common Tasks

**Add a new provider:** Create `src/owl/providers/newprovider.py` extending `Provider`, add source mapping to `registry.py`, add model entry to `models.py`.

**Debug missing models:** Run `owl models`. Deep research models need env vars set. Standard models need llm plugins installed (e.g. `llm install llm-claude-4`).

**Increase timeout:** `HttpProvider.timeout` is 300s for single-POST providers.
Polling budgets live in the provider: `openai_deep.MAX_RESEARCH_SECONDS` (1800s)
and `google_deep.MAX_RESEARCH_SECONDS` (600s), both with a 5s `POLL_INTERVAL`.

**Test a single provider:** Check the provider's env var is set, then add only that model to config and run `owl ask "test"`.
