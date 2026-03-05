# Parliament of Owls — Design Document

## Overview

**Parliament of Owls** (`owl`) is a Python CLI tool that queries multiple LLMs and deep research APIs in parallel, displaying results in the terminal and optionally posting each response as a comment on a GitHub Issue.

Built on top of Simon Willison's [`llm`](https://github.com/simonw/llm) Python library for standard model access, with direct API integrations for deep research endpoints.

## CLI Commands

```bash
# Query the council
owl ask "What is the best approach to distributed caching?"

# Post to GitHub (creates new issue)
owl ask "..." --gh owner/repo

# Post to existing issue
owl ask "..." --gh owner/repo --issue 42

# Interactive TUI to pick council members
owl council

# Show current council
owl council list

# Show all available models
owl models
```

## Architecture

```
┌─────────────┐
│   owl ask    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐     ┌──────────────────┐
│ Model Discovery │────▶│ ~/.owl/config.yaml│
│ (llm + deep)    │     └──────────────────┘
└──────┬──────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Async Parallel Dispatch       │
├────────┬────────┬───────┬───────┤
│ llm    │ llm    │Direct │Direct │
│ model  │ model  │API    │API    │
│(GPT-5) │(Claude)│(o3-DR)│(Sonar)│
└───┬────┴───┬────┴───┬───┴───┬───┘
    │        │        │       │
    ▼        ▼        ▼       ▼
┌─────────────────────────────────┐
│        Output Formatter         │
├─────────────────┬───────────────┤
│  Terminal (rich) │ GitHub Issues │
│  (default)       │ (--gh flag)   │
└─────────────────┴───────────────┘
```

## Model Sources

### Via `llm` (auto-detected from installed plugins)

Any model installed via `llm install` is available: GPT-5, Claude, Gemini, Mistral, Ollama local models, etc.

### Direct Deep Research APIs (built-in)

| Provider | Model/Agent | API Surface | Notes |
|----------|------------|-------------|-------|
| OpenAI | `o3-deep-research` / `o4-mini-deep-research` | Responses API (separate model) | Sync-ish, uses web_search tool |
| Perplexity | `sonar-deep-research` | `/chat/completions` (separate model) | Same endpoint as standard, different model name |
| Google Gemini | `deep-research-pro-preview-12-2025` | Interactions API (async polling) | Requires `background=True`, poll for results |
| xAI Grok | `grok-4.1` + thinking + agentic search | Agent Tools API | Always reasons; agentic search for web |
| DeepSeek | `deepseek-reasoner` | `/chat/completions` (separate model or flag) | Or use `thinking: {type: enabled}` on `deepseek-chat` |

## Council Configuration

Interactive TUI (`owl council`) with checkbox picker grouped by category. Saved to `~/.owl/config.yaml`:

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
  - name: gemini-deep-research
    source: google-deep
  - name: deepseek-reasoner
    source: deepseek
  - name: grok-agentic
    source: xai
```

## Output

### Terminal (default)

Rich panels per model response, streamed as they arrive.

### GitHub Issues (--gh flag)

One comment per model, posted to any repo. Header identifies the model:

```markdown
## 🦉 GPT-5

Response text here...

---
*Posted by Parliament of Owls*
```

## API Keys

- Standard models: reuses `llm keys` configuration
- Deep research: env vars (`OPENAI_API_KEY`, `PERPLEXITY_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`)
- GitHub: `GITHUB_TOKEN` env var or `gh` CLI auth

## Dependencies

- `llm` — model management + standard queries
- `click` — CLI framework
- `rich` — TUI picker + terminal formatting
- `httpx` — async HTTP for deep research APIs + GitHub API
- `pyyaml` — config file
- `pytest` + `pytest-asyncio` — testing
