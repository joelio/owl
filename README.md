# 🦉 Parliament of Owls

Query multiple LLMs and deep research APIs in parallel. Results displayed in the terminal and optionally posted as comments to GitHub Issues.

Built on [Simon Willison's `llm`](https://github.com/simonw/llm) for standard model access, with direct API integrations for deep research endpoints.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# 1. Configure your council (interactive picker)
owl council

# 2. Ask the council
owl ask "What are the tradeoffs between Redis and Memcached for session storage?"

# 3. Optionally post to a GitHub issue
owl ask "..." --gh owner/repo
owl ask "..." --gh owner/repo --issue 42
```

## Setting Up Models

### Standard Models (via `llm`)

Install `llm` plugins and set their API keys:

```bash
# OpenAI
llm install llm-openai          # included by default with llm
llm keys set openai              # paste your OpenAI API key

# Anthropic (Claude)
llm install llm-anthropic
llm keys set anthropic           # paste your Anthropic API key

# Google Gemini
llm install llm-gemini
llm keys set gemini              # paste your Google AI API key

# Mistral
llm install llm-mistral
llm keys set mistral             # paste your Mistral API key

# Grok (xAI)
llm install llm-grok
llm keys set grok                # paste your xAI API key

# DeepSeek
llm install llm-deepseek
llm keys set deepseek            # paste your DeepSeek API key

# Cohere
llm install llm-command-r
llm keys set cohere              # paste your Cohere API key

# OpenRouter (access dozens of models with one key)
llm install llm-openrouter
llm keys set openrouter          # paste your OpenRouter API key

# Local models via Ollama
llm install llm-ollama
# No key needed - just have Ollama running
```

Verify your installed models:

```bash
llm models                       # list all available models
```

Keys are stored in `~/Library/Application Support/io.datasette.llm/keys.json` (macOS) or `~/.config/io.datasette.llm/keys.json` (Linux).

You can also pass keys via environment variables (e.g. `OPENAI_API_KEY`) or inline with `--key`.

See the full [llm plugin directory](https://llm.datasette.io/en/stable/plugins/directory.html) for more providers.

### Deep Research APIs

Deep research models use direct API calls (not `llm` plugins). Set their keys as environment variables:

```bash
# OpenAI Deep Research (o3-deep-research, o4-mini-deep-research)
export OPENAI_API_KEY=sk-...

# Perplexity Deep Research (sonar-deep-research)
export PERPLEXITY_API_KEY=pplx-...

# Google Gemini Deep Research (Interactions API)
export GOOGLE_API_KEY=AI...

# DeepSeek Reasoner
export DEEPSEEK_API_KEY=sk-...

# xAI Grok Agentic Search
export XAI_API_KEY=xai-...
```

Add these to your `~/.zshrc` or `~/.bashrc` to persist them.

### GitHub Integration

For posting results to GitHub Issues, owl uses your `gh` CLI auth or a `GITHUB_TOKEN`:

```bash
# Option A: gh CLI (recommended)
gh auth login

# Option B: environment variable
export GITHUB_TOKEN=ghp_...
```

## Commands

```bash
owl ask "prompt"                  # Query all council members
owl ask "prompt" --gh owner/repo  # Create a new issue with responses
owl ask "prompt" --gh owner/repo --issue 42  # Post to existing issue
owl council                       # Interactive TUI to select council members
owl council-list                  # Show current council
owl models                        # Show all available models
```

## Council Configuration

Run `owl council` to open the interactive selector:

```
🦉 Parliament of Owls — Select Your Council

 #   Model                      Source        Description
     Standard Models (via llm)
 1  ☑ gpt-5                     llm
 2  ☑ claude-sonnet-4.6         llm
 3  ☐ gemini-2.5-pro            llm
     Deep Research
 4  ☑ o3-deep-research          openai-deep   OpenAI Deep Research
 5  ☑ sonar-deep-research       perplexity    Perplexity Deep Research
 6  ☐ gemini-deep-research      google-deep   Gemini Deep Research Agent
 7  ☐ deepseek-reasoner         deepseek      DeepSeek Reasoner
 8  ☐ grok-agentic              xai           Grok 4.1 agentic search

Enter number to toggle, a=all, n=none, s=save, q=quit:
```

Saved to `~/.owl/config.yaml`.

## How Deep Research Works

Each provider implements deep research differently:

| Provider | What Happens | API |
|----------|-------------|-----|
| **OpenAI** | Separate model (`o3-deep-research`) that searches the web and synthesizes reports | Responses API |
| **Perplexity** | Separate model (`sonar-deep-research`) with multi-step retrieval and citations | `/chat/completions` |
| **Google Gemini** | Async agent that plans, searches, reads, and reasons (can take minutes) | Interactions API |
| **DeepSeek** | Reasoning model with chain-of-thought (`deepseek-reasoner`) | `/chat/completions` |
| **xAI Grok** | Grok 4.1 with agentic web + X search and thinking mode | Chat Completions + tools |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```
