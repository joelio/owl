# Changelog

## Unreleased

### Fixed
- xAI provider now sends the configured model instead of a hardcoded `grok-4.1-fast`; friendly name `grok-agentic` maps to a real API id, and any other name passes through.
- Transient failures (429/502/503 and timeouts) are now retried via `with_retry` across all HTTP providers (OpenAI, Perplexity, DeepSeek, xAI, Gemini). The retry helper existed but was never wired in.
- GitHub issue/comment requests now set a 30s timeout so they can no longer hang indefinitely.

## 0.1.0 (2026-03-05)

Initial release.

- CLI tool `owl` with commands: `ask`, `council`, `council-list`, `models`
- Query multiple LLMs in parallel via Simon Willison's `llm` library
- Direct deep research API support: OpenAI, Perplexity, Gemini, DeepSeek, xAI Grok
- Interactive TUI council selector with `rich`
- GitHub Issues integration — post each LLM response as a separate comment to any repo
- File input support (`-f` / `--file`) and stdin piping
- Config stored in `~/.owl/config.yaml`
