# Changelog

## Unreleased

### Changed
- Refactored the HTTP-based providers (OpenAI, Perplexity, DeepSeek, xAI) onto a shared `HttpProvider` base that centralises the API-key check, request/retry/timing flow, response parsing, and error logging. Each provider now only declares its request shape and how to read the reply.

### Added
- Provider failures are now logged with a stack trace (`logger.exception`) instead of being silently swallowed into an error string.
- Malformed/unexpected API responses now produce a clear error (with a payload snippet) instead of a raw `KeyError`/`IndexError`.

### Fixed
- xAI provider now sends the configured model instead of a hardcoded `grok-4.1-fast`; friendly name `grok-agentic` maps to a real API id, and any other name passes through.
- Transient failures (429/502/503 and timeouts) are now retried via `with_retry` across all HTTP providers (OpenAI, Perplexity, DeepSeek, xAI, Gemini). The retry helper existed but was never wired in.
- GitHub issue/comment requests now set a 30s timeout so they can no longer hang indefinitely.
- Gemini deep-research polling is now bounded by an overall wall-clock budget (`MAX_RESEARCH_SECONDS`, 10 min) rather than a raw attempt count, and each poll is retried on transient failures so a brief blip no longer discards in-progress research.

## 0.1.0 (2026-03-05)

Initial release.

- CLI tool `owl` with commands: `ask`, `council`, `council-list`, `models`
- Query multiple LLMs in parallel via Simon Willison's `llm` library
- Direct deep research API support: OpenAI, Perplexity, Gemini, DeepSeek, xAI Grok
- Interactive TUI council selector with `rich`
- GitHub Issues integration — post each LLM response as a separate comment to any repo
- File input support (`-f` / `--file`) and stdin piping
- Config stored in `~/.owl/config.yaml`
