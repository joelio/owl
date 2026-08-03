# Changelog

## Unreleased

### Added
- `--synthesise` (`-s`) adds an arbiter pass: one model reads every council answer in full and produces a single reconciled answer, printed above the individual responses and posted uncollapsed at the top of the GitHub comment. `--arbiter MODEL` picks the model explicitly; otherwise an optional `arbiter:` config entry is used, falling back to the first `llm` council member.
  - The arbiter is told to weigh reasoning rather than count votes, and to report confidence, material disagreement, and claims only one member made. Practitioners report that majority voting plateaus while synthesis over full reasoning recovers correct answers even when a council agrees on a wrong one, so agreement is treated as evidence to weigh rather than as the result.
  - Skipped when fewer than two members answered, rather than spending a request restating a single answer. Deep research models are never chosen as the default arbiter: slow and costly for a reconciling job.

### Fixed
- `owl council` no longer tracebacks on Ctrl-D or Ctrl-C; both cancel and leave the council unchanged.
- `owl council` is usable with a large catalogue. With OpenRouter installed the list runs to several hundred models, and every keystroke reprinted all of them. The table is now capped at 30 rows with a count of what is hidden, and `/text` filters by model name or source. `a` and `n` act on what is shown, so `/:free` then `a` selects the free models.

- Realigned three providers with their current APIs after checking each contract against primary vendor documentation. Combined with the OpenAI fix in 0.2.0, four of the five deep research providers were calling endpoints, models or parameters that no longer exist.
  - **DeepSeek**: `deepseek-reasoner` was discontinued on 2026-07-24 along with `deepseek-chat`. The default is now `deepseek-v4-flash` with `thinking` enabled explicitly, since V4 makes thinking opt-in where the old reasoner model had it on implicitly. The retired name is remapped rather than left to fail.
  - **xAI**: the provider sent `grok-4.1-fast`, which is not a current model id, to `/v1/chat/completions` with `tools: [{"type": "web_search"}]` and a `chain_limit` parameter that has never existed in the xAI API. Chat Completions is documented as a deprecated endpoint supporting function calling only, so the agentic search this provider exists for was never reachable. It now posts to the Responses API as `grok-4.5` with server-side `web_search` and `x_search` tools.
  - **Gemini**: the poll loop waited for a `done` boolean and read text from `response.outputParts[]`. Interactions actually report a `status` string and carry text in `steps[].content[].text`, so a completed research run would never have been recognised. Terminal failure statuses are now surfaced as errors with the reason instead of being polled until timeout.

### Changed
- Retries now honour a `Retry-After` header when the server sends one, cover 500 and 504 alongside 429/502/503, and retry dropped connections (`ConnectError`, `ReadError`, `WriteError`, `RemoteProtocolError`) rather than only timeouts. Backoff is jittered so a whole council rate-limited at once does not march back in lockstep, and an absurd `Retry-After` is ignored rather than stalling the query.
- The test suite no longer sleeps through real backoff: 197 tests run in under a second, down from 178 in 16.5 seconds.
- Provider request shapes are now pinned by contract tests, so a rename or endpoint change fails a test rather than silently returning nothing.

## 0.2.0 (2026-08-03)

### Security
- The Gemini API key is no longer sent as a `?key=` query parameter. httpx puts the full request URL into `str(HTTPStatusError)`, that string was used as the error text, and error text is posted to GitHub issues — so a Gemini auth failure during `owl ask --gh owner/repo` could publish the key. It now travels in the `x-goog-api-key` header, and all error text passes through a redaction step that masks credential-bearing query parameters.

### Fixed
- OpenAI deep research now works at all. It was sending the `web_search` tool, which deep research models reject with a 400, so `o3-deep-research` and `o4-mini-deep-research` never returned an answer. It now sends `web_search_preview`, runs the request in background mode, and polls to completion within a 30-minute budget rather than blocking on a 5-minute request timeout.
- Console markup no longer leaks into terminal output. Labels were built as `[dim]Reasoning:[/dim]` and handed to `Markdown()`, which does not parse console markup, so the tags were printed verbatim on every response carrying reasoning or citations.
- Error panels no longer swallow bracketed text. Errors were interpolated into a markup string, so an API payload snippet containing `[...]` had it parsed away as a style tag.
- Responses too large for one GitHub comment are truncated instead of being posted at full size. GitHub rejects an oversized body with a 422, and because responses are posted in one pass, a single huge answer previously lost them all.
- `--issue` without `--gh` is now rejected rather than silently ignored.
- xAI provider sends the configured model instead of a hardcoded `grok-4.1-fast`; friendly name `grok-agentic` maps to a real API id, and any other name passes through.
- Transient failures (429/502/503 and timeouts) are retried via `with_retry` across all HTTP providers. The retry helper existed but was never wired in.
- GitHub issue/comment requests set a 30s timeout so they can no longer hang indefinitely.
- Gemini deep research polling is bounded by an overall wall-clock budget (`MAX_RESEARCH_SECONDS`, 10 min) rather than a raw attempt count, and each poll is retried on transient failures so a brief blip no longer discards in-progress research.

### Added
- `-v` / `--verbose` shows provider stack traces and retry activity on stderr.
- Error messages now carry the reason the API gave (invalid model, exhausted quota) rather than the bare status line.
- Malformed or unexpected API responses produce a clear error with a payload snippet instead of a raw `KeyError` / `IndexError`.

### Changed
- Provider stack traces are no longer printed by default. Nothing configured logging, so records reached `logging.lastResort` and every failure dumped a traceback on top of the error panel. A `NullHandler` now suppresses that; use `-v` to get it back. This reverses part of the unreleased behaviour that followed the previous change.
- Refactored the HTTP-based providers (OpenAI, Perplexity, DeepSeek, xAI) onto a shared `HttpProvider` base that centralises the API-key check, request/retry/timing flow, response parsing and error logging. Each provider now declares only its request shape and how to read the reply.
- `ruff` is pinned to a compatible release and the lint rules are selected explicitly. Ruff's default rule set expands between minor versions, which had turned CI red with no code change.
- The version is single-sourced from `owl.__version__`; `pyproject.toml` reads it via `[tool.setuptools.dynamic]`.

## 0.1.0 (2026-03-05)

Initial release.

- CLI tool `owl` with commands: `ask`, `council`, `council-list`, `models`
- Query multiple LLMs in parallel via Simon Willison's `llm` library
- Direct deep research API support: OpenAI, Perplexity, Gemini, DeepSeek, xAI Grok
- Interactive TUI council selector with `rich`
- GitHub Issues integration — post each LLM response as a separate comment to any repo
- File input support (`-f` / `--file`) and stdin piping
- Config stored in `~/.owl/config.yaml`
