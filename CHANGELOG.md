# Changelog

## 0.1.0 (2026-03-05)

Initial release.

- CLI tool `owl` with commands: `ask`, `council`, `council-list`, `models`
- Query multiple LLMs in parallel via Simon Willison's `llm` library
- Direct deep research API support: OpenAI, Perplexity, Gemini, DeepSeek, xAI Grok
- Interactive TUI council selector with `rich`
- GitHub Issues integration — post each LLM response as a separate comment to any repo
- File input support (`-f` / `--file`) and stdin piping
- Config stored in `~/.owl/config.yaml`
