# Changelog

## 0.1.0 (unreleased)

- `AyInboxToolkit`: builds LangChain tools from `GET /v1/ai/tools`, with `toolsets`, `include` and `exclude` filters and sync and async loading.
- `AyInboxTool`: runs `POST /v1/ai/tools/{name}/call` on the sync and async paths, returns the API's fenced `result_text` to the model and the whole `data` object as the artifact.
- API errors become tool errors carrying the API's `code` and `request_id`.
- Retries on `429`, `502`, `503` and `504` honouring `Retry-After`; a tool call is retried on a server error only when it carries an `idempotency_key`.
