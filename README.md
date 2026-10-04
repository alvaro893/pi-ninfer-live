# pi-ninfer-live

A [pi](https://pi.dev) provider extension for a local **NInfer** [OpenAI-compatible inference server](https://github.com/Neroued/ninfer).

Like pi's built-in llama.cpp integration, it discovers models **live** from the
server on every model refresh — no `models.json` editing, and server-side
changes (model swapped, context window changed) propagate automatically.

## Features

- **Live model discovery** — the catalog is fetched from `/v1/models` on every
  refresh; `max_model_len` maps to the model's context window.
- **Load-aware** — NInfer serves one resident model per server, so a model only
  appears in the picker while the server reports it. Server down or model
  unloaded → the model disappears from the picker (same as llama.cpp hiding
  unloaded models). The last good catalog is restored offline across restarts.
- **Thinking levels** — models are registered as reasoning-capable with the
  `qwen-chat-template` thinking format, so pi sends
  `chat_template_kwargs: { enable_thinking: ... }` and the built-in
  thinking-level picker (`/model` → `:<level>`) works:
  `off | minimal | low | medium | high`.
- **Streaming** — delegates to pi's built-in `openai-completions`
  implementation (message conversion, tool handling, usage accounting).

## Install

```bash
pi install git:github.com/alvaro893/pi-ninfer-live
```

or from a local checkout:

```bash
pi install ./pi-ninfer-live
```

## Configure

The server URL (and optional API key) is resolved in this order — first match
wins:

1. **Environment variables** — `NINFER_BASE_URL`, `NINFER_API_KEY`
2. **Config file** — `~/.pi/agent/ninfer.json`
3. **`/ninfer <url>`** — set and persist the URL at runtime
4. **Default** — `http://127.0.0.1:8080`

### Config file (recommended)

Create `~/.pi/agent/ninfer.json` once and it persists across sessions:

```json
{
  "baseUrl": "http://192.168.0.62:8080",
  "apiKey": "none"
}
```

`apiKey` is optional — omit it (or use `"none"`) unless the server was started
with `--api-key`.

### At runtime

```
/ninfer                        # refresh the catalog from the configured server
/ninfer http://192.168.0.62:8080   # set + persist the server URL, then refresh
```

`/ninfer` contacts the server, reports which model is loaded, and refreshes the
catalog (same role as the built-in `/llama`). Startup refreshes are
offline-only by design, so run `/ninfer` after a server change (new model
loaded, URL moved) to pick it up.

### Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `NINFER_BASE_URL` | Server root, e.g. `http://192.168.0.62:8080` | `http://127.0.0.1:8080` |
| `NINFER_API_KEY` | Bearer token, if the server was started with `--api-key` | `none` |

Then pick the model in pi:

```
/model ninfer/qwen3.8-27b          # thinking off
/model ninfer/qwen3.8-27b:medium   # thinking on (medium effort)
```

Any non-`off` level enables thinking; the Qwen chat template has a single
thinking mode, so levels only gate it on/off.

## Notes

- Max output tokens are fixed at 32768 in
  [`extensions/ninfer-live.ts`](extensions/ninfer-live.ts) because
  `/v1/models` does not report a per-model output limit.
- The model list fetch has a 4 s timeout; a slow or missing server leaves the
  previously stored catalog in place.
