# Docs AI Chat Widget (Epic 13)

A lightweight, zero-dependency chat widget that adds knowledge graph-powered AI
answers to the docs.mareanalytica.com Wiki.js site. Queries are sent to
RAGAnything, which combines vector search and graph traversal for accurate,
context-rich responses.

## Architecture

```
docs.mareanalytica.com (Wiki.js)
  |
  |  inject.html loaded in browser
  |
  v
raganything.mareanalytica.com (Traefik IngressRoute)
  |
  v
raganything.memory.svc.cluster.local:9621 (LightRAG)
  |
  +-- Qdrant (vector search)
  +-- Neo4j  (knowledge graph)
  +-- PostgreSQL (KV + doc status)
```

## Installation

### 1. Ensure the IngressRoute is applied

The RAGAnything IngressRoute exposes the API externally at
`raganything.mareanalytica.com`. It lives at:

```
liquidmetal/manifests/base/raganything/ingress.yaml
```

Apply it:

```bash
kubectl apply -f liquidmetal/manifests/base/raganything/ingress.yaml
```

Verify:

```bash
curl -s https://raganything.mareanalytica.com/health
```

### 2. Inject the widget into Wiki.js

1. Log into Wiki.js at `docs.mareanalytica.com` as an administrator.
2. Go to **Site Administration** > **Theme** > **Code Injection**.
3. In the **Body HTML** field, paste the entire contents of `inject.html`.
4. Click **Apply**.

The chat widget will appear as a floating button in the bottom-right corner of
every page.

### 3. Verify

Open any docs page. You should see a blue circular button in the bottom-right.
Click it to open the chat panel and type a question.

## Configuration

The only configuration value is the RAGAnything URL. In `inject.html`, edit the
variable near the top of the script:

```javascript
var MA_RAG_URL = 'https://raganything.mareanalytica.com';
```

Change this if you move RAGAnything to a different host or subpath.

### Query mode

The widget sends queries with `mode: 'mix'` (hybrid graph + vector). To change
the query mode, edit the `mode` value in the `JSON.stringify` call inside the
`send()` function. Supported modes from LightRAG:

- `local` -- entity-centric, graph-only
- `global` -- community-centric, graph-only
- `hybrid` -- local + global graph
- `naive` -- vector-only (no graph)
- `mix` -- graph + vector hybrid (recommended)

## Files

| File | Purpose |
|------|---------|
| `chat-widget.html` | Full standalone HTML preview for development and testing |
| `inject.html` | Minified snippet to paste into Wiki.js Code Injection |

## How it works

1. The widget creates a floating action button and a chat panel using vanilla
   JavaScript. No frameworks or external dependencies.

2. When the user submits a question, the widget sends a POST request to
   `/query/stream` on the RAGAnything API with the query and mode.

3. RAGAnything returns a streaming NDJSON response. Each line is a JSON object
   with a `response` field containing a text token.

4. The widget reads the response stream using the Fetch API's ReadableStream
   reader and appends tokens to the assistant message in real time.

5. CORS is handled by a Traefik middleware (`raganything-cors`) that allows
   requests from `https://docs.mareanalytica.com`.

## Theming

The widget uses CSS custom properties and respects:

- `prefers-color-scheme: dark` media query
- Wiki.js `.dark` class on the document root
- Wiki.js `[data-theme="dark"]` attribute

No manual theme toggle is needed; the widget follows the system/Wiki.js theme
automatically.

## Network requirements

The widget makes cross-origin requests from `docs.mareanalytica.com` to
`raganything.mareanalytica.com`. This requires:

1. **IngressRoute** -- `ingress.yaml` exposes RAGAnything externally.
2. **CORS middleware** -- `raganything-cors` in the same manifest allows the
   docs origin.
3. **NetworkPolicy** -- `networkpolicy.yaml` allows Traefik pods in
   `kube-system` to reach RAGAnything pods in `memory`.
