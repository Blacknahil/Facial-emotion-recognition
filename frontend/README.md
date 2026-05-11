# Frontend — Emotion demo UI

React 18 + TypeScript + **Vite 6**. Uses the browser **webcam** (mirrored preview) or **file upload** to capture a frame and sends it to the backend as `multipart/form-data`.

## Setup

```bash
npm install
```

If install warnings or broken `node_modules` appear (tar errors, missing Vite files), try a clean install:

```bash
rm -rf node_modules package-lock.json
npm install
```

## Development

Start the **backend** first on port **8000** (see [`../backend/README.md`](../backend/README.md)), then:

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### API proxy

`vite.config.ts` proxies **`/api`** → `http://127.0.0.1:8000`. The client posts to `/api/predict`, which becomes `/predict` on the API.

## Production build

```bash
npm run build
```

Output is written to `dist/`.

Preview the static build locally (API must still be running):

```bash
npm run preview
```

For preview, ensure CORS on the backend includes the preview origin if it differs from port 5173 (defaults in backend already include `4173`).

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Dev server with HMR (default port 5173) |
| `npm run build` | Type-check + bundle to `dist/` |
| `npm run preview` | Serve `dist/` for smoke-testing |

## Stack

- React 18, TypeScript  
- Vite 6, `@vitejs/plugin-react`
