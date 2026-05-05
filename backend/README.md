---
title: ВТОРник Classifier API
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Waste classifier ONNX inference API for ВТОРник app
---

# ВТОРник — Backend API

Server-side ONNX inference for the ВТОРник waste sorting app.
Model is kept inside the Docker image and never reaches the browser.

## Endpoints

- `GET /` — health check, returns `{status, classes, input}`
- `POST /classify` — classify image
  - Headers: `X-API-Key: <token>`
  - Body: `multipart/form-data` with `image` (JPEG/PNG, ≤2 MB)
  - Response: `{ "class": "plastic", "confidence": 0.87, "probs": {...} }`

## Required Space Secrets

Set in **Settings → Variables and secrets**:

| Name | Value |
|---|---|
| `API_KEY` | long random string — must match `API_KEY` in frontend |
| `ALLOWED_ORIGINS` | comma-separated origins, e.g. `https://dismooont.github.io,http://localhost:8080` |

## Local development

```bash
pip install -r requirements.txt
export API_KEY=dev-key-change-me
export ALLOWED_ORIGINS=http://localhost:8080
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Frontend integration

In `index.html`:
```js
const API_URL = 'https://<owner>-<space>.hf.space/classify';
const API_KEY = '<same value as Space secret>';
```
