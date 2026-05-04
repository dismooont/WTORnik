# ВТОРник — умная сортировка отходов

Веб-приложение классификации мусора (стекло / металл / бумага / пластик) с серверным ONNX-инференсом.

## Архитектура

```
Frontend (GitHub Pages)  ──POST /classify──►  Backend API (HF Spaces / Render)
  index.html                                     FastAPI + onnxruntime
  без model.onnx                                 model.onnx — приватный
```

Модель **никогда не уходит в браузер**: фронт отправляет JPEG, получает `{class, confidence, probs}`.

## Запуск локально

### 1. Бэкенд

```bash
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # отредактируй API_KEY и ALLOWED_ORIGINS
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Проверка:
```bash
curl http://localhost:8000/                           # {"status":"ok",...}
curl -F image=@test.jpg -H "X-API-Key: dev-key-change-me" http://localhost:8000/classify
```

### 2. Фронтенд

```bash
python -m http.server 8080
# открой http://localhost:8080
```

`API_URL` и `API_KEY` в `index.html` (раздел `// ─── API ───`) уже указывают на `localhost:8000`.

## Деплой

### Бэкенд → Hugging Face Spaces (Docker SDK)

1. Создай **приватный** Space → SDK: Docker.
2. Загрузи содержимое `backend/` (включая `model.onnx`).
3. Settings → Variables and secrets → добавь:
   - `API_KEY` (длинный случайный токен)
   - `ALLOWED_ORIGINS` (`https://<твой-юзернейм>.github.io`)
4. URL Space-а будет `https://<owner>-<space>.hf.space`.

### Фронтенд → GitHub Pages

1. **Удали `model.onnx`** из корня (он остаётся только в `backend/`).
2. В `index.html` поменяй:
   ```js
   const API_URL = 'https://<owner>-<space>.hf.space/classify';
   const API_KEY = '<тот же токен, что в Space Secrets>';
   ```
3. Push на `main`, включи Pages.

## Безопасность модели

- `model.onnx` лежит **только** в приватном бэкенде.
- API защищён `X-API-Key`, CORS-allowlist'ом и rate-limit `30 req/min` на IP.
- Картинки сжимаются до JPEG q=0.9, лимит 2 MB.

**Остаточный риск**: API можно дёргать тысячами запросов и обучить клон-модель (knowledge distillation). Защита — rate-limit, CAPTCHA, биллинг.

## Структура

```
.
├── index.html          # фронт (без модели)
├── backend/
│   ├── app.py          # FastAPI + onnxruntime
│   ├── model.onnx      # ⚠ приватный, не коммитить в публичный репо
│   ├── requirements.txt
│   ├── Dockerfile      # для HF Spaces
│   └── .env.example
└── README.md
```
