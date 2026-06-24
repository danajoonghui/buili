# Buili

Buili is a construction AI MVP2 implementation based on the supplied BuildLens AI plan. It includes:

- Next.js PWA frontend for desktop web and phone mode
- FastAPI backend with project, upload, job, issue, RFI, overlay, and report APIs
- SQLite local mode with PostgreSQL-ready SQLAlchemy models
- Ingestion, media, RAG, inference orchestration, and report worker entrypoints
- Model gateway skeleton for KAIST A6000 deployment
- Public-data training pipeline that forces GPU 7 via `CUDA_VISIBLE_DEVICES=7`

The app is intentionally evidence-first: it generates issue candidates with confidence, citations, crops/frames, and model run ids for human review.

## Quick Start

```bash
conda activate cjh_buili
python -m services.api.buili.main
```

In another shell:

```bash
cd apps/web
npm run dev
```

Default URLs:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Training

The training scripts force GPU 7. They fetch a small public DocLayNet sample when available and train a compact layout classifier smoke model.

```bash
conda activate cjh_buili
python ml/download_public_data.py --limit 80
python ml/train_doc_layout_smoke.py --epochs 3
```

Artifacts are written to `data/artifacts/`.

## Render Deployment

The repository includes `render.yaml` for a two-service Render deployment:

- `buili-web`: Next.js PWA web app
- `buili-api`: FastAPI core API with a persistent disk for uploaded plans/media/reports
- `buili-postgres`: PostgreSQL database, initialized with the `vector` extension
- `buili-cache`: Render Key Value for queue/cache wiring

Create a Render Blueprint from this repo. The web service calls `/api/*`; the Next.js route handler proxies those requests to the FastAPI service over Render's private network, so the browser does not need a hardcoded public API origin.

For a production deployment, set the optional API environment variables in the Render dashboard:

- `BUILI_MODEL_GATEWAY_URL` for a KAIST GPU, SGLang, vLLM, or OpenAI-compatible model gateway
- `BUILI_R2_ACCOUNT_ID`, `BUILI_R2_ACCESS_KEY_ID`, `BUILI_R2_SECRET_ACCESS_KEY`, and `BUILI_R2_BUCKET` if object storage is wired in

Render free web services are suitable for demos, but production file persistence requires the API service disk in the Blueprint. Render persistent disks require a paid service instance, so `buili-api` uses the `starter` plan.

Manual Render API service disk settings:

- Disk name: `buili-api-storage`
- Mount path: `/var/data`
- Size: `1 GB`
- Environment variable: `BUILI_STORAGE_ROOT=/var/data/buili/storage`

Single-domain Render mode is supported: if the Python API service owns a domain such as
`https://buili.onrender.com`, `GET /` serves the Buili web UI and `/api/*` is accepted as an alias
for the FastAPI routes. This lets uploads and report downloads work with `BUILI_PUBLIC_BASE_URL=/api`.
