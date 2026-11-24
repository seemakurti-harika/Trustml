# Phase 6 — Integration QA Audit

## Scope
Frontend route-to-API integration review and backend regression verification for TrustML Phase 5.

## Frontend route inventory
- Overview (`/`)
- Dataset Explorer (`/datasets`)
- Model Lab (`/model-lab`)
- Reliability Analysis (`/reliability`)
- Prediction Playground (`/playground`)
- Drift Monitor (`/drift`)
- Experiment History (`/history`)
- About TrustML (`/about`)

The route UI and most page components currently live in `frontend/src/App.tsx` rather than separate files in `src/pages/`. This is runnable in principle, but the single-file organization is not ideal for long-term maintenance. Refactoring is recommended after dependency installation/build verification, to reduce risk of breaking current behavior.

## API integration inventory
Frontend API service functions include health, datasets, upload summaries, training, experiments, predictions, uncertainty, selective prediction, robustness, drift, explanations, calibration comparison, conformal analysis, and advanced drift analysis.

Backend routes include:
- `GET /api/health`
- `GET /api/datasets`
- `GET /api/datasets/uploads`
- `POST /api/datasets/upload`
- `GET /api/datasets/{dataset_id}/summary`
- `POST /api/train`
- `GET /api/experiments`
- `GET /api/experiments/{experiment_id}`
- `POST /api/predict`
- `POST /api/calibration/analyze`
- `POST /api/calibration/compare`
- `POST /api/uncertainty/analyze`
- `POST /api/selective-prediction/analyze`
- `POST /api/robustness/test`
- `POST /api/drift/analyze`
- `GET /api/models/{model_id}/schema`
- `GET /api/models/{model_id}/explain`
- `POST /api/ensemble/analyze`
- `POST /api/conformal/analyze`
- `POST /api/drift/advanced`

## Backend regression test
Verified on the extracted Phase 5 package:
`13 passed, 2 warnings in 2.34s`

Warnings: FastAPI's deprecated `on_event` startup hook. Recommend migrating to lifespan context manager in a follow-up maintenance patch.

## Frontend build verification
`tsc --noEmit` was attempted. It could not complete a meaningful type-check because npm dependencies and React typings are not installed. The errors are dominated by missing module/type declarations (`react`, `react-router-dom`, `lucide-react`, `recharts`, `react/jsx-runtime`). `npm install` was attempted but was interrupted after the environment's time limit while waiting on package registry access.

Therefore:
- Frontend build: **not verified**
- Browser run/E2E: **not verified**
- Do not report the UI as build-clean until dependencies install and `npm run build` passes.

## Risks / next actions
1. Install frontend dependencies locally (`npm install`) in an internet-enabled Windows/macOS/Linux environment.
2. Run `npm run build` and fix any TypeScript/Vite errors.
3. Run both servers and manually walk all eight routes.
4. Verify uploaded CSV target selection, numeric/categorical validation, and edge cases.
5. Refactor `App.tsx` into page/component modules without changing functionality.
6. Add frontend tests and an end-to-end smoke test for upload → train → predict → calibration → conformal → drift.
7. Fix FastAPI startup event deprecation warning using a lifespan handler.

## Honesty note
This audit documents tests and checks that were actually attempted. It does not imply a browser session, npm build, or end-to-end test was completed.
