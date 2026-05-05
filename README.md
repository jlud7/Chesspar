# Chesspar

Initial scaffold for the Chesspar web app (Stage 1: Openings Coach).

## Stack

- Next.js (App Router, TypeScript)
- Tailwind CSS
- GitHub Pages deployment via GitHub Actions

## Run locally

```bash
npm install
npm run dev
```

## Deploy to GitHub Pages

1. In GitHub, go to **Settings → Pages**.
2. Under **Build and deployment**, choose **Source: GitHub Actions**.
3. Push this branch to `main`.
4. The workflow `.github/workflows/deploy-pages.yml` will build and publish `out/` automatically.

> The app uses static export (`output: "export"`) so it can run on Pages without a Node server.

## Current scope

- Product shell landing page
- 3-stage roadmap visualization
- Architectural foundation for Stage 1 build-out

## Next implementation steps

1. Board + move engine integration
2. Opening curriculum data model
3. Drill + spaced repetition loop
4. Coaching narration service interface
