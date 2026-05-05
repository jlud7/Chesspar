# Chesspar

Initial scaffold for the Chesspar web app (Stage 1: Openings Coach).

## Stack

- Next.js (App Router, TypeScript)
- Tailwind CSS
- GitHub Pages deployment via GitHub Actions

## Deploy to GitHub Pages (no local npm required)

1. In GitHub, go to **Settings → Pages**.
2. Under **Build and deployment**, choose **Source: GitHub Actions**.
3. Push to `main`.
4. The workflow `.github/workflows/deploy-pages.yml` installs dependencies, builds, and publishes automatically.

> The app uses static export (`output: "export"`) so it runs on Pages without a Node server.

## Current scope

- Product shell landing page
- 3-stage roadmap visualization
- Architectural foundation for Stage 1 build-out

## Next implementation steps

1. Board + move engine integration
2. Opening curriculum data model
3. Drill + spaced repetition loop
4. Coaching narration service interface
