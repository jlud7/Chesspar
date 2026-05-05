import { ProductStage } from "@/lib/types";

export const productStages: ProductStage[] = [
  {
    id: "stage-1",
    name: "Openings Coach",
    status: "now",
    headline: "Ship a standalone training product with coaching + spaced repetition.",
    goals: [
      "Interactive board with drill-mode lines (main lines, sidelines, traps)",
      "24-opening curriculum covering all major structures",
      "Progress + mastery scoring powered by spaced repetition"
    ]
  },
  {
    id: "stage-2",
    name: "Personalized Coaching",
    status: "next",
    headline: "Import real games and train user-specific weak points.",
    goals: [
      "Chess.com/Lichess import pipeline and nightly re-import",
      "Pattern detection (openings, blunder windows, tactical themes)",
      "Auto-generated custom drills from historically weak positions"
    ]
  },
  {
    id: "stage-3",
    name: "AI You",
    status: "later",
    headline: "Generate a shareable clone that plays like the user.",
    goals: [
      "Style-model profile from imported games",
      "Public clone URLs with game history + leaderboard",
      "Strength toggles: honest, sharper, and +200 elo variants"
    ]
  }
];
