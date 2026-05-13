import Link from "next/link";
import { StageCard } from "@/components/stage-card";
import { productStages } from "@/lib/stages";

type ModeCard = {
  href: string;
  badge: string;
  title: string;
  description: string;
  cta: string;
  tone: "primary" | "secondary" | "tertiary";
};

const MODES: ModeCard[] = [
  {
    href: "/capture",
    badge: "Live",
    title: "Capture a real game",
    description:
      "Phone-as-clock with a camera feed. Tap your half each move; the board is auto-detected, the move inferred, and a clean PGN written for you.",
    cta: "Start a live game →",
    tone: "primary",
  },
  {
    href: "/detect",
    badge: "Test",
    title: "Test on still photos",
    description:
      "Upload a snapshot, watch the auto-detector lock onto the four corners, see per-square occupancy + move inference run end-to-end.",
    cta: "Open the detector →",
    tone: "secondary",
  },
  {
    href: "/play",
    badge: "Practice",
    title: "Play on the screen",
    description:
      "Pass-and-play or vs Stockfish in the browser. Same engine that the inference pipeline will plug into for opening drills.",
    cta: "Open the board →",
    tone: "tertiary",
  },
];

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-6 py-10">
      <header className="mb-10 grid gap-8 rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/80 to-zinc-900/20 p-8 lg:grid-cols-[1.2fr_0.8fr] lg:p-10">
        <div>
          <p className="mb-4 inline-block rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs uppercase tracking-wider text-emerald-200">
            From the board to a PGN — automatically
          </p>
          <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">
            Play OTB. We&apos;ll keep score.
          </h1>
          <p className="mb-7 max-w-2xl text-zinc-300">
            Set your phone over the board. Each time you tap your clock we
            rectify the photo, infer the move, and append it to a clean PGN
            you can paste anywhere. No piece tagging, no NFC, no special
            board — just your set, your phone, and a tap.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/capture"
              className="inline-flex items-center gap-2 rounded-md border border-emerald-500/50 bg-emerald-500/20 px-4 py-2 text-sm font-medium text-emerald-100 hover:bg-emerald-500/30"
            >
              Start a live game →
            </Link>
            <Link
              href="/detect"
              className="inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-800"
            >
              Test on a photo
            </Link>
            <a
              href="#pipeline"
              className="inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-800"
            >
              How it works
            </a>
          </div>
        </div>
        <div className="grid gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 text-sm text-zinc-200">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">
            Why this matters
          </h2>
          <p>
            Casual OTB games go un-recorded because writing the score sheet
            is friction. Chesspar replaces that with a clock you already use.
          </p>
          <p>
            Every game becomes data you can review, share, or feed into
            opening drills.
          </p>
          <p className="rounded-md border border-zinc-700 bg-zinc-950/70 px-3 py-2 text-zinc-300">
            Current pipeline: auto board-detection, calibrated occupancy
            classifier, legal-move inference, optional VLM tie-break.
          </p>
        </div>
      </header>

      <section className="mb-12 grid gap-4 md:grid-cols-3">
        {MODES.map((m) => (
          <ModeTile key={m.href} mode={m} />
        ))}
      </section>

      <section id="pipeline" className="mb-12 rounded-xl border border-zinc-800 bg-zinc-900/40 p-6">
        <h2 className="mb-1 text-xl font-semibold">How a move becomes a PGN</h2>
        <p className="mb-5 text-sm text-zinc-400">
          Constrained search beats raw model intelligence. The vision model
          (if used) only picks from at most a few dozen legal SAN strings —
          it never has to read the position from scratch.
        </p>
        <ol className="grid gap-3 md:grid-cols-5">
          {[
            ["Photo", "Phone camera or test upload."],
            ["Rectify", "Auto-detect 4 corners → homography warp."],
            ["Classify", "Per-square occupancy with per-board baseline."],
            ["Infer", "Diff vs previous FEN ∩ legal moves."],
            ["VLM (opt.)", "Vision tie-break for the long tail."],
          ].map(([title, body], i) => (
            <li
              key={title}
              className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3"
            >
              <div className="mb-1 text-[10px] uppercase tracking-widest text-emerald-300">
                {i + 1}
              </div>
              <div className="mb-1 text-sm font-medium text-zinc-100">
                {title}
              </div>
              <p className="text-xs leading-snug text-zinc-400">{body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section id="roadmap" className="mb-10 grid gap-4 md:grid-cols-3">
        {productStages.map((stage) => (
          <StageCard key={stage.id} stage={stage} />
        ))}
      </section>

      <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-6">
        <h2 className="mb-2 text-lg font-semibold">Built so far</h2>
        <ul className="list-disc space-y-2 pl-5 text-zinc-300">
          <li>
            Auto corner detection + 4-cyclic auto-orientation against a
            starting-position score.
          </li>
          <li>
            Per-board calibrated occupancy classifier (RGB + texture
            nearest-prototype against the starting frame).
          </li>
          <li>
            Legal-move inference: occupancy diff intersected with chess.js
            moves; promotions surfaced as ambiguous candidates.
          </li>
          <li>
            VLM fallback for unmatched frames — Gemini, GPT-4o, or Claude
            Opus, swappable from Settings.
          </li>
          <li>
            Chess.com-style clock with two stacked panels, Fischer
            increment, wake-lock, captures drawer, and PGN export.
          </li>
        </ul>
      </section>
    </main>
  );
}

function ModeTile({ mode }: { mode: ModeCard }) {
  const accent =
    mode.tone === "primary"
      ? "border-emerald-500/50 bg-gradient-to-b from-emerald-500/15 to-zinc-900/60 hover:from-emerald-500/25"
      : mode.tone === "secondary"
        ? "border-sky-500/40 bg-gradient-to-b from-sky-500/10 to-zinc-900/60 hover:from-sky-500/20"
        : "border-zinc-800 bg-zinc-900/60 hover:bg-zinc-900";
  const cta =
    mode.tone === "primary"
      ? "text-emerald-200"
      : mode.tone === "secondary"
        ? "text-sky-200"
        : "text-zinc-200";
  return (
    <Link
      href={mode.href}
      className={`group flex flex-col rounded-xl border p-5 transition-colors ${accent}`}
    >
      <span
        className={`mb-3 inline-flex w-fit rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-widest ${
          mode.tone === "primary"
            ? "border-emerald-500/40 text-emerald-200"
            : mode.tone === "secondary"
              ? "border-sky-500/40 text-sky-200"
              : "border-zinc-700 text-zinc-400"
        }`}
      >
        {mode.badge}
      </span>
      <h3 className="mb-2 text-lg font-semibold text-zinc-100">{mode.title}</h3>
      <p className="mb-4 flex-1 text-sm leading-snug text-zinc-300">
        {mode.description}
      </p>
      <span
        className={`text-sm font-medium ${cta} group-hover:translate-x-0.5 transition-transform`}
      >
        {mode.cta}
      </span>
    </Link>
  );
}
