import Link from "next/link";
import { StageCard } from "@/components/stage-card";
import { productStages } from "@/lib/stages";

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-6 py-12">
      <header className="mb-12 grid gap-8 rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/80 to-zinc-900/20 p-8 lg:grid-cols-[1.2fr_0.8fr] lg:p-10">
        <div>
          <p className="mb-4 inline-block rounded-full border border-zinc-700 bg-zinc-900 px-3 py-1 text-xs uppercase tracking-wider text-zinc-300">
            Stage 1: Openings Coach
          </p>
          <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">
            Stop forgetting opening lines after 3 moves.
          </h1>
          <p className="mb-7 max-w-2xl text-zinc-300">
            Chesspar helps you train openings like a habit loop: learn a line, test recall on the board, and reinforce what you miss.
            This build includes a complete playable board and engine foundation for the coach experience.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/play"
              className="inline-flex items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/15 px-4 py-2 text-sm font-medium text-emerald-200 hover:bg-emerald-500/25"
            >
              Start playing →
            </Link>
            <Link
              href="/capture"
              className="inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-800"
            >
              Live game + capture
            </Link>
            <a
              href="#roadmap"
              className="inline-flex items-center rounded-md border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-800"
            >
              View roadmap
            </a>
          </div>
        </div>
        <div className="grid gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 text-sm text-zinc-200">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">Why this matters</h2>
          <p>Most players know what to study, but forget positions under pressure.</p>
          <p>Chesspar is designed around repetition from realistic move trees — not static flash cards.</p>
          <p className="rounded-md border border-zinc-700 bg-zinc-950/70 px-3 py-2 text-zinc-300">
            Current build: interactive board, legal move handling, Stockfish play mode, and persistent game state.
          </p>
        </div>
      </header>

      <section className="mb-10 grid gap-4 md:grid-cols-3">
        {[
          ["1", "Learn", "Study core lines, side-lines, and tactical traps in context."],
          ["2", "Recall", "Play through lines from memory with immediate correctness feedback."],
          ["3", "Reinforce", "Revisit weak positions via spaced repetition until they stick."],
        ].map(([step, title, body]) => (
          <article key={title} className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
            <p className="mb-2 text-xs uppercase tracking-wider text-emerald-300">Step {step}</p>
            <h2 className="mb-2 text-lg font-semibold">{title}</h2>
            <p className="text-sm text-zinc-300">{body}</p>
          </article>
        ))}
      </section>

      <section id="roadmap" className="mb-10 grid gap-4 md:grid-cols-3">
        {productStages.map((stage) => (
          <StageCard key={stage.id} stage={stage} />
        ))}
      </section>

      <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-6">
        <h2 className="mb-4 text-xl font-semibold">Immediate Build Targets</h2>
        <ol className="list-decimal space-y-3 pl-5 text-zinc-200">
          <li>Ship opening curriculum schema (24 openings, line graph, mastery fields, spaced repetition metadata).</li>
          <li>Create drill session state machine (line selection, feedback loop, correctness tracking).</li>
          <li>Introduce first-time onboarding and guided board UX for new users.</li>
          <li>Stub coaching narration service contract for future LLM integration.</li>
          <li>Add session analytics events (line attempts, recall accuracy, completion rate).</li>
        </ol>
      </section>

      <section className="mt-8 rounded-xl border border-zinc-800 bg-zinc-900/40 p-6">
        <h2 className="mb-2 text-lg font-semibold">Built so far</h2>
        <ul className="list-disc space-y-2 pl-5 text-zinc-300">
          <li>Playable board with legal moves, promotions, check, and game-over states.</li>
          <li>Pass-and-play plus vs-computer mode with adjustable engine strength.</li>
          <li>Move history, capture tracking, and shareable position/game export tools.</li>
          <li>Local save/resume so users can continue where they left off.</li>
        </ul>
      </section>
    </main>
  );
}
