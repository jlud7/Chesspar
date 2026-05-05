import { StageCard } from "@/components/stage-card";
import { productStages } from "@/lib/stages";

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-6 py-12">
      <header className="mb-10">
        <p className="mb-3 inline-block rounded-full border border-zinc-700 bg-zinc-900 px-3 py-1 text-xs uppercase tracking-wider text-zinc-300">
          Stage 1 MVP foundation
        </p>
        <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">Chesspar</h1>
        <p className="max-w-3xl text-zinc-300">
          Personalized chess training built around how you actually play. This first build initializes the product shell,
          ships the stage roadmap, and sets up a clean architecture for the openings coach MVP.
        </p>
      </header>

      <section className="mb-10 grid gap-4 md:grid-cols-3">
        {productStages.map((stage) => (
          <StageCard key={stage.id} stage={stage} />
        ))}
      </section>

      <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-6">
        <h2 className="mb-4 text-xl font-semibold">Immediate Next Build Targets</h2>
        <ol className="list-decimal space-y-3 pl-5 text-zinc-200">
          <li>Implement chessboard + move legality primitives using chess.js and react-chessboard.</li>
          <li>Add opening curriculum schema (24 openings, line graph, mastery fields, spaced repetition metadata).</li>
          <li>Create drill session state machine (line selection, feedback loop, correctness tracking).</li>
          <li>Stub coaching narration service contract for future LLM integration.</li>
        </ol>
      </section>
    </main>
  );
}
