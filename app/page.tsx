import Link from "next/link";

const PIPELINE = [
  {
    n: "01",
    title: "Lock the board",
    body:
      "One photo. Florence-2 finds the playing surface. The four corners cache for the rest of the session.",
  },
  {
    n: "02",
    title: "Tap to capture",
    body:
      "Each move, a 5-frame burst picks the sharpest still. No motion blur, no manual focus.",
  },
  {
    n: "03",
    title: "Diff-first inference",
    body:
      "We look at which 2–4 squares changed, not all 64. Legal moves whose template matches get scored.",
  },
  {
    n: "04",
    title: "VLM tiebreak",
    body:
      "On the rare ambiguous capture, Gemini 2.5 Pro adjudicates against the candidate list.",
  },
  {
    n: "05",
    title: "Tap to confirm",
    body:
      "If we still aren't sure (<3% of moves), pick from the top two. Never a silent wrong move.",
  },
] as const;

export default function HomePage() {
  return (
    <main className="bg-zinc-950 text-zinc-100">
      <header className="absolute inset-x-0 top-0 z-10 flex items-center justify-between px-6 py-5 sm:px-10">
        <div className="text-sm font-semibold tracking-tight text-zinc-50">
          Chesspar
        </div>
        <Link
          href="/capture"
          className="rounded-full bg-white/10 px-3 py-1 text-[11px] font-medium uppercase tracking-widest text-zinc-200 backdrop-blur transition hover:bg-white/15"
        >
          Open capture →
        </Link>
      </header>

      <section className="relative flex min-h-[92vh] flex-col items-center justify-center overflow-hidden px-6 pt-24 pb-12 text-center sm:px-10">
        <BackgroundGlow />
        <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-emerald-300">
          Phone over the board · automatic PGN
        </p>
        <h1 className="mt-5 max-w-3xl text-balance text-[clamp(2.75rem,8.5vw,5.75rem)] font-semibold leading-[0.95] tracking-tight">
          Play OTB.
          <br />
          <span className="text-zinc-400">We&apos;ll keep score.</span>
        </h1>
        <p className="mx-auto mt-6 max-w-xl text-pretty text-[15px] leading-relaxed text-zinc-400">
          Set your phone above the board. Tap once at the start, then tap
          after each move. Chesspar reads the move and writes a clean PGN —
          targeting 99% per-move accuracy with a calibrated tap-to-confirm
          fallback.
        </p>
        <div className="mt-8 flex flex-col items-center gap-3 sm:flex-row sm:gap-4">
          <Link
            href="/capture"
            className="inline-flex items-center gap-2 rounded-full bg-emerald-500/90 px-6 py-3 text-base font-semibold text-emerald-950 transition hover:bg-emerald-400"
          >
            Start a live game →
          </Link>
          <Link
            href="/play"
            className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-5 py-2.5 text-sm font-medium text-zinc-100 transition hover:bg-white/10"
          >
            Practice board
          </Link>
        </div>
        <a
          href="#how"
          className="mt-12 text-[11px] uppercase tracking-[0.3em] text-zinc-500 transition hover:text-zinc-300"
        >
          How it works ↓
        </a>
      </section>

      <section id="how" className="mx-auto max-w-6xl px-6 py-24 sm:px-10">
        <div className="text-[11px] font-semibold uppercase tracking-[0.32em] text-emerald-300">
          Under the hood
        </div>
        <h2 className="mt-3 max-w-3xl text-balance text-[clamp(1.75rem,4vw,2.75rem)] font-semibold leading-tight tracking-tight text-zinc-50">
          Diff-first hybrid.
          <span className="text-zinc-500">
            {" "}
            Legality and beam search beat raw model intelligence.
          </span>
        </h2>
        <ol className="mt-10 grid gap-3 md:grid-cols-5">
          {PIPELINE.map((step) => (
            <li
              key={step.n}
              className="rounded-3xl border border-white/5 bg-white/5 p-5"
            >
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.3em] text-emerald-300">
                {step.n}
              </div>
              <div className="mb-1 text-base font-semibold text-zinc-50">
                {step.title}
              </div>
              <p className="text-[12px] leading-snug text-zinc-400">{step.body}</p>
            </li>
          ))}
        </ol>
        <div className="mt-12 flex justify-center">
          <Link
            href="/capture"
            className="inline-flex items-center gap-2 rounded-full bg-emerald-500/90 px-5 py-3 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-400"
          >
            Start your first game →
          </Link>
        </div>
      </section>

      <footer className="border-t border-white/5 px-6 py-10 sm:px-10">
        <div className="mx-auto flex max-w-6xl items-center justify-between text-[11px] uppercase tracking-widest text-zinc-500">
          <span>Chesspar · OTB scoresheet, automatic</span>
          <Link
            href="/play"
            className="text-zinc-400 transition hover:text-zinc-200"
          >
            Play board →
          </Link>
        </div>
      </footer>
    </main>
  );
}

function BackgroundGlow() {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 -z-0 overflow-hidden"
    >
      <div className="absolute left-1/2 top-1/3 h-[60vh] w-[60vh] -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-500/10 blur-[100px]" />
      <div className="absolute left-[20%] top-[60%] h-[40vh] w-[40vh] -translate-x-1/2 -translate-y-1/2 rounded-full bg-sky-500/8 blur-[120px]" />
    </div>
  );
}
