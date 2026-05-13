import Link from "next/link";

type ModeCard = {
  href: string;
  badge: string;
  title: string;
  description: string;
  tone: "primary" | "secondary" | "tertiary";
};

const MODES: ModeCard[] = [
  {
    href: "/capture",
    badge: "Live",
    title: "Capture a real game",
    description:
      "Phone over the board. Tap your clock each move. We rectify the photo, infer the move, and write a clean PGN.",
    tone: "primary",
  },
  {
    href: "/detect",
    badge: "Test",
    title: "Test on photos",
    description:
      "Upload a still and watch the auto-detector lock onto the four corners. Inspect occupancy and move inference end-to-end.",
    tone: "secondary",
  },
  {
    href: "/play",
    badge: "Practice",
    title: "Play on the screen",
    description:
      "Pass-and-play or vs Stockfish in the browser. Same engine the inference pipeline plugs into.",
    tone: "tertiary",
  },
];

const PIPELINE = [
  ["Photo", "Phone camera or test upload."],
  ["Rectify", "Auto-detect the 4 corners, homography warp."],
  ["Classify", "Per-square occupancy, calibrated to your board."],
  ["Infer", "Diff vs previous FEN ∩ legal moves."],
  ["VLM", "Vision tie-break — only on the long tail."],
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
          Open app →
        </Link>
      </header>

      <Hero />

      <section className="mx-auto max-w-6xl px-6 pb-24 pt-12 sm:px-10">
        <SectionHeading
          eyebrow="Three ways in"
          title={
            <>
              Pick a mode.
              <span className="text-zinc-500"> Same pipeline behind each.</span>
            </>
          }
        />
        <div className="mt-10 grid gap-4 md:grid-cols-3">
          {MODES.map((m) => (
            <ModeTile key={m.href} mode={m} />
          ))}
        </div>
      </section>

      <section
        id="how"
        className="mx-auto max-w-6xl px-6 py-24 sm:px-10"
      >
        <SectionHeading
          eyebrow="Under the hood"
          title={
            <>
              How a move becomes a PGN.
              <span className="text-zinc-500">
                {" "}
                Constrained search beats raw model intelligence.
              </span>
            </>
          }
        />
        <ol className="mt-10 grid gap-3 md:grid-cols-5">
          {PIPELINE.map(([title, body], i) => (
            <li
              key={title}
              className="rounded-3xl border border-white/5 bg-white/5 p-5"
            >
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.3em] text-emerald-300">
                {String(i + 1).padStart(2, "0")}
              </div>
              <div className="mb-1 text-base font-semibold text-zinc-50">
                {title}
              </div>
              <p className="text-[12px] leading-snug text-zinc-400">{body}</p>
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

function Hero() {
  return (
    <section className="relative flex min-h-[92vh] flex-col items-center justify-center overflow-hidden px-6 pt-24 pb-12 text-center sm:px-10">
      <BackgroundGlow />
      <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-emerald-300">
        From the board to a PGN
      </p>
      <h1 className="mt-5 max-w-3xl text-balance text-[clamp(2.75rem,8.5vw,5.75rem)] font-semibold leading-[0.95] tracking-tight">
        Play OTB.
        <br />
        <span className="text-zinc-400">We&apos;ll keep score.</span>
      </h1>
      <p className="mx-auto mt-6 max-w-xl text-pretty text-[15px] leading-relaxed text-zinc-400">
        Set your phone over the board. Each tap of your clock captures a
        photo, infers the move, and appends to a clean PGN. No piece tags.
        No NFC. No special board.
      </p>
      <div className="mt-8 flex flex-col items-center gap-3 sm:flex-row sm:gap-4">
        <Link
          href="/capture"
          className="inline-flex items-center gap-2 rounded-full bg-emerald-500/90 px-6 py-3 text-base font-semibold text-emerald-950 transition hover:bg-emerald-400"
        >
          Start a live game →
        </Link>
        <Link
          href="/detect"
          className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-5 py-2.5 text-sm font-medium text-zinc-100 transition hover:bg-white/10"
        >
          Test on a photo
        </Link>
      </div>
      <a
        href="#how"
        className="mt-12 text-[11px] uppercase tracking-[0.3em] text-zinc-500 transition hover:text-zinc-300"
      >
        How it works ↓
      </a>
    </section>
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

function SectionHeading({
  eyebrow,
  title,
}: {
  eyebrow: string;
  title: React.ReactNode;
}) {
  return (
    <div>
      <div className="text-[11px] font-semibold uppercase tracking-[0.32em] text-emerald-300">
        {eyebrow}
      </div>
      <h2 className="mt-3 max-w-3xl text-balance text-[clamp(1.75rem,4vw,2.75rem)] font-semibold leading-tight tracking-tight text-zinc-50">
        {title}
      </h2>
    </div>
  );
}

function ModeTile({ mode }: { mode: ModeCard }) {
  const accent =
    mode.tone === "primary"
      ? "border-emerald-400/30 bg-gradient-to-b from-emerald-500/15 to-zinc-900/40 hover:from-emerald-500/25"
      : mode.tone === "secondary"
        ? "border-sky-400/25 bg-gradient-to-b from-sky-500/10 to-zinc-900/40 hover:from-sky-500/20"
        : "border-white/5 bg-white/5 hover:bg-white/10";
  const badgeAccent =
    mode.tone === "primary"
      ? "border-emerald-400/40 text-emerald-200"
      : mode.tone === "secondary"
        ? "border-sky-400/40 text-sky-200"
        : "border-white/10 text-zinc-400";
  const ctaAccent =
    mode.tone === "primary"
      ? "text-emerald-200"
      : mode.tone === "secondary"
        ? "text-sky-200"
        : "text-zinc-200";
  return (
    <Link
      href={mode.href}
      className={`group relative flex flex-col overflow-hidden rounded-3xl border p-6 transition-colors ${accent}`}
    >
      <span
        className={`mb-4 inline-flex w-fit rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.3em] ${badgeAccent}`}
      >
        {mode.badge}
      </span>
      <h3 className="mb-2 text-[22px] font-semibold tracking-tight text-zinc-50">
        {mode.title}
      </h3>
      <p className="mb-5 flex-1 text-[14px] leading-snug text-zinc-300">
        {mode.description}
      </p>
      <span
        className={`text-sm font-medium ${ctaAccent} transition-transform group-hover:translate-x-0.5`}
      >
        Open →
      </span>
    </Link>
  );
}
