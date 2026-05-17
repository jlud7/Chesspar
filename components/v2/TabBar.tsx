"use client";

export type Tab = "clock" | "score" | "camera";

const ITEMS: Array<{ id: Tab; label: string; icon: string }> = [
  { id: "clock", label: "Clock", icon: "◔" },
  { id: "score", label: "Score", icon: "▤" },
  { id: "camera", label: "Camera", icon: "◉" },
];

export function TabBar({
  active,
  onPick,
  pendingCount,
  failurePending,
}: {
  active: Tab;
  onPick: (t: Tab) => void;
  /** Pending capture count — shown as a small badge on the Score tab. */
  pendingCount?: number;
  /** When true, the Score badge turns amber and pulses to signal a stuck queue. */
  failurePending?: boolean;
}) {
  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-3 border-t border-white/10 bg-zinc-950/95 pb-[env(safe-area-inset-bottom)] backdrop-blur"
      style={{ fontFamily: "var(--font-ui, ui-sans-serif)" }}
    >
      {ITEMS.map((item) => {
        const isActive = item.id === active;
        const showBadge = item.id === "score" && (pendingCount ?? 0) > 0;
        const badgeTone = failurePending
          ? "bg-amber-300 text-amber-950"
          : "bg-emerald-400 text-emerald-950";
        const badgeAnim = failurePending
          ? { animation: "chesspar-pulse 1.6s ease-in-out infinite" as const }
          : undefined;
        return (
          <button
            key={item.id}
            onClick={() => onPick(item.id)}
            className={[
              "flex flex-col items-center justify-center gap-0.5 py-2.5 text-[10px] uppercase tracking-[0.22em] transition",
              isActive
                ? "text-emerald-300"
                : "text-zinc-500 hover:text-zinc-200",
            ].join(" ")}
          >
            <span className="relative text-[18px] leading-none">
              {item.icon}
              {showBadge && (
                <span
                  className={`absolute -right-2.5 -top-1.5 inline-flex h-4 min-w-[1rem] items-center justify-center rounded-full px-1 text-[9px] font-semibold ${badgeTone}`}
                  style={badgeAnim}
                >
                  {failurePending ? "!" : pendingCount}
                </span>
              )}
            </span>
            <span>{item.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
