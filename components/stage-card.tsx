import { ProductStage } from "@/lib/types";

const statusClass: Record<ProductStage["status"], string> = {
  now: "bg-emerald-500/15 text-emerald-300 border-emerald-500/25",
  next: "bg-blue-500/15 text-blue-300 border-blue-500/25",
  later: "bg-violet-500/15 text-violet-300 border-violet-500/25"
};

export function StageCard({ stage }: { stage: ProductStage }) {
  return (
    <article className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-lg font-semibold">{stage.name}</h3>
        <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-wide ${statusClass[stage.status]}`}>
          {stage.status}
        </span>
      </div>
      <p className="mb-4 text-sm text-zinc-300">{stage.headline}</p>
      <ul className="space-y-2 text-sm text-zinc-200">
        {stage.goals.map((goal) => (
          <li key={goal} className="flex gap-2">
            <span className="pt-1 text-emerald-300">•</span>
            <span>{goal}</span>
          </li>
        ))}
      </ul>
    </article>
  );
}
