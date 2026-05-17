"use client";

/**
 * Game-mode picker shown after the camera is attached and before
 * "Capture starting position". Untimed is the default; a chess.com-
 * style grid offers Bullet/Blitz/Rapid/Classical presets plus a custom
 * base+increment input. Last choice persists in localStorage so a
 * repeat player skips re-picking.
 */

import { useEffect, useMemo, useState } from "react";
import {
  TIME_CONTROL_PRESETS,
  type GameMode,
  type TimeControlPreset,
} from "@/lib/v2/types";

const STORAGE_KEY = "chesspar.v2.gameMode";

export function loadStoredGameMode(): GameMode {
  if (typeof window === "undefined") return { kind: "untimed" };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { kind: "untimed" };
    const parsed = JSON.parse(raw) as GameMode;
    if (parsed.kind === "untimed") return parsed;
    if (
      parsed.kind === "timed" &&
      Number.isFinite(parsed.baseMs) &&
      Number.isFinite(parsed.incrementMs)
    ) {
      return parsed;
    }
  } catch {
    /* fall through */
  }
  return { kind: "untimed" };
}

export function saveGameMode(mode: GameMode) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(mode));
  } catch {
    /* ignore quota */
  }
}

export function GameModePicker({
  initial,
  onConfirm,
}: {
  initial: GameMode;
  onConfirm: (mode: GameMode) => void;
}) {
  const [mode, setMode] = useState<GameMode>(initial);
  const [customBaseMin, setCustomBaseMin] = useState<string>(() =>
    initial.kind === "timed" ? String(Math.round(initial.baseMs / 60000)) : "5",
  );
  const [customIncSec, setCustomIncSec] = useState<string>(() =>
    initial.kind === "timed" ? String(Math.round(initial.incrementMs / 1000)) : "0",
  );

  useEffect(() => setMode(initial), [initial]);

  const grouped = useMemo(() => {
    const groups: Record<string, TimeControlPreset[]> = {};
    for (const p of TIME_CONTROL_PRESETS) {
      (groups[p.group] ??= []).push(p);
    }
    return groups;
  }, []);

  const matchingPresetId = mode.kind === "timed"
    ? TIME_CONTROL_PRESETS.find(
        (p) => p.baseMs === mode.baseMs && p.incrementMs === mode.incrementMs,
      )?.id ?? null
    : null;

  const applyCustom = () => {
    const baseMs = Math.max(0, Number(customBaseMin) * 60_000);
    const incrementMs = Math.max(0, Number(customIncSec) * 1000);
    if (!Number.isFinite(baseMs) || !Number.isFinite(incrementMs)) return;
    setMode({ kind: "timed", baseMs, incrementMs });
  };

  const confirm = () => {
    saveGameMode(mode);
    onConfirm(mode);
  };

  return (
    <div className="absolute inset-0 z-40 flex flex-col bg-zinc-950/95 backdrop-blur">
      <div className="flex shrink-0 items-center justify-between border-b border-white/5 px-4 py-3 sm:px-6">
        <span className="text-[11px] font-semibold uppercase tracking-[0.3em] text-zinc-400">
          Game mode
        </span>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-4 sm:px-6">
        <UntimedRow
          active={mode.kind === "untimed"}
          onClick={() => setMode({ kind: "untimed" })}
        />
        <div className="mt-6 space-y-5">
          {(["Bullet", "Blitz", "Rapid", "Classical"] as const).map((group) => (
            <div key={group}>
              <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.28em] text-zinc-500">
                {group}
              </h3>
              <div className="grid grid-cols-3 gap-2">
                {(grouped[group] ?? []).map((p) => (
                  <PresetButton
                    key={p.id}
                    preset={p}
                    active={matchingPresetId === p.id}
                    onClick={() =>
                      setMode({
                        kind: "timed",
                        baseMs: p.baseMs,
                        incrementMs: p.incrementMs,
                      })
                    }
                  />
                ))}
              </div>
            </div>
          ))}
          <div>
            <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.28em] text-zinc-500">
              Custom
            </h3>
            <div className="flex items-end gap-2">
              <NumberField
                label="min"
                value={customBaseMin}
                onChange={setCustomBaseMin}
              />
              <span className="pb-2 text-zinc-500">+</span>
              <NumberField
                label="sec"
                value={customIncSec}
                onChange={setCustomIncSec}
              />
              <button
                onClick={applyCustom}
                className="ml-2 mb-0.5 rounded-full border border-white/15 bg-white/5 px-3 py-2 text-[11px] uppercase tracking-widest text-zinc-200 transition hover:bg-white/10"
              >
                Set
              </button>
            </div>
          </div>
        </div>
      </div>
      <div className="border-t border-white/5 bg-zinc-950/95 px-4 py-4 sm:px-6">
        <button
          onClick={confirm}
          className="flex h-12 w-full items-center justify-center rounded-full bg-emerald-500 text-sm font-semibold text-emerald-950 shadow-xl transition hover:bg-emerald-400"
        >
          Use {describeMode(mode)}
        </button>
      </div>
    </div>
  );
}

function UntimedRow({
  active,
  onClick,
}: {
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "flex w-full items-center justify-between rounded-2xl border px-4 py-3 text-left transition",
        active
          ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-100"
          : "border-white/10 bg-white/5 text-zinc-200 hover:bg-white/10",
      ].join(" ")}
    >
      <div>
        <div className="text-sm font-semibold">Untimed (open game)</div>
        <div className="mt-0.5 text-[11px] text-zinc-400">
          No clock. Sides show <span className="font-mono">YOUR TURN</span> /{" "}
          <span className="font-mono">WAITING</span>.
        </div>
      </div>
      {active && <span className="text-[10px] uppercase tracking-widest text-emerald-300">Selected</span>}
    </button>
  );
}

function PresetButton({
  preset,
  active,
  onClick,
}: {
  preset: TimeControlPreset;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "rounded-xl border px-3 py-3 text-center transition",
        active
          ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-100"
          : "border-white/10 bg-white/5 text-zinc-200 hover:bg-white/10",
      ].join(" ")}
    >
      <div className="font-mono text-sm">{preset.label}</div>
    </button>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (s: string) => void;
}) {
  return (
    <label className="flex flex-1 flex-col gap-1">
      <span className="text-[10px] uppercase tracking-widest text-zinc-500">{label}</span>
      <input
        inputMode="numeric"
        type="number"
        min={0}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 font-mono text-sm text-zinc-100 focus:border-emerald-400/40 focus:outline-none"
      />
    </label>
  );
}

function describeMode(mode: GameMode): string {
  if (mode.kind === "untimed") return "untimed";
  const m = Math.round(mode.baseMs / 60_000);
  return `${m} + ${Math.round(mode.incrementMs / 1000)}`;
}
