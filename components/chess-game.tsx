"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { Chess, type Move, type Square } from "chess.js";
import { Chessboard } from "react-chessboard";
import clsx from "clsx";

type Mode = "pass-and-play" | "vs-computer";
type Side = "white" | "black";

type Status =
  | { tone: "normal" | "check"; text: string }
  | { tone: "win" | "draw"; text: string };

const LIGHT_SQ = "#f2e8d5";
const DARK_SQ = "#b58863";

export function ChessGame() {
  const gameRef = useRef<Chess>(new Chess());
  const game = gameRef.current;

  const [fen, setFen] = useState<string>(() => game.fen());
  const [history, setHistory] = useState<Move[]>([]);
  const [mode, setMode] = useState<Mode>("pass-and-play");
  const [playerSide, setPlayerSide] = useState<Side>("white");
  const [orientation, setOrientation] = useState<Side>("white");
  const [selected, setSelected] = useState<Square | null>(null);

  const sync = useCallback(() => {
    setFen(game.fen());
    setHistory(game.history({ verbose: true }) as Move[]);
  }, [game]);

  const isHumanTurn = useCallback(() => {
    if (mode === "pass-and-play") return true;
    const turn: Side = game.turn() === "w" ? "white" : "black";
    return turn === playerSide;
  }, [game, mode, playerSide]);

  const tryMove = useCallback(
    (from: Square, to: Square): Move | null => {
      try {
        const move = game.move({ from, to, promotion: "q" });
        if (move) {
          sync();
          return move as Move;
        }
      } catch {
        /* illegal */
      }
      return null;
    },
    [game, sync]
  );

  const playComputerMove = useCallback(() => {
    if (game.isGameOver()) return;
    const moves = game.moves();
    if (moves.length === 0) return;
    const choice = moves[Math.floor(Math.random() * moves.length)];
    game.move(choice);
    sync();
  }, [game, sync]);

  useEffect(() => {
    if (mode !== "vs-computer") return;
    if (game.isGameOver()) return;
    if (isHumanTurn()) return;
    const t = setTimeout(playComputerMove, 350);
    return () => clearTimeout(t);
  }, [fen, mode, game, isHumanTurn, playComputerMove]);

  function newGame(opts: { mode?: Mode; side?: Side } = {}) {
    game.reset();
    if (opts.mode) setMode(opts.mode);
    if (opts.side) {
      setPlayerSide(opts.side);
      setOrientation(opts.side);
    }
    setSelected(null);
    sync();
  }

  function flipBoard() {
    setOrientation((o) => (o === "white" ? "black" : "white"));
  }

  function undoLastMove() {
    if (history.length === 0) return;
    const undoCount = mode === "vs-computer" && history.length >= 2 ? 2 : 1;
    for (let i = 0; i < undoCount; i++) game.undo();
    setSelected(null);
    sync();
  }

  const legalTargets = useMemo<Square[]>(() => {
    if (!selected) return [];
    return (game.moves({ square: selected, verbose: true }) as Move[]).map((m) => m.to);
  }, [selected, fen, game]);

  const onPieceDrop = ({
    sourceSquare,
    targetSquare,
  }: {
    sourceSquare: string;
    targetSquare: string | null;
  }): boolean => {
    if (!targetSquare || !isHumanTurn()) return false;
    const moved = tryMove(sourceSquare as Square, targetSquare as Square);
    if (moved) setSelected(null);
    return Boolean(moved);
  };

  const onSquareClick = ({ square }: { square: string }) => {
    if (!isHumanTurn()) return;
    const sq = square as Square;
    if (selected && selected !== sq) {
      const moved = tryMove(selected, sq);
      if (moved) {
        setSelected(null);
        return;
      }
    }
    const piece = game.get(sq);
    if (piece && piece.color === game.turn()) {
      setSelected(sq);
    } else {
      setSelected(null);
    }
  };

  const squareStyles = useMemo<Record<string, CSSProperties>>(() => {
    const styles: Record<string, CSSProperties> = {};
    const last = history[history.length - 1];
    if (last) {
      styles[last.from] = { backgroundColor: "rgba(255, 220, 90, 0.30)" };
      styles[last.to] = { backgroundColor: "rgba(255, 220, 90, 0.45)" };
    }
    if (selected) {
      styles[selected] = {
        ...styles[selected],
        backgroundColor: "rgba(110, 200, 90, 0.55)",
      };
    }
    legalTargets.forEach((sq) => {
      const occupied = Boolean(game.get(sq));
      styles[sq] = {
        ...styles[sq],
        background: occupied
          ? "radial-gradient(circle, transparent 58%, rgba(20,20,20,0.45) 60%)"
          : "radial-gradient(circle, rgba(20,20,20,0.45) 22%, transparent 22%)",
      };
    });
    if (game.inCheck()) {
      const turn = game.turn();
      const board = game.board();
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const p = board[r][c];
          if (p && p.type === "k" && p.color === turn) {
            const sq = `${"abcdefgh"[c]}${8 - r}`;
            styles[sq] = {
              ...styles[sq],
              boxShadow: "inset 0 0 0 4px rgba(220,40,40,0.7)",
            };
          }
        }
      }
    }
    return styles;
  }, [selected, legalTargets, history, fen, game]);

  const status: Status = useMemo(() => {
    if (game.isCheckmate()) {
      const winner = game.turn() === "w" ? "Black" : "White";
      return { tone: "win", text: `Checkmate — ${winner} wins` };
    }
    if (game.isStalemate()) return { tone: "draw", text: "Stalemate — draw" };
    if (game.isInsufficientMaterial())
      return { tone: "draw", text: "Draw — insufficient material" };
    if (game.isThreefoldRepetition())
      return { tone: "draw", text: "Draw — threefold repetition" };
    if (game.isDraw()) return { tone: "draw", text: "Draw" };
    const turn = game.turn() === "w" ? "White" : "Black";
    if (game.inCheck()) return { tone: "check", text: `${turn} to move — check!` };
    return { tone: "normal", text: `${turn} to move` };
  }, [fen, game]);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="mx-auto w-full max-w-[640px]">
        <Chessboard
          options={{
            id: "chesspar-board",
            position: fen,
            boardOrientation: orientation,
            onPieceDrop,
            onSquareClick,
            squareStyles,
            darkSquareStyle: { backgroundColor: DARK_SQ },
            lightSquareStyle: { backgroundColor: LIGHT_SQ },
            animationDurationInMs: 200,
            allowDrawingArrows: true,
          }}
        />
      </div>

      <aside className="flex flex-col gap-4">
        <StatusBanner status={status} />
        <Controls
          mode={mode}
          playerSide={playerSide}
          canUndo={history.length > 0}
          onNewGame={newGame}
          onFlip={flipBoard}
          onUndo={undoLastMove}
          onChangeMode={(m) => newGame({ mode: m, side: playerSide })}
          onChangeSide={(s) => newGame({ mode, side: s })}
        />
        <MoveHistory moves={history} />
      </aside>
    </div>
  );
}

function StatusBanner({ status }: { status: Status }) {
  const tone =
    status.tone === "win"
      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
      : status.tone === "draw"
        ? "border-amber-500/40 bg-amber-500/10 text-amber-200"
        : status.tone === "check"
          ? "border-rose-500/40 bg-rose-500/10 text-rose-200"
          : "border-zinc-700 bg-zinc-900/60 text-zinc-200";
  return (
    <div className={clsx("rounded-lg border px-4 py-3 text-sm font-medium", tone)}>
      {status.text}
    </div>
  );
}

function Controls({
  mode,
  playerSide,
  canUndo,
  onNewGame,
  onFlip,
  onUndo,
  onChangeMode,
  onChangeSide,
}: {
  mode: Mode;
  playerSide: Side;
  canUndo: boolean;
  onNewGame: (opts?: { mode?: Mode; side?: Side }) => void;
  onFlip: () => void;
  onUndo: () => void;
  onChangeMode: (m: Mode) => void;
  onChangeSide: (s: Side) => void;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-3">
        <div className="mb-1 text-xs uppercase tracking-wider text-zinc-400">Mode</div>
        <Segmented
          value={mode}
          options={[
            { value: "pass-and-play", label: "Pass & Play" },
            { value: "vs-computer", label: "vs Computer" },
          ]}
          onChange={(v) => onChangeMode(v as Mode)}
        />
      </div>

      {mode === "vs-computer" && (
        <div className="mb-3">
          <div className="mb-1 text-xs uppercase tracking-wider text-zinc-400">Play as</div>
          <Segmented
            value={playerSide}
            options={[
              { value: "white", label: "White" },
              { value: "black", label: "Black" },
            ]}
            onChange={(v) => onChangeSide(v as Side)}
          />
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => onNewGame()}
          className="rounded-md border border-emerald-500/40 bg-emerald-500/15 px-3 py-1.5 text-sm text-emerald-200 hover:bg-emerald-500/25"
        >
          New game
        </button>
        <button
          onClick={onFlip}
          className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-700"
        >
          Flip board
        </button>
        <button
          onClick={onUndo}
          disabled={!canUndo}
          className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Undo
        </button>
      </div>
    </div>
  );
}

function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-zinc-700 bg-zinc-900 p-0.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={clsx(
            "rounded-sm px-3 py-1 text-sm transition",
            value === opt.value
              ? "bg-zinc-700 text-zinc-50"
              : "text-zinc-300 hover:bg-zinc-800"
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function MoveHistory({ moves }: { moves: Move[] }) {
  const pairs: { num: number; white?: Move; black?: Move }[] = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push({ num: i / 2 + 1, white: moves[i], black: moves[i + 1] });
  }
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-2 text-xs uppercase tracking-wider text-zinc-400">Moves</div>
      {pairs.length === 0 ? (
        <div className="text-sm text-zinc-500">No moves yet.</div>
      ) : (
        <ol className="max-h-72 overflow-y-auto pr-1 text-sm tabular-nums">
          {pairs.map((p) => (
            <li key={p.num} className="grid grid-cols-[2.5rem_1fr_1fr] gap-2 py-0.5">
              <span className="text-zinc-500">{p.num}.</span>
              <span className="text-zinc-100">{p.white?.san}</span>
              <span className="text-zinc-300">{p.black?.san ?? ""}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
