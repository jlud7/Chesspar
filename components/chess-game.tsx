"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { Chess, type Move, type PieceSymbol, type Square } from "chess.js";
import { Chessboard } from "react-chessboard";
import clsx from "clsx";
import { useChessEngine } from "@/lib/use-chess-engine";

type Mode = "pass-and-play" | "vs-computer";
type Side = "white" | "black";
type PromoPiece = "q" | "r" | "b" | "n";

type Status =
  | { tone: "normal" | "check"; text: string }
  | { tone: "win" | "draw"; text: string };

const LIGHT_SQ = "#f2e8d5";
const DARK_SQ = "#b58863";
const STORAGE_KEY = "chesspar:game-v1";
const ENGINE_MOVETIME_MS = 800;
const PIECE_VALUE: Record<PieceSymbol, number> = {
  p: 1,
  n: 3,
  b: 3,
  r: 5,
  q: 9,
  k: 0,
};

type SavedState = {
  v: 1;
  pgn: string;
  mode: Mode;
  playerSide: Side;
  orientation: Side;
  skill: number;
};

export function ChessGame() {
  const gameRef = useRef<Chess>(new Chess());
  const game = gameRef.current;

  const [fen, setFen] = useState<string>(() => game.fen());
  const [history, setHistory] = useState<Move[]>([]);
  const [mode, setMode] = useState<Mode>("pass-and-play");
  const [playerSide, setPlayerSide] = useState<Side>("white");
  const [orientation, setOrientation] = useState<Side>("white");
  const [selected, setSelected] = useState<Square | null>(null);
  const [skill, setSkill] = useState<number>(8);
  const [pendingPromo, setPendingPromo] = useState<{
    from: Square;
    to: Square;
  } | null>(null);
  const [resultDismissed, setResultDismissed] = useState<boolean>(false);
  const [hydrated, setHydrated] = useState<boolean>(false);

  const engine = useChessEngine(mode === "vs-computer");

  const sync = useCallback(() => {
    setFen(game.fen());
    setHistory(game.history({ verbose: true }) as Move[]);
  }, [game]);

  useEffect(() => {
    if (typeof window === "undefined") {
      setHydrated(true);
      return;
    }
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const data = JSON.parse(raw) as SavedState;
        if (data?.v === 1) {
          if (data.pgn) {
            try {
              game.loadPgn(data.pgn);
            } catch {
              game.reset();
            }
          }
          if (data.mode === "pass-and-play" || data.mode === "vs-computer")
            setMode(data.mode);
          if (data.playerSide === "white" || data.playerSide === "black")
            setPlayerSide(data.playerSide);
          if (data.orientation === "white" || data.orientation === "black")
            setOrientation(data.orientation);
          if (typeof data.skill === "number") setSkill(data.skill);
          sync();
        }
      }
    } catch {
      /* ignore */
    }
    setHydrated(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!hydrated || typeof window === "undefined") return;
    try {
      const data: SavedState = {
        v: 1,
        pgn: game.pgn(),
        mode,
        playerSide,
        orientation,
        skill,
      };
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch {
      /* ignore quota */
    }
  }, [fen, mode, playerSide, orientation, skill, hydrated, game]);

  useEffect(() => {
    if (game.isGameOver()) setResultDismissed(false);
  }, [fen, game]);

  const isHumanTurn = useCallback(() => {
    if (mode === "pass-and-play") return true;
    const turn: Side = game.turn() === "w" ? "white" : "black";
    return turn === playerSide;
  }, [game, mode, playerSide]);

  const tryMove = useCallback(
    (from: Square, to: Square, promotion: PromoPiece = "q"): Move | null => {
      try {
        const move = game.move({ from, to, promotion });
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

  const isPromotionMove = useCallback(
    (from: Square, to: Square): boolean => {
      const piece = game.get(from);
      if (!piece || piece.type !== "p") return false;
      const targetRank = to[1];
      if (piece.color === "w" && targetRank !== "8") return false;
      if (piece.color === "b" && targetRank !== "1") return false;
      const legal = (
        game.moves({ square: from, verbose: true }) as Move[]
      ).some((m) => m.to === to);
      return legal;
    },
    [game]
  );

  const engineRef = useRef(engine);
  engineRef.current = engine;

  useEffect(() => {
    if (mode !== "vs-computer") return;
    if (game.isGameOver()) return;
    if (isHumanTurn()) return;
    if (engine.status !== "ready") return;

    let cancelled = false;
    (async () => {
      const move = await engineRef.current.findBestMove(game.fen(), {
        skill,
        movetimeMs: ENGINE_MOVETIME_MS,
      });
      if (cancelled || !move) return;
      try {
        game.move({
          from: move.from,
          to: move.to,
          promotion: move.promotion ?? "q",
        });
        sync();
      } catch {
        /* engine returned a now-invalid move; ignore */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [fen, mode, engine.status, isHumanTurn, skill, game, sync]);

  function newGame(opts: { mode?: Mode; side?: Side } = {}) {
    game.reset();
    if (opts.mode) setMode(opts.mode);
    if (opts.side) {
      setPlayerSide(opts.side);
      setOrientation(opts.side);
    }
    setSelected(null);
    setPendingPromo(null);
    setResultDismissed(false);
    sync();
  }

  function flipBoard() {
    setOrientation((o) => (o === "white" ? "black" : "white"));
  }

  function undoLastMove() {
    if (history.length === 0) return;
    const undoCount =
      mode === "vs-computer" && history.length >= 2 && !isHumanTurn()
        ? 1
        : mode === "vs-computer" && history.length >= 2
          ? 2
          : 1;
    for (let i = 0; i < undoCount; i++) game.undo();
    setSelected(null);
    setPendingPromo(null);
    setResultDismissed(false);
    sync();
  }

  const legalTargets = useMemo<Square[]>(() => {
    if (!selected) return [];
    return (game.moves({ square: selected, verbose: true }) as Move[]).map(
      (m) => m.to
    );
  }, [selected, game]);

  const onPieceDrop = ({
    sourceSquare,
    targetSquare,
  }: {
    sourceSquare: string;
    targetSquare: string | null;
  }): boolean => {
    if (!targetSquare || !isHumanTurn()) return false;
    const from = sourceSquare as Square;
    const to = targetSquare as Square;
    if (isPromotionMove(from, to)) {
      setPendingPromo({ from, to });
      setSelected(null);
      return false;
    }
    const moved = tryMove(from, to);
    if (moved) setSelected(null);
    return Boolean(moved);
  };

  const onSquareClick = ({ square }: { square: string }) => {
    if (!isHumanTurn() || pendingPromo) return;
    const sq = square as Square;
    if (selected && selected !== sq) {
      if (isPromotionMove(selected, sq)) {
        setPendingPromo({ from: selected, to: sq });
        setSelected(null);
        return;
      }
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
  }, [selected, legalTargets, history, game]);

  const captured = useMemo(() => {
    const byWhite: PieceSymbol[] = [];
    const byBlack: PieceSymbol[] = [];
    for (const m of history) {
      if (m.captured) {
        if (m.color === "w") byWhite.push(m.captured);
        else byBlack.push(m.captured);
      }
    }
    const sortFn = (a: PieceSymbol, b: PieceSymbol) =>
      PIECE_VALUE[b] - PIECE_VALUE[a];
    byWhite.sort(sortFn);
    byBlack.sort(sortFn);
    const sumW = byWhite.reduce((s, p) => s + PIECE_VALUE[p], 0);
    const sumB = byBlack.reduce((s, p) => s + PIECE_VALUE[p], 0);
    return { byWhite, byBlack, advantage: sumW - sumB };
  }, [history]);

  const status: Status = useMemo(() => {
    if (game.isCheckmate()) {
      const winner = game.turn() === "w" ? "Black" : "White";
      return { tone: "win", text: `Checkmate — ${winner} wins` };
    }
    if (game.isStalemate())
      return { tone: "draw", text: "Stalemate — draw" };
    if (game.isInsufficientMaterial())
      return { tone: "draw", text: "Draw — insufficient material" };
    if (game.isThreefoldRepetition())
      return { tone: "draw", text: "Draw — threefold repetition" };
    if (game.isDraw()) return { tone: "draw", text: "Draw" };
    const turn = game.turn() === "w" ? "White" : "Black";
    if (game.inCheck())
      return { tone: "check", text: `${turn} to move — check!` };
    return { tone: "normal", text: `${turn} to move` };
  }, [game]);

  const topSide: Side = orientation === "white" ? "black" : "white";
  const topCaps = topSide === "white" ? captured.byWhite : captured.byBlack;
  const bottomCaps = topSide === "white" ? captured.byBlack : captured.byWhite;
  const topCapsAreOf: "w" | "b" = topSide === "white" ? "b" : "w";
  const bottomCapsAreOf: "w" | "b" = topCapsAreOf === "w" ? "b" : "w";
  const topAdv = topSide === "white" ? captured.advantage : -captured.advantage;
  const bottomAdv = -topAdv;

  const showResult = game.isGameOver() && !resultDismissed && hydrated;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="mx-auto flex w-full max-w-[640px] flex-col gap-2">
        <CapturedStrip
          pieces={topCaps}
          piecesColor={topCapsAreOf}
          advantage={topAdv}
          label={topSide === playerSide ? "You" : "Opponent"}
          showLabel={mode === "vs-computer"}
        />
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
        <CapturedStrip
          pieces={bottomCaps}
          piecesColor={bottomCapsAreOf}
          advantage={bottomAdv}
          label={
            (orientation === "white" ? "white" : "black") === playerSide
              ? "You"
              : "Opponent"
          }
          showLabel={mode === "vs-computer"}
        />
      </div>

      <aside className="flex flex-col gap-4">
        <QuickTips mode={mode} />
        <StatusBanner status={status} />
        {mode === "vs-computer" && <EnginePill engine={engine} />}
        <Controls
          mode={mode}
          playerSide={playerSide}
          skill={skill}
          canUndo={history.length > 0}
          onNewGame={newGame}
          onFlip={flipBoard}
          onUndo={undoLastMove}
          onChangeMode={(m) => newGame({ mode: m, side: playerSide })}
          onChangeSide={(s) => newGame({ mode, side: s })}
          onChangeSkill={setSkill}
        />
        <SharePanel game={game} />
        <MoveHistory moves={history} />
      </aside>

      {pendingPromo && (
        <PromotionPicker
          color={game.turn() === "w" ? "w" : "b"}
          onPick={(piece) => {
            const { from, to } = pendingPromo;
            tryMove(from, to, piece);
            setPendingPromo(null);
          }}
          onCancel={() => setPendingPromo(null)}
        />
      )}

      {showResult && (
        <ResultModal
          status={status}
          onNewGame={() => newGame()}
          onClose={() => setResultDismissed(true)}
        />
      )}
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
    <div
      className={clsx(
        "rounded-lg border px-4 py-3 text-sm font-medium",
        tone
      )}
    >
      {status.text}
    </div>
  );
}

function EnginePill({
  engine,
}: {
  engine: ReturnType<typeof useChessEngine>;
}) {
  const { status, thinking } = engine;
  let text: string;
  let tone: string;
  if (status === "loading") {
    text = "Loading engine…";
    tone = "border-blue-500/40 bg-blue-500/10 text-blue-200";
  } else if (status === "error") {
    text = "Engine failed to load";
    tone = "border-rose-500/40 bg-rose-500/10 text-rose-200";
  } else if (thinking) {
    text = "Computer is thinking…";
    tone = "border-violet-500/40 bg-violet-500/10 text-violet-200";
  } else if (status === "ready") {
    text = "Engine ready";
    tone = "border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
  } else {
    text = "Engine idle";
    tone = "border-zinc-700 bg-zinc-900/60 text-zinc-300";
  }
  return (
    <div className={clsx("rounded-md border px-3 py-2 text-xs", tone)}>
      <span className="inline-flex items-center gap-2">
        <span
          className={clsx(
            "inline-block h-1.5 w-1.5 rounded-full",
            thinking
              ? "animate-pulse bg-violet-300"
              : status === "ready"
                ? "bg-emerald-300"
                : status === "error"
                  ? "bg-rose-300"
                  : "bg-zinc-400"
          )}
        />
        {text}
      </span>
    </div>
  );
}

function Controls({
  mode,
  playerSide,
  skill,
  canUndo,
  onNewGame,
  onFlip,
  onUndo,
  onChangeMode,
  onChangeSide,
  onChangeSkill,
}: {
  mode: Mode;
  playerSide: Side;
  skill: number;
  canUndo: boolean;
  onNewGame: (opts?: { mode?: Mode; side?: Side }) => void;
  onFlip: () => void;
  onUndo: () => void;
  onChangeMode: (m: Mode) => void;
  onChangeSide: (s: Side) => void;
  onChangeSkill: (n: number) => void;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-3">
        <div className="mb-1 text-xs uppercase tracking-wider text-zinc-400">
          Mode
        </div>
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
        <>
          <div className="mb-3">
            <div className="mb-1 text-xs uppercase tracking-wider text-zinc-400">
              Play as
            </div>
            <Segmented
              value={playerSide}
              options={[
                { value: "white", label: "White" },
                { value: "black", label: "Black" },
              ]}
              onChange={(v) => onChangeSide(v as Side)}
            />
          </div>
          <SkillSlider skill={skill} onChange={onChangeSkill} />
        </>
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

function SkillSlider({
  skill,
  onChange,
}: {
  skill: number;
  onChange: (n: number) => void;
}) {
  const label =
    skill <= 3
      ? "Beginner"
      : skill <= 7
        ? "Casual"
        : skill <= 12
          ? "Club"
          : skill <= 16
            ? "Strong"
            : "Master";
  return (
    <div className="mb-3">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wider text-zinc-400">
          Engine strength
        </span>
        <span className="text-xs text-zinc-300">
          {label} · level {skill}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={20}
        step={1}
        value={skill}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1 w-full cursor-pointer appearance-none rounded-full bg-zinc-700 accent-emerald-400"
      />
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
    <details className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4" open>
      <summary className="mb-2 cursor-pointer list-none text-xs uppercase tracking-wider text-zinc-400">
        Moves
      </summary>
      {pairs.length === 0 ? (
        <div className="text-sm text-zinc-500">No moves yet.</div>
      ) : (
        <ol className="max-h-72 overflow-y-auto pr-1 text-sm tabular-nums">
          {pairs.map((p) => (
            <li
              key={p.num}
              className="grid grid-cols-[2.5rem_1fr_1fr] gap-2 py-0.5"
            >
              <span className="text-zinc-500">{p.num}.</span>
              <span className="text-zinc-100">{p.white?.san}</span>
              <span className="text-zinc-300">{p.black?.san ?? ""}</span>
            </li>
          ))}
        </ol>
      )}
    </details>
  );
}

function QuickTips({ mode }: { mode: Mode }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-4 py-3 text-xs text-zinc-300">
      <div className="mb-1 font-medium uppercase tracking-wider text-zinc-400">Quick tips</div>
      <ul className="list-disc space-y-1 pl-4">
        <li>Select a piece to see legal targets, then click destination or drag.</li>
        <li>Use Undo to review mistakes; in vs-computer it rewinds both moves.</li>
        {mode === "vs-computer" && <li>Adjust engine strength anytime to ramp difficulty.</li>}
      </ul>
    </div>
  );
}

const PIECE_GLYPH: Record<"w" | "b", Record<PieceSymbol, string>> = {
  w: { p: "♙", n: "♘", b: "♗", r: "♖", q: "♕", k: "♔" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
};

function CapturedStrip({
  pieces,
  piecesColor,
  advantage,
  label,
  showLabel,
}: {
  pieces: PieceSymbol[];
  piecesColor: "w" | "b";
  advantage: number;
  label: string;
  showLabel: boolean;
}) {
  return (
    <div className="flex h-6 items-center gap-2 px-1 text-zinc-300">
      {showLabel && (
        <span className="text-xs uppercase tracking-wider text-zinc-500">
          {label}
        </span>
      )}
      <span
        className={clsx(
          "select-none text-lg leading-none tracking-tight",
          piecesColor === "w" ? "text-zinc-100" : "text-zinc-900"
        )}
        style={{ textShadow: piecesColor === "b" ? "0 0 1px #f4f4f5" : undefined }}
      >
        {pieces.map((p, i) => (
          <span key={`${p}-${i}`} className="-mr-1.5">
            {PIECE_GLYPH[piecesColor][p]}
          </span>
        ))}
      </span>
      {advantage > 0 && (
        <span className="text-xs font-medium text-zinc-400">+{advantage}</span>
      )}
    </div>
  );
}

function SharePanel({ game }: { game: Chess }) {
  const [copied, setCopied] = useState<string | null>(null);

  async function copy(text: string, label: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
      setTimeout(() => setCopied((c) => (c === label ? null : c)), 1400);
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-2 text-xs uppercase tracking-wider text-zinc-400">
        Share
      </div>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => copy(game.fen(), "FEN")}
          className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700"
        >
          {copied === "FEN" ? "FEN copied" : "Copy FEN"}
        </button>
        <button
          onClick={() => copy(game.pgn(), "PGN")}
          className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700"
        >
          {copied === "PGN" ? "PGN copied" : "Copy PGN"}
        </button>
      </div>
    </div>
  );
}

function PromotionPicker({
  color,
  onPick,
  onCancel,
}: {
  color: "w" | "b";
  onPick: (piece: PromoPiece) => void;
  onCancel: () => void;
}) {
  const choices: PromoPiece[] = ["q", "r", "b", "n"];
  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 px-4"
      onClick={onCancel}
    >
      <div
        className="rounded-xl border border-zinc-700 bg-zinc-900 p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 text-sm font-medium text-zinc-200">
          Promote pawn to:
        </div>
        <div className="flex gap-2">
          {choices.map((p) => (
            <button
              key={p}
              onClick={() => onPick(p)}
              className="flex h-16 w-16 items-center justify-center rounded-md border border-zinc-700 bg-zinc-800 text-4xl leading-none hover:bg-zinc-700"
              style={{
                color: color === "w" ? "#f4f4f5" : "#0a0a0a",
                textShadow: color === "b" ? "0 0 1px #f4f4f5" : "0 0 1px #0a0a0a",
              }}
              aria-label={`Promote to ${p}`}
            >
              {PIECE_GLYPH[color][p]}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function ResultModal({
  status,
  onNewGame,
  onClose,
}: {
  status: Status;
  onNewGame: () => void;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-30 flex items-center justify-center bg-black/60 px-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-sm rounded-xl border border-zinc-700 bg-zinc-900 p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-1 text-xs uppercase tracking-wider text-zinc-400">
          Game over
        </div>
        <div className="mb-5 text-2xl font-semibold text-zinc-100">
          {status.text}
        </div>
        <div className="flex gap-2">
          <button
            onClick={onNewGame}
            className="flex-1 rounded-md border border-emerald-500/40 bg-emerald-500/15 px-4 py-2 text-sm text-emerald-200 hover:bg-emerald-500/25"
          >
            New game
          </button>
          <button
            onClick={onClose}
            className="rounded-md border border-zinc-700 bg-zinc-800 px-4 py-2 text-sm text-zinc-200 hover:bg-zinc-700"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
