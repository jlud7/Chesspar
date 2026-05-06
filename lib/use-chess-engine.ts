"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type EngineStatus = "idle" | "loading" | "ready" | "error";

export type EngineMove = {
  from: string;
  to: string;
  promotion?: "q" | "r" | "b" | "n";
};

const STOCKFISH_JS = "/stockfish/stockfish-18-lite-single.js";

export function useChessEngine(active: boolean) {
  const workerRef = useRef<Worker | null>(null);
  const handlerRef = useRef<((line: string) => void) | null>(null);
  const [status, setStatus] = useState<EngineStatus>("idle");
  const [thinking, setThinking] = useState(false);

  useEffect(() => {
    if (!active || workerRef.current || typeof window === "undefined") return;

    const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    setStatus("loading");

    let worker: Worker;
    try {
      worker = new Worker(`${basePath}${STOCKFISH_JS}`);
    } catch {
      setStatus("error");
      return;
    }

    workerRef.current = worker;
    worker.onmessage = (e: MessageEvent) => {
      const line = typeof e.data === "string" ? e.data : "";
      handlerRef.current?.(line);
    };
    worker.onerror = () => setStatus("error");

    let uciOk = false;
    handlerRef.current = (line) => {
      if (line === "uciok") {
        uciOk = true;
        worker.postMessage("setoption name Use NNUE value true");
        worker.postMessage("isready");
      } else if (line === "readyok" && uciOk) {
        handlerRef.current = null;
        setStatus("ready");
      }
    };
    worker.postMessage("uci");
  }, [active]);

  useEffect(() => {
    return () => {
      const w = workerRef.current;
      if (w) {
        try {
          w.postMessage("quit");
        } catch {
          /* ignore */
        }
        try {
          w.terminate();
        } catch {
          /* ignore */
        }
        workerRef.current = null;
      }
    };
  }, []);

  const findBestMove = useCallback(
    (
      fen: string,
      opts: { skill: number; movetimeMs: number }
    ): Promise<EngineMove | null> => {
      return new Promise((resolve) => {
        const w = workerRef.current;
        if (!w) {
          resolve(null);
          return;
        }

        let resolved = false;
        const finalize = (move: EngineMove | null) => {
          if (resolved) return;
          resolved = true;
          clearTimeout(timeout);
          if (handlerRef.current === onLine) handlerRef.current = null;
          setThinking(false);
          resolve(move);
        };

        const onLine = (line: string) => {
          if (line.startsWith("bestmove ")) {
            const move = line.split(/\s+/)[1];
            if (!move || move === "(none)") {
              finalize(null);
              return;
            }
            finalize({
              from: move.slice(0, 2),
              to: move.slice(2, 4),
              promotion:
                move.length === 5 ? (move[4] as EngineMove["promotion"]) : undefined,
            });
          }
        };

        const timeout = setTimeout(
          () => {
            try {
              w.postMessage("stop");
            } catch {
              /* ignore */
            }
            finalize(null);
          },
          Math.max(opts.movetimeMs * 4, 30000)
        );

        setThinking(true);
        handlerRef.current = onLine;

        const skill = Math.max(0, Math.min(20, Math.round(opts.skill)));
        w.postMessage(`setoption name Skill Level value ${skill}`);
        w.postMessage(`position fen ${fen}`);
        w.postMessage(`go movetime ${Math.max(50, Math.round(opts.movetimeMs))}`);
      });
    },
    []
  );

  return { status, thinking, findBestMove };
}
