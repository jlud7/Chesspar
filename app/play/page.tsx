import Link from "next/link";
import { ChessGame } from "@/components/chess-game";

export const metadata = {
  title: "Chesspar — Play",
  description: "Play a chess game in the browser. Pass-and-play or vs computer."
};

export default function PlayPage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-6 py-10">
      <header className="mb-8 flex items-center justify-between">
        <div>
          <Link
            href="/"
            className="text-xs uppercase tracking-wider text-zinc-400 hover:text-zinc-200"
          >
            ← Back to home
          </Link>
          <h1 className="mt-2 text-3xl font-bold tracking-tight md:text-4xl">Play</h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-400">
            Drag a piece or tap a square. Legal moves are highlighted; the king flashes red on
            check.
          </p>
        </div>
      </header>

      <ChessGame />
    </main>
  );
}
