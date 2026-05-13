import Link from "next/link";
import { BoardRectifier } from "@/components/board-rectifier";

export const metadata = {
  title: "Chesspar — Detect",
  description:
    "Rectify a chessboard photo into a clean 8×8 grid by tapping its four corners.",
};

export default function DetectPage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-6 py-10">
      <header className="mb-8">
        <Link
          href="/"
          className="text-xs uppercase tracking-wider text-zinc-400 hover:text-zinc-200"
        >
          ← Back to home
        </Link>
        <h1 className="mt-2 text-3xl font-bold tracking-tight md:text-4xl">
          Detect
        </h1>
        <p className="mt-1 max-w-2xl text-sm text-zinc-400">
          Upload a board photo, tap the four named corners, and we rectify it
          into a clean 8×8 grid for downstream move detection.
        </p>
      </header>

      <BoardRectifier />
    </main>
  );
}
