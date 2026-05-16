import { Capture } from "@/components/v2/Capture";

export const metadata = {
  title: "Chesspar — Capture",
  description:
    "Aim your phone at the board. Tap to capture each move. Chesspar writes the PGN.",
};

export default function CapturePage() {
  return <Capture />;
}
