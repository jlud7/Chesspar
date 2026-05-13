import { CaptureGame } from "@/components/capture-game";

export const metadata = {
  title: "Chesspar — Capture",
  description:
    "Play a timed game. Each tap of your clock captures the board for move inference.",
};

export default function CapturePage() {
  return <CaptureGame />;
}
