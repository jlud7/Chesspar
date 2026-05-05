import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Chesspar",
  description: "Personalized chess training from your real games"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
