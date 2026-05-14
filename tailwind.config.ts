import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        board: {
          light: "#f2e8d5",
          dark: "#b58863"
        },
        canvas: "var(--cp-canvas)",
        ink: "var(--cp-ink)"
      },
      fontFamily: {
        sans: ["var(--font-ui)"],
        serif: ["var(--font-serif)"],
        mono: ["var(--font-mono)"]
      }
    }
  },
  plugins: []
};

export default config;
