import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        board: {
          light: "#f2e8d5",
          dark: "#b58863"
        }
      }
    }
  },
  plugins: []
};

export default config;
