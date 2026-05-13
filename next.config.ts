import type { NextConfig } from "next";

const isGithubActions = process.env.GITHUB_ACTIONS === "true";
const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1] ?? "";
const basePath = isGithubActions && repoName ? `/${repoName}` : "";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  basePath,
  assetPrefix: basePath ? `${basePath}/` : undefined,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
    NEXT_PUBLIC_VLM_PROXY_URL: process.env.NEXT_PUBLIC_VLM_PROXY_URL ?? "",
    NEXT_PUBLIC_VLM_PROXY_PROVIDER:
      process.env.NEXT_PUBLIC_VLM_PROXY_PROVIDER ?? "anthropic"
  }
};

export default nextConfig;
