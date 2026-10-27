import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Browser calls /api/* on this origin; Next forwards to the API, so remote access needs only one port.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${process.env.LOCALLAB_API_INTERNAL_URL ?? "http://127.0.0.1:8321"}/api/:path*` }];
  },
  /* config options here */
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
