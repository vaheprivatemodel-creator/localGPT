import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  async rewrites() {
    return [
      // /api/stream is now handled by src/app/api/stream/route.ts (a real
      // Next.js route handler) so SSE bytes are flushed unbuffered. Do NOT
      // re-introduce a rewrite for this path or the dev server will buffer.
      {
        source: "/api/:path*",
        destination: "http://localhost:8002/:path*",
      },
    ];
  },
};

export default nextConfig;
