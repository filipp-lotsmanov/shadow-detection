/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Disable production builds optimizing images for this local-only demo.
  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;
