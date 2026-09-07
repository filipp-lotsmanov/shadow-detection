import { FlatCompat } from "@eslint/eslintrc";

// eslint-config-next 15.x still ships eslintrc-style configs, so it is bridged
// into flat config via FlatCompat. This replaces `next lint`, which is
// deprecated and removed in Next.js 16.
const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const config = [
  { ignores: [".next/**", "out/**", "node_modules/**", "next-env.d.ts"] },
  ...compat.extends("next/core-web-vitals"),
  {
    rules: {
      // The demo renders local sample thumbnails and object-URL previews of
      // user uploads. next/image adds no value for either (the loader is
      // disabled in next.config.js) and cannot take a blob: URL.
      "@next/next/no-img-element": "off",
    },
  },
];

export default config;
