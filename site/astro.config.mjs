// @ts-check
import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import tailwindcss from "@tailwindcss/vite";

// SITE_URL / BASE_PATH let the same build target GitHub Pages (project path) or a custom domain.
const site = process.env.SITE_URL ?? "https://alhaidari-catalog.example.org";
const base = process.env.BASE_PATH ?? "/";

export default defineConfig({
  site,
  base,
  trailingSlash: "always",
  build: { format: "directory" },
  integrations: [sitemap()],
  vite: { plugins: [tailwindcss()] },
});
