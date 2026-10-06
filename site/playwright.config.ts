import { defineConfig, devices } from "@playwright/test";

// Runs against the production build (`npm run build` first). Set CHROMIUM_PATH to use a
// system Chromium (e.g. on Alpine/musl, where Playwright's bundled browsers don't run).
const executablePath = process.env.CHROMIUM_PATH || undefined;

export default defineConfig({
  testDir: "tests",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: "http://localhost:4321",
    launchOptions: { executablePath },
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], launchOptions: { executablePath } } },
    { name: "mobile", use: { ...devices["Pixel 7"], launchOptions: { executablePath } } },
  ],
  webServer: {
    command: "npx astro preview --port 4321",
    port: 4321,
    reuseExistingServer: !process.env.CI,
  },
});
