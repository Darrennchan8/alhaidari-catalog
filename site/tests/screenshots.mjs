// Ad-hoc visual check: node tests/screenshots.mjs [outdir] (needs `npx astro preview` running).
import { chromium } from "@playwright/test";
const out = process.argv[2] ?? "screenshots";
const pages = process.argv.slice(3).length ? process.argv.slice(3) : ["/", "/videos/", "/series/", "/topics/", "/about/"];
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined, args: ["--no-sandbox"] });
for (const [name, viewport] of [["desktop", { width: 1366, height: 900 }], ["mobile", { width: 390, height: 844 }]]) {
  for (const theme of ["light", "dark"]) {
    const ctx = await browser.newContext({ viewport, colorScheme: theme });
    const page = await ctx.newPage();
    for (const p of pages) {
      await page.goto(`http://localhost:4321${p}`, { waitUntil: "load", timeout: 20000 });
      const file = `${out}/${name}-${theme}${p.replace(/[^\w]+/g, "_")}.png`;
      await page.screenshot({ path: file, fullPage: false });
      console.log(file);
    }
    await ctx.close();
  }
}
await browser.close();
