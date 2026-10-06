import { expect, test } from "@playwright/test";

test("home page shows stats, series and search", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("catalogued in English");
  await expect(page.getByRole("searchbox")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Lecture series" })).toBeVisible();
});

test("video browser filters and paginates", async ({ page }) => {
  await page.goto("/videos/");
  const count = page.locator("#count");
  await expect(page.locator("#results article").first()).toBeVisible();
  const total = Number((await count.textContent())!.replace(/,/g, ""));
  expect(total).toBeGreaterThan(100);
  await page.locator("#f-len").selectOption("l");
  await expect.poll(async () => Number((await count.textContent())!.replace(/,/g, ""))).toBeLessThan(total);
  await expect(page).toHaveURL(/len=l/);
});

test("series page lists episodes in order", async ({ page }) => {
  await page.goto("/series/");
  await page.locator("main a[href*='/series/']").first().click();
  await expect(page.locator("main ol > li").first()).toBeVisible();
});

test("video page renders player, Arabic title and transcript or placeholder", async ({ page }) => {
  await page.goto("/videos/?transcript=1");
  const first = page.locator("#results article a").first();
  const hasTranscribed = (await first.count()) > 0;
  if (!hasTranscribed) await page.goto("/videos/");
  await page.locator("#results article a").first().click();
  await expect(page.locator("#player-wrap")).toBeVisible();
  await expect(page.locator("main [lang=ar]").first()).toBeVisible();
  if (hasTranscribed) {
    await expect(page.locator(".seg").first()).toBeVisible();
    await page.locator("#show-terms").check();
    await expect(page.locator("#segments")).toHaveAttribute("data-terms", "on");
  } else {
    await expect(page.getByText("Transcript in progress")).toBeVisible();
  }
});

test("search finds results", async ({ page }) => {
  await page.goto("/search/?q=Imam");
  await expect(page.locator(".pagefind-ui__result").first()).toBeVisible({ timeout: 15_000 });
});

test("no horizontal overflow on key pages", async ({ page }) => {
  for (const path of ["/", "/videos/", "/series/", "/topics/", "/about/"]) {
    await page.goto(path);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow, path).toBeLessThanOrEqual(1);
  }
});
