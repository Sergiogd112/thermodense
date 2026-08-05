import { expect, test } from "@playwright/test";

const baseURL = process.env.FIGURE_REVIEW_URL ?? "http://127.0.0.1:8124/";

test.beforeEach(async ({ page }) => {
  await page.goto(baseURL);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
});

test("desktop board persists movement and supports detail and comparison", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await expect(page.locator(".b-col")).toHaveCount(5);

  const firstCard = page.locator(".b-card").first();
  await firstCard.locator(".card-decision").selectOption("include");
  await expect(page.locator('[data-col="Include"] .b-card')).toHaveCount(1);
  await page.reload();
  await expect(page.locator('[data-col="Include"] .b-card')).toHaveCount(1);

  await page.locator(".b-card .row2 button").first().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByText("Preview PNG SHA-256 verified")).toBeVisible();
  await expect(page.getByText("Publication PDF SHA-256 verified")).toBeVisible();
  await page.getByRole("button", { name: "Close figure detail" }).click();

  const compare = page.locator('.b-card input[type="checkbox"]');
  await compare.nth(0).check();
  await compare.nth(1).check();
  await expect(page.locator(".compare-grid .col")).toHaveCount(2);
});

test("phone board uses one touch-sized column and unshrunk physical paper", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".b-mobile-hint")).toBeVisible();
  const columnWidth = await page.locator(".b-col").first().evaluate((element) => element.getBoundingClientRect().width);
  expect(columnWidth).toBeGreaterThan(360);
  const selectHeight = await page.locator(".card-decision").first().evaluate((element) => element.getBoundingClientRect().height);
  expect(selectHeight).toBeGreaterThanOrEqual(40);

  await page.locator(".b-card .row2 button").first().click();
  await page.locator(".view-select").selectOption("print");
  const paperWidth = await page.locator(".print-sheet").evaluate((element) => element.getBoundingClientRect().width);
  expect(paperWidth).toBeGreaterThan(790);
});
