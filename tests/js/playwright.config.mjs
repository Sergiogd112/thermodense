import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "figure_review_smoke.spec.mjs",
  reporter: "line",
  use: {
    headless: true,
    launchOptions: {
      executablePath: process.env.CHROMIUM_PATH ?? "/usr/bin/chromium",
    },
  },
});
