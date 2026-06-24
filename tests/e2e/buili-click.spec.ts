import { expect, test } from "@playwright/test";
import { writeFileSync } from "node:fs";

test("primary Buili buttons provide visible results", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("http://143.248.47.23:3000", { waitUntil: "networkidle" });

  await page.getByRole("button", { name: /Run review/i }).click();
  await expect(page.getByText(/Review run queued|Review complete/i)).toBeVisible();

  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await page.getByTitle("Search").click();
  await expect(page.getByText(/Found \d+ citation/i)).toBeVisible();

  await page.getByRole("button", { name: "Reports", exact: true }).click();
  await page.getByRole("button", { name: /Punch PDF/i }).click();
  await expect(page.getByText(/PDF report generated/i)).toBeVisible();

  await page.getByRole("button", { name: /Install/i }).click();
  await expect(
    page.getByText(/browser address bar|Buili install started|Install was dismissed/i),
  ).toBeVisible();

  writeFileSync("/tmp/buili_upload_check.txt", "E-101\nInstall two duplex outlets on north wall.\n");
  await page.setInputFiles('input[type="file"]', "/tmp/buili_upload_check.txt");
  await expect(page.getByText(/uploaded\. Run review/i)).toBeVisible();
});
