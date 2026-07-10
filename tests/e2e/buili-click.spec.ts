import { expect, test } from "@playwright/test";

const APP_URL = process.env.BUILI_E2E_URL ?? "http://127.0.0.1:3000";

async function openAuthenticated(page: import("@playwright/test").Page) {
  await page.goto(APP_URL, { waitUntil: "networkidle" });

  const signInForm = page.getByRole("form", { name: "Sign in to Buili" });
  if (!(await signInForm.isVisible().catch(() => false))) return;

  const email = process.env.BUILI_E2E_EMAIL;
  const password = process.env.BUILI_E2E_PASSWORD;
  expect(email, "BUILI_E2E_EMAIL is required when the deployment has login enabled").toBeTruthy();
  expect(password, "BUILI_E2E_PASSWORD is required when the deployment has login enabled").toBeTruthy();

  await signInForm.getByRole("textbox", { name: "Work email" }).fill(email!);
  await signInForm.getByRole("textbox", { name: "Password" }).fill(password!);
  await signInForm.getByRole("button", { name: "Sign in securely" }).click();
  await expect(page.getByRole("heading", { name: /Good morning,/i })).toBeVisible();
}

test("pilot login is cookie-bound, tenant-scoped, and revocable", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(APP_URL, { waitUntil: "networkidle" });
  const signInForm = page.getByRole("form", { name: "Sign in to Buili" });
  test.skip(!(await signInForm.isVisible().catch(() => false)), "Run with BUILI_AUTH_REQUIRED=true");

  const email = process.env.BUILI_E2E_EMAIL;
  const password = process.env.BUILI_E2E_PASSWORD;
  expect(email, "BUILI_E2E_EMAIL is required for authenticated deployment QA").toBeTruthy();
  expect(password, "BUILI_E2E_PASSWORD is required for authenticated deployment QA").toBeTruthy();

  const forged = await page.context().request.get(`${APP_URL}/api/v1/projects`, {
    headers: { "X-Buili-Actor": "forged-user", "X-Buili-Role": "admin" }
  });
  expect(forged.status(), "browser-supplied identity headers must not bypass login").toBe(401);

  await signInForm.getByRole("textbox", { name: "Work email" }).fill(email!);
  await signInForm.getByRole("textbox", { name: "Password" }).fill(password!);
  await signInForm.getByRole("button", { name: "Sign in securely" }).click();
  await expect(page.getByRole("heading", { name: /Good morning, Jordan/i })).toBeVisible();

  const sessionCookie = (await page.context().cookies()).find(cookie => cookie.name === "buili_session");
  expect(sessionCookie?.httpOnly, "session credential must not be readable by browser scripts").toBe(true);
  expect(sessionCookie?.sameSite).toBe("Lax");

  const projectsResponse = await page.context().request.get(`${APP_URL}/api/v1/projects`);
  expect(projectsResponse.status()).toBe(200);
  const projects = await projectsResponse.json() as Array<{ project_id: string }>;
  expect(projects, "the pilot identity is intentionally scoped to one project").toHaveLength(1);

  const documentsResponse = await page.context().request.get(
    `${APP_URL}/api/v1/projects/${projects[0].project_id}/documents`
  );
  expect(documentsResponse.status()).toBe(200);
  const documents = await documentsResponse.json() as Array<{
    type?: string;
    is_current?: boolean;
  }>;
  expect(
    documents.filter(document => document.type === "plan" && document.is_current),
    "the pilot workspace has one authoritative current drawing"
  ).toHaveLength(1);

  await page.getByRole("button", { name: "Open account menu" }).click();
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByRole("form", { name: "Sign in to Buili" })).toBeVisible();
  const afterLogout = await page.context().request.get(`${APP_URL}/api/v1/projects`);
  expect(afterLogout.status()).toBe(401);
});

test("specification workspaces are navigable and responsive", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openAuthenticated(page);

  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(page.getByRole("heading", { name: /Good morning, Jordan/i })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Mobile" })).toBeVisible();
  const mobileNav = page.getByRole("navigation", { name: "Mobile" });
  for (const [name, accessibleName] of [["Home", "Home"], ["Capture", "Capture"], ["Issues", "Issues"], ["More", "Open workspace menu"]]) {
    const button = mobileNav.getByRole("button", { name: accessibleName, exact: true });
    await expect(button).toBeVisible();
    const box = await button.boundingBox();
    expect(box?.height ?? 0, `${name} must meet the 44px mobile tap target`).toBeGreaterThanOrEqual(44);
    expect(box?.width ?? 0, `${name} must meet the 44px mobile tap target`).toBeGreaterThanOrEqual(44);
  }
  await page.getByRole("button", { name: "Issues", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Issues", exact: true })).toBeVisible();
  await mobileNav.getByRole("button", { name: "Open workspace menu", exact: true }).click();
  const workspaceMenu = page.getByRole("dialog", { name: "Workspace menu" });
  await expect(workspaceMenu).toBeVisible();
  await workspaceMenu.getByRole("button", { name: "Project settings", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Project settings", exact: true })).toBeVisible();

  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole("button", { name: "Files & revisions", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Files & revisions" })).toBeVisible();
  await expect(page.getByText(/Current-set integrity is enforced/i)).toBeVisible();
  await expect(page.getByText(/current/i).first()).toBeVisible();
  await expect(page.getByText(/superseded/i).first()).toBeVisible();
});

test("installed app shell remains usable when the network drops", async ({ context, page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openAuthenticated(page);
  await page.evaluate(async () => {
    if (!("serviceWorker" in navigator)) throw new Error("service workers are unavailable");
    await navigator.serviceWorker.ready;
  });

  // Reload once while online so the active worker controls this tab and its Next.js assets.
  await page.reload({ waitUntil: "networkidle" });
  const sensitiveCacheEntries = await page.evaluate(async () => {
    const keys = await caches.keys();
    const urls = (
      await Promise.all(
        keys.map(async (key) => (await caches.open(key)).keys().then((items) => items.map((item) => item.url)))
      )
    ).flat();
    return urls.filter((entry) => /\/api\/|\/v1\/reports\/|\/v1\/projects\//.test(entry));
  });
  expect(sensitiveCacheEntries, "service worker must not cache tenant API or report responses").toEqual([]);
  await context.setOffline(true);
  await page.reload({ waitUntil: "domcontentloaded" });

  await expect(page.getByRole("navigation", { name: "Mobile" })).toBeVisible();
  await expect(page.getByRole("heading", { name: /Good morning, Jordan/i })).toBeVisible();
  await page.getByRole("button", { name: "Capture", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Field capture" })).toBeVisible();
});

test("mobile capture is locally durable and sync is explicit", async ({ context, page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openAuthenticated(page);
  await page.getByRole("button", { name: "Capture", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Field capture" })).toBeVisible();

  await context.setOffline(true);
  await page.getByRole("textbox", { name: "Room / zone" }).fill("Room 204");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Measure" }).click();
  await page.getByRole("textbox", { name: "Measurement" }).fill("31 7/8 in clear opening");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Save observation" }).click();
  await expect(page.getByRole("region", { name: "Offline queue" })).toContainText(/saved|queued/i);

  const queuedBeforeReload = await page.evaluate(async () => {
    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open("buili-field-capture", 1);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return await new Promise<Array<{ id: string; size: number }>>((resolve, reject) => {
      const request = database.transaction("captures", "readonly").objectStore("captures").getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  });
  expect(queuedBeforeReload.length).toBeGreaterThan(0);
  expect(queuedBeforeReload[0].id).toMatch(/^capture_/);
  expect(queuedBeforeReload[0].size).toBeGreaterThan(0);

  await page.reload({ waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Capture", exact: true }).click();
  await expect(page.getByRole("region", { name: "Offline queue" })).toContainText(/saved|queued|pending/i);

  await context.setOffline(false);
  await page.getByRole("button", { name: "Close Field capture" }).click();
  await page.getByRole("button", { name: "Sync now" }).click();
  await expect(page.getByRole("region", { name: "Offline queue" })).toContainText(/synced|complete|0 pending/i);
});

test("review decisions require a reason and remain visible in history", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openAuthenticated(page);
  await page.getByRole("button", { name: "Review queue", exact: true }).click();
  await page.getByRole("button", { name: "Request evidence", exact: true }).first().click();

  const decisionDialog = page.getByRole("dialog", { name: "Request evidence" });
  await expect(decisionDialog).toBeVisible();
  const confirm = decisionDialog.getByRole("button", { name: "Send evidence request" });
  await confirm.click();
  await expect(decisionDialog.getByRole("alert")).toContainText(/reason/i);
  await decisionDialog.getByRole("combobox", { name: "Reason" }).selectOption("context_photo");
  await decisionDialog
    .getByRole("textbox", { name: "Instructions to field team" })
    .fill("Add a wide-angle context photo.");
  await confirm.click();

  await expect(page.getByRole("region", { name: "Review history" })).toContainText(
    "Add a wide-angle context photo."
  );
});

test("2D source and generated 3D stay synchronized through issue verification", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openAuthenticated(page);
  await page.getByRole("button", { name: "Drawings & 3D", exact: true }).click();

  await expect(page.getByRole("heading", { name: "Drawings & 3D" })).toBeVisible();
  await expect(page.getByText(/Current drawing set/i)).toBeVisible();
  await expect(page.getByRole("img", { name: "Current drawing source" })).toBeVisible();

  await page.getByRole("button", { name: "Split View", exact: true }).click();
  const model = page.getByRole("region", { name: "Generated 3D model viewer" });
  await expect(model).toBeVisible();
  await expect(model.locator("canvas")).toBeVisible();
  await expect(model).toContainText(/Generated from current 2D source/i);
  await expect(page.getByText(/Source coordinates preserved|Generated 3D context/i).first()).toBeVisible();

  await page.getByRole("button", { name: "Verify issue", exact: true }).click();
  const steps = page.getByRole("navigation", { name: "Issue verification steps" });
  await expect(steps).toBeVisible();
  for (const step of ["Locate", "Compare", "Decide"]) {
    await expect(steps.getByRole("button", { name: new RegExp(step, "i") })).toBeVisible();
  }
  await steps.getByRole("button", { name: /Locate/i }).click();
  await expect(page.getByRole("img", { name: /Current 2D drawing with synchronized issue pins/i })).toBeVisible();
  await steps.getByRole("button", { name: /Compare/i }).click();
  await expect(page.getByText(/Selection and source coordinates stay synchronized/i)).toBeVisible();
  await expect(page.getByLabel("Issue readiness")).toContainText(/Current source/i);
  await expect(page.getByLabel("Issue readiness")).toContainText(/Field observation/i);
});

test("RFI and Punch builders generate selected source-backed draft versions", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openAuthenticated(page);
  await page.getByRole("button", { name: "Reports", exact: true }).click();

  await expect(page.getByRole("heading", { name: "Reports" })).toBeVisible();
  const output = page.getByLabel("Output");
  const generate = page.getByRole("button", { name: "Generate report" });

  await output.selectOption("rfi");
  await expect(page.getByRole("heading", { name: "RFI preview" })).toBeVisible();
  await expect(page.getByText(/current source/i).last()).toBeVisible();
  await expect(generate).toBeEnabled();
  const rfiResponsePromise = page.waitForResponse(response =>
    response.request().method() === "POST" && /\/v1\/projects\/[^/]+\/reports$/.test(new URL(response.url()).pathname)
  );
  await generate.click();
  const rfiResponse = await rfiResponsePromise;
  expect(rfiResponse.status()).toBe(200);
  const rfi = await rfiResponse.json() as { report_id: string; download_url: string };
  expect(rfi.report_id.length).toBeGreaterThan(8);
  expect(rfi.download_url).toContain("/v1/reports/");
  await expect(page.getByRole("link", { name: /Download latest generated report/i })).toBeVisible();

  await output.selectOption("punch");
  await expect(page.getByRole("heading", { name: "Punch list preview" })).toBeVisible();
  await expect(page.getByText(/responsible party/i)).toBeVisible();
  await expect(page.getByText(/due date/i)).toBeVisible();
  await expect(generate).toBeEnabled();
  const punchResponsePromise = page.waitForResponse(response =>
    response.request().method() === "POST" && /\/v1\/projects\/[^/]+\/reports$/.test(new URL(response.url()).pathname)
  );
  await generate.click();
  const punchResponse = await punchResponsePromise;
  expect(punchResponse.status()).toBe(200);
  const punch = await punchResponse.json() as { report_id: string };
  expect(punch.report_id.length).toBeGreaterThan(8);
  expect(punch.report_id).not.toBe(rfi.report_id);
  await expect(page.getByLabel("Report history")).toContainText(/rfi/i);
  await expect(page.getByLabel("Report history")).toContainText(/punch/i);
});
