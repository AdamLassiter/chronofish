import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const packageJson = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));

test("build output declares package version on the main page", async () => {
  const html = await readFile(path.join(root, "dist/index.html"), "utf8");
  const appVersion = await readFile(path.join(root, "dist/app-version.js"), "utf8");

  assert.match(html, new RegExp(`Web v${escapeRegExp(packageJson.version)}`));
  assert.equal(appVersion.trim(), `export const APP_VERSION = ${JSON.stringify(packageJson.version)};`);
});

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
