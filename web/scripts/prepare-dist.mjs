import { copyFile, mkdir, rm } from "node:fs/promises";

const webRoot = new URL("../", import.meta.url);
const output = new URL("dist/", webRoot);

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await Promise.all(
  ["index.html", "styles.css", "favicon.svg"].map((filename) =>
    copyFile(new URL(filename, webRoot), new URL(filename, output)),
  ),
);
