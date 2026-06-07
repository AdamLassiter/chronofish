import { mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";

const root = process.cwd();
const src = path.join(root, "src");
const dist = path.join(root, "dist");
const packageJson = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });
await copyTree(src, dist);
await run("npx", ["tsc"]);
await writeFile(
  path.join(dist, "app-version.js"),
  `export const APP_VERSION = ${JSON.stringify(packageJson.version)};\n`,
  "utf8"
);

async function copyTree(from, to) {
  await mkdir(to, { recursive: true });
  for (const entry of await readdir(from)) {
    const source = path.join(from, entry);
    const target = path.join(to, entry);
    const info = await stat(source);
    if (info.isDirectory()) {
      await copyTree(source, target);
      continue;
    }
    if (entry.endsWith(".ts")) {
      continue;
    }
    let contents = await readFile(source);
    if (entry === "index.html") {
      contents = Buffer.from(
        contents
          .toString("utf8")
          .replace("%CHRONOFISH_WEB_VERSION%", packageJson.version),
        "utf8"
      );
    }
    await writeFile(target, contents);
  }
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: "inherit", shell: process.platform === "win32" });
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command} ${args.join(" ")} exited with ${code}`));
      }
    });
  });
}
