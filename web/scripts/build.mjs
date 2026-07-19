import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import * as esbuild from "esbuild";

const root = process.cwd();
const src = path.join(root, "src");
const dist = path.join(root, "dist");
const packageJson = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

await esbuild.build({
  entryPoints: [
    path.join(src, "main.ts"),
    path.join(src, "ai-worker.ts"),
    path.join(src, "cpu-ai-worker.ts"),
    path.join(src, "training-worker.ts"),
    path.join(src, "training-label-worker.ts")
  ],
  outdir: dist,
  bundle: true,
  format: "esm",
  target: "es2022",
  sourcemap: true,
  splitting: true,
  chunkNames: "chunks/[name]-[hash]",
  entryNames: "[dir]/[name]",
  platform: "browser",
  plugins: [appVersionPlugin()],
  logLevel: "info"
});

await Promise.all([
  copyHtml(),
  copyCss(),
  writeFile(
    path.join(dist, "app-version.js"),
    `export const APP_VERSION = ${JSON.stringify(packageJson.version)};\n`,
    "utf8"
  )
]);

async function copyHtml() {
  const html = await readFile(path.join(src, "index.html"), "utf8");
  await writeFile(
    path.join(dist, "index.html"),
    html.replace("%CHRONOFISH_WEB_VERSION%", packageJson.version),
    "utf8"
  );
}

async function copyCss() {
  await writeFile(
    path.join(dist, "styles.css"),
    await readFile(path.join(src, "styles.css")),
    "utf8"
  );
}

function appVersionPlugin() {
  const appVersionPath = path.join(src, "app-version.ts");
  return {
    name: "chronofish-app-version",
    setup(build) {
      build.onLoad({ filter: /app-version\.ts$/ }, (args) => {
        if (path.normalize(args.path) !== path.normalize(appVersionPath)) {
          return null;
        }
        return {
          loader: "ts",
          contents: `export const APP_VERSION = ${JSON.stringify(packageJson.version)};\n`
        };
      });
    }
  };
}
