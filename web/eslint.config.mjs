import js from "@eslint/js";
import tseslint from "typescript-eslint";

const browserGlobals = {
  AbortController: "readonly",
  ArrayBuffer: "readonly",
  Blob: "readonly",
  BroadcastChannel: "readonly",
  clearInterval: "readonly",
  clearTimeout: "readonly",
  console: "readonly",
  crypto: "readonly",
  document: "readonly",
  ErrorEvent: "readonly",
  EventSource: "readonly",
  fetch: "readonly",
  FileReader: "readonly",
  GPUBufferUsage: "readonly",
  GPUMapMode: "readonly",
  localStorage: "readonly",
  MessageEvent: "readonly",
  navigator: "readonly",
  performance: "readonly",
  requestAnimationFrame: "readonly",
  self: "readonly",
  setInterval: "readonly",
  setTimeout: "readonly",
  URL: "readonly",
  URLSearchParams: "readonly",
  WebAssembly: "readonly",
  window: "readonly"
};

const nodeGlobals = {
  Buffer: "readonly",
  clearTimeout: "readonly",
  console: "readonly",
  process: "readonly",
  setTimeout: "readonly",
  WebSocket: "readonly"
};

export default [
  {
    ignores: [
      "dist/**",
      "node_modules/**"
    ]
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.ts"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: browserGlobals
    },
    rules: {
      "no-undef": "off",
      "no-constant-binary-expression": "off",
      "no-useless-assignment": "off",
      "prefer-const": "off",
      "@typescript-eslint/no-empty-object-type": "off",
      "@typescript-eslint/no-unused-vars": "off"
    }
  },
  {
    files: [
      "scripts/**/*.mjs",
      "tests/**/*.js"
    ],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: nodeGlobals
    }
  }
];
