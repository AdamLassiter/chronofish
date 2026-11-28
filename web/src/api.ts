/** Authenticated JSON requests to the Chronofish server. */

import { state } from "./state.js";
import { ApiOptions } from "./types.js";

export async function api<T>(
  path: string,
  options: ApiOptions = {},
): Promise<T> {
  const { body: value, ...requestOptions } = options;
  const headers = new Headers(options.headers);
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  const body = value === undefined ? undefined : JSON.stringify(value);
  if (body !== undefined) headers.set("Content-Type", "application/json");

  const response = await fetch(path, { ...requestOptions, headers, body });
  const payload: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      typeof payload === "object" &&
      payload !== null &&
      "error" in payload &&
      typeof payload.error === "string"
        ? payload.error
        : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return payload as T;
}
