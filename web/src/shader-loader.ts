const shaderRequests = new Map<string, Promise<string>>();

export function loadShader(path: string): Promise<string> {
  const existing = shaderRequests.get(path);
  if (existing) {
    return existing;
  }

  const request = fetch(path).then(async (response) => {
    if (!response.ok) {
      throw new Error(`Could not load WebGPU shader ${path}: ${response.status} ${response.statusText}`);
    }
    return response.text();
  });
  shaderRequests.set(path, request);
  return request;
}
