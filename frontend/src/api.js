// Thin wrapper around the Flask API. In dev, Vite proxies /api -> http://127.0.0.1:5000.
const BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request("/api/health"),
  listPapers: () => request("/api/papers"),
  getPaper: (id) => request(`/api/papers/${id}`),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/papers", { method: "POST", body: form });
  },
  remove: (id) => request(`/api/papers/${id}`, { method: "DELETE" }),
  ask: (id, question) => request(`/api/papers/${id}/ask`, json({ question })),
  search: (id, query) => request(`/api/papers/${id}/search`, json({ query, k: 6 })),
  history: (id) => request(`/api/papers/${id}/history`),
  reportUrl: (id, format) => `${BASE}/api/papers/${id}/report?format=${format}`,
};
