// Client-side filtering/sorting/pagination for /videos/. State lives in the URL query.
type Row = { i: string; t: string; a: string; d: string | null; l: number | null; n: number | null; k: string[]; p: string[]; r: 0 | 1 };

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const fmtDur = (s: number | null) => {
  if (!s) return "";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
};
const fmtDate = (d: string | null) =>
  d ? new Date(`${d}T00:00:00Z`).toLocaleDateString("en-GB", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" }) : "";
// Arabic-insensitive normalisation: strip diacritics/tatweel, unify alef/ya/ta marbuta forms.
const norm = (s: string) =>
  s.toLowerCase().normalize("NFKD").replace(/[̀-ًͯ-ٰٟـ]/g, "")
    .replace(/[أإآ]/g, "ا").replace(/ى/g, "ي").replace(/ة/g, "ه");

function card(v: Row, base: string): string {
  return `<article class="group relative flex flex-col rounded-xl bg-surface border border-line overflow-hidden hover:border-accent/60 hover:shadow-md transition">
  <div class="relative aspect-video bg-surface-2">
    <img src="https://i.ytimg.com/vi/${v.i}/mqdefault.jpg" alt="" loading="lazy" decoding="async" width="320" height="180" class="size-full object-cover" />
    ${v.l ? `<span class="absolute bottom-1.5 right-1.5 rounded bg-black/75 px-1.5 py-0.5 text-[11px] font-medium text-white">${fmtDur(v.l)}</span>` : ""}
    ${v.r ? `<span class="absolute top-1.5 right-1.5 rounded bg-accent px-1.5 py-0.5 text-[11px] font-semibold text-accent-ink">Transcript</span>` : ""}
  </div>
  <div class="flex flex-col gap-1.5 p-3.5">
    <h3 class="font-serif font-semibold leading-snug line-clamp-3"><a href="${base}v/${v.i}/" class="after:absolute after:inset-0 group-hover:text-accent"${v.t ? "" : ' lang="ar" dir="rtl"'}>${esc(v.t || v.a)}</a></h3>
    ${v.t ? `<p lang="ar" dir="rtl" class="text-sm text-muted line-clamp-1">${esc(v.a)}</p>` : ""}
    <p class="text-xs text-muted mt-auto pt-1">${fmtDate(v.d)}</p>
  </div>
</article>`;
}

export async function initBrowser(root: HTMLElement) {
  const form = root.querySelector<HTMLFormElement>("#filters")!;
  const results = root.querySelector<HTMLElement>("#results")!;
  const pager = root.querySelector<HTMLElement>("#pager")!;
  const count = document.getElementById("count")!;
  const base = root.dataset.base!;
  const pageSize = Number(root.dataset.page);
  const rows: Row[] = await (await fetch(root.dataset.index!)).json();
  const searchText = new Map(rows.map((r) => [r.i, norm(`${r.t} ${r.a}`)]));
  let page = 1;

  // Restore state from the URL.
  const params = new URLSearchParams(location.search);
  for (const el of form.elements as unknown as HTMLInputElement[]) {
    if (!el.name || !params.has(el.name)) continue;
    if (el.type === "checkbox") el.checked = params.get(el.name) === "1";
    else el.value = params.get(el.name)!;
  }
  page = Number(params.get("page")) || 1;

  function state() {
    const f = new FormData(form);
    return {
      q: norm(String(f.get("q") ?? "").trim()),
      topic: String(f.get("topic") ?? ""),
      series: String(f.get("series") ?? ""),
      year: String(f.get("year") ?? ""),
      sort: String(f.get("sort") ?? "newest"),
      transcript: f.get("transcript") != null,
      len: String(f.get("len") ?? ""),
    };
  }

  function apply(resetPage: boolean) {
    if (resetPage) page = 1;
    const s = state();
    const words = s.q.split(/\s+/).filter(Boolean);
    let out = rows.filter((r) => {
      if (s.transcript && !r.r) return false;
      if (s.year && !r.d?.startsWith(s.year)) return false;
      if (s.topic && !r.k.some((k) => (s.topic.endsWith("/") ? k.startsWith(s.topic) : k === s.topic))) return false;
      if (s.series === "-" ? r.p.length > 0 : s.series && !r.p.includes(s.series)) return false;
      const l = r.l ?? 0;
      if (s.len === "s" && l >= 1200) return false;
      if (s.len === "m" && (l < 1200 || l > 3600)) return false;
      if (s.len === "l" && l <= 3600) return false;
      if (words.length) {
        const t = searchText.get(r.i)!;
        if (!words.every((w) => t.includes(w))) return false;
      }
      return true;
    });
    const by: Record<string, (a: Row, b: Row) => number> = {
      newest: (a, b) => (b.d ?? "").localeCompare(a.d ?? ""),
      oldest: (a, b) => (a.d ?? "9").localeCompare(b.d ?? "9"),
      views: (a, b) => (b.n ?? 0) - (a.n ?? 0),
      longest: (a, b) => (b.l ?? 0) - (a.l ?? 0),
      shortest: (a, b) => (a.l ?? 0) - (b.l ?? 0),
      title: (a, b) => (a.t || a.a).localeCompare(b.t || b.a),
    };
    out = out.sort(by[s.sort] ?? by.newest);
    const pages = Math.max(1, Math.ceil(out.length / pageSize));
    page = Math.min(page, pages);
    const slice = out.slice((page - 1) * pageSize, page * pageSize);
    count.textContent = out.length.toLocaleString("en");
    results.innerHTML = slice.length
      ? `<div class="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">${slice.map((v) => card(v, base)).join("")}</div>`
      : `<p class="rounded-xl border border-dashed border-line p-10 text-center text-muted">No videos match these filters.</p>`;
    pager.innerHTML =
      pages > 1
        ? `<button data-p="${page - 1}" ${page === 1 ? "disabled" : ""} class="rounded-lg border border-line px-3 py-1.5 disabled:opacity-40">← Prev</button>
           <span class="text-sm text-muted">Page ${page} of ${pages}</span>
           <button data-p="${page + 1}" ${page === pages ? "disabled" : ""} class="rounded-lg border border-line px-3 py-1.5 disabled:opacity-40">Next →</button>`
        : "";

    const q = new URLSearchParams();
    const raw = new FormData(form);
    for (const [k, v] of raw) if (v && !(k === "sort" && v === "newest")) q.set(k, v === "on" ? "1" : String(v));
    if (page > 1) q.set("page", String(page));
    history.replaceState(null, "", q.size ? `?${q}` : location.pathname);
  }

  let timer: number | undefined;
  form.addEventListener("input", (e) => {
    clearTimeout(timer);
    timer = window.setTimeout(() => apply(true), (e.target as HTMLElement).id === "f-q" ? 150 : 0);
  });
  form.addEventListener("reset", () => setTimeout(() => apply(true)));
  pager.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>("button[data-p]");
    if (!b || b.disabled) return;
    page = Number(b.dataset.p);
    apply(false);
    root.scrollIntoView({ behavior: "smooth" });
  });
  apply(false);
}
