export function duration(seconds: number | null | undefined): string {
  if (!seconds) return "";
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
}

export function timestamp(seconds: number): string {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

export function date(iso: string | null | undefined): string {
  if (!iso) return "";
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function compact(n: number | null | undefined): string {
  if (n == null) return "";
  return new Intl.NumberFormat("en", { notation: "compact" }).format(n);
}

export const thumb = (id: string, q: "mq" | "hq" | "maxres" = "mq") =>
  `https://i.ytimg.com/vi/${id}/${q}default.jpg`;

/** Prefix a site-relative path with the configured base path. */
export function href(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

/** English title when translated, otherwise the Arabic original (flagged for lang/dir). */
export function title(v: { title_en: string; title_ar: string }): { text: string; ar: boolean } {
  return v.title_en ? { text: v.title_en, ar: false } : { text: v.title_ar, ar: true };
}

/** Attributes for an element whose text may be Arabic. */
export const langAttrs = (ar: boolean) => (ar ? { lang: "ar", dir: "rtl" } : {});
