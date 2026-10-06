// Compact index for the client-side browser on /videos/.
import type { APIRoute } from "astro";
import { getVideos } from "~/lib/data";

export const GET: APIRoute = () => {
  const rows = getVideos().map((v) => ({
    i: v.id,
    t: v.title_en,
    a: v.title_ar,
    d: v.upload_date,
    l: v.duration,
    n: v.view_count,
    k: v.topics,
    p: v.playlists,
    r: v.has_transcript ? 1 : 0,
  }));
  return new Response(JSON.stringify(rows), { headers: { "Content-Type": "application/json" } });
};
