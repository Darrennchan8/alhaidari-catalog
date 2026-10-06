// Build-time access to the JSON produced by `catalog export` (data/site/).
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Category, Playlist, Stats, Video, VideoSummary } from "./types";

const DATA_DIR = resolve(process.env.CATALOG_SITE_DATA ?? resolve(process.cwd(), "../data/site"));

const cache = new Map<string, unknown>();
function load<T>(rel: string): T {
  if (!cache.has(rel)) {
    try {
      cache.set(rel, JSON.parse(readFileSync(resolve(DATA_DIR, rel), "utf-8")));
    } catch (err) {
      throw new Error(
        `Cannot read ${rel} from ${DATA_DIR}. Run \`uv run catalog export\` in pipeline/ first.\n${err}`,
      );
    }
  }
  return cache.get(rel) as T;
}

export const getStats = () => load<Stats>("stats.json");
export const getVideos = () => load<VideoSummary[]>("index.json");
export const getPlaylists = () => load<Playlist[]>("playlists.json");
export const getCategories = () => load<Category[]>("topics.json");

/** Full record incl. transcript; not cached (only needed once per page). */
export function getVideo(id: string): Video {
  return JSON.parse(readFileSync(resolve(DATA_DIR, "videos", `${id}.json`), "utf-8")) as Video;
}

let byId: Map<string, VideoSummary> | undefined;
export function videoById(id: string): VideoSummary | undefined {
  byId ??= new Map(getVideos().map((v) => [v.id, v]));
  return byId.get(id);
}

let topicNames: Map<string, { category: Category; name: string }> | undefined;
export function topicInfo(key: string) {
  if (!topicNames) {
    topicNames = new Map();
    for (const c of getCategories())
      for (const s of c.subtopics) topicNames.set(s.key, { category: c, name: s.name });
  }
  return topicNames.get(key);
}

export function playlistBySlug(slug: string) {
  return getPlaylists().find((p) => p.slug === slug);
}

/** Videos of a playlist, in playlist order. */
export function playlistVideos(p: Playlist): VideoSummary[] {
  return p.video_ids.map(videoById).filter((v): v is VideoSummary => !!v);
}
