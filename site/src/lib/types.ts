// Mirrors the pydantic models in pipeline/catalog/models.py (Site* classes).

export type SegmentKind = "speech" | "quran" | "hadith" | "poetry" | "dua" | "other";

export interface Segment {
  start: number;
  end: number;
  speaker: string | null;
  kind: SegmentKind;
  text: string;
  terms: string[];
}

export interface QuranRef {
  surah: number;
  ayah_start: number | null;
  ayah_end: number | null;
  note: string;
}

export interface References {
  quran: QuranRef[];
  people: string[];
  works: string[];
}

export interface Chapter {
  start: number;
  end: number;
  title: string;
}

export interface VideoSummary {
  id: string;
  title_en: string;
  title_ar: string;
  upload_date: string | null;
  duration: number | null;
  view_count: number | null;
  topics: string[];
  playlists: string[];
  series: string | null;
  episode: number | null;
  format: string | null;
  has_transcript: boolean;
  summary: string;
}

export interface Notice {
  start: number;
  end: number;
  message: string;
}

export interface Video extends VideoSummary {
  description_ar: string;
  description_en: string;
  key_points: string[];
  references: References;
  chapters: Chapter[];
  segments: Segment[];
  transcript_model: string | null;
  notices: Notice[];
}

export interface Playlist {
  id: string;
  slug: string;
  title_en: string;
  title_ar: string;
  description_en: string;
  video_ids: string[];
}

export interface Subtopic {
  slug: string;
  key: string;
  name: string;
  description: string;
  count: number;
}

export interface Category {
  slug: string;
  name: string;
  description: string;
  count: number;
  subtopics: Subtopic[];
}

export interface Stats {
  videos: number;
  transcribed: number;
  translated: number;
  with_topics: number;
  playlists: number;
  hours: number;
  transcribed_hours: number;
  channel_url: string;
  synced_at: string;
  exported_at: string;
}
