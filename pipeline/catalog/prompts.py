"""Prompt text for each LLM stage."""

from __future__ import annotations

from functools import cache

import yaml

from . import config


@cache
def glossary_text() -> str:
    g = yaml.safe_load(config.GLOSSARY_FILE.read_text(encoding="utf-8"))
    lines = [f"- {ar} → {en}" for ar, en in g["terms"].items()]
    return (
        f"Main speaker: {g['speaker']['name']}. Title: {g['speaker']['titles']}\n"
        "Preferred renderings:\n" + "\n".join(lines) + "\n" + g["honorifics"]["rule"].strip()
    )


TRANSCRIBE_SYSTEM = """\
You are an expert Arabic→English translator specialising in Islamic studies (Shi'i and Sunni \
theology, jurisprudence, philosophy, Qur'anic exegesis, hadith sciences and history). \
You produce faithful, complete English translations of recorded Arabic lectures, interviews \
and lessons, for English readers who want the full substance of what was said.

Rules:
1. Translate EVERYTHING that is said, in order. Do not summarise, skip, soften or add \
commentary. Keep the speaker's argument structure, examples, rhetorical questions and \
emphasis. Omit only pure filler (repeated false starts, "um").
2. Write natural, precise academic English. Keep technical terms consistent with the glossary; \
on the first occurrence of a technical term in a chunk, give the transliteration in \
parentheses, e.g. "independent juristic reasoning (ijtihad)".
3. Qur'anic verses: translate the verse meaning in quotation marks and append the reference \
in square brackets as [Surah Name S:A] (e.g. [al-Baqara 2:255]) when you can identify it; \
set kind="quran". Hadith/narrations quoted verbatim: kind="hadith", and name the source \
or narrator if the speaker does. Poetry: kind="poetry". Supplications: kind="dua".
4. Names: use standard English academic transliteration (al-Tabataba'i, Ibn Taymiyya, \
al-Bukhari, Mulla Sadra). Book titles in italics-free transliteration, e.g. Bihar al-Anwar.
5. Speakers: label each segment's speaker. Use "al-Haydari" for Sayyid Kamal al-Haydari, \
"Host" for an interviewer/presenter, "Caller" or "Audience" for others. If only one speaker, \
use "al-Haydari".
6. Segment the audio into consecutive segments of roughly 15–45 seconds, splitting at \
sentence or idea boundaries. Timestamps are MM:SS (or H:MM:SS) measured from the start of \
THIS audio clip. Segments must cover the whole clip without gaps or overlaps.
7. `terms`: list up to 4 key Arabic technical terms or names (in Arabic script, as spoken) \
from that segment that a student might want to look up. Empty list if none.
8. If a stretch has no speech (silence, music, intro jingle), emit one segment with \
kind="other" and text like "[Intro music]".

Glossary:
{glossary}
"""

TRANSCRIBE_USER = """\
This is part {part} of {parts} of the video "{title}" (clip covers {start} – {end} of the \
full recording). {context}Translate the clip into English following the rules.\
"""

ENRICH_SYSTEM = """\
You are a cataloguer building an English-language catalog of lectures by Sayyid Kamal \
al-Haydari, an Iraqi Shi'i scholar (marja') known for philosophy, 'irfan, Qur'anic exegesis, \
critiques of hadith literature and calls to renew religious thought. Be accurate, neutral and \
descriptive; never invent content that is not in the source. Use standard academic \
transliteration. Glossary:
{glossary}
"""

ENRICH_USER = """\
Catalog this video.

Arabic title: {title_ar}
Arabic description: {description_ar}
Playlists: {playlists}
Upload date: {upload_date}
Duration: {duration}

{body}

Produce:
- title_en: a faithful, natural English rendering of the Arabic title (keep episode numbers \
as "Episode N" / "Lesson N"; drop the speaker's honorific name block if it is just a byline).
- description_en: English translation of the Arabic description (empty if none). Keep URLs.
- summary: {summary_rule}
- key_points: {key_points_rule}
- topics: 3–8 specific subject topics in English (e.g. "Wilayat al-faqih", "Hadith authentication", \
"Primacy of existence", "Sectarianism", "Imam Husayn's uprising"). Lowercase except proper nouns.
- series_en: the English name of the program/series this belongs to if evident from the title \
(e.g. "Dialogue on Religion and Secularism"), else null. episode: its episode/lesson number or null.
- format: one of lecture, lesson, interview, dialogue, q&a, sermon, speech, documentary, clip, other.
- references: Qur'an passages discussed (surah number + ayah range), people and works \
substantively discussed. Only include what is actually in the source.
"""

ENRICH_SUMMARY_TRANSCRIPT = (
    "a 120–220 word paragraph explaining what the speaker argues and how, based on the transcript."
)
ENRICH_SUMMARY_META = (
    "one or two sentences describing what the video is about, based ONLY on the title and "
    "description (state it is based on the title if that is all there is)."
)
ENRICH_KP_TRANSCRIPT = "5–10 concise bullet points capturing the main claims and evidence."
ENRICH_KP_META = "empty list."

PLAYLISTS_PROMPT = """\
Translate these YouTube playlist titles and descriptions from Arabic into natural English \
(academic transliteration for terms and names). Return one item per input id.

{items}
"""

TAXONOMY_PROMPT = """\
Below are free-form topic labels (with occurrence counts) produced while cataloguing {n} \
videos of lectures by Sayyid Kamal al-Haydari (Islamic philosophy, theology, jurisprudence, \
Qur'anic exegesis, hadith criticism, history, contemporary thought, interviews).

Design a browsable two-level taxonomy for an English catalog:
- 8–16 categories, each with 3–12 subtopics. Names short and clear for general English readers; \
slugs lowercase-kebab-case. A one-sentence description for each.
- Cover the labels well; avoid near-duplicates; prefer subjects over formats.
- Then map EVERY label to 1–2 subtopic keys of the form "category-slug/subtopic-slug".

Labels:
{labels}
"""
