// Transliterated surah names, index = surah number - 1.
export const SURAHS = [
  "al-Fatiha", "al-Baqara", "Al 'Imran", "al-Nisa'", "al-Ma'ida", "al-An'am", "al-A'raf", "al-Anfal", "al-Tawba", "Yunus",
  "Hud", "Yusuf", "al-Ra'd", "Ibrahim", "al-Hijr", "al-Nahl", "al-Isra'", "al-Kahf", "Maryam", "Ta Ha",
  "al-Anbiya'", "al-Hajj", "al-Mu'minun", "al-Nur", "al-Furqan", "al-Shu'ara'", "al-Naml", "al-Qasas", "al-'Ankabut", "al-Rum",
  "Luqman", "al-Sajda", "al-Ahzab", "Saba'", "Fatir", "Ya Sin", "al-Saffat", "Sad", "al-Zumar", "Ghafir",
  "Fussilat", "al-Shura", "al-Zukhruf", "al-Dukhan", "al-Jathiya", "al-Ahqaf", "Muhammad", "al-Fath", "al-Hujurat", "Qaf",
  "al-Dhariyat", "al-Tur", "al-Najm", "al-Qamar", "al-Rahman", "al-Waqi'a", "al-Hadid", "al-Mujadila", "al-Hashr", "al-Mumtahana",
  "al-Saff", "al-Jumu'a", "al-Munafiqun", "al-Taghabun", "al-Talaq", "al-Tahrim", "al-Mulk", "al-Qalam", "al-Haqqa", "al-Ma'arij",
  "Nuh", "al-Jinn", "al-Muzzammil", "al-Muddaththir", "al-Qiyama", "al-Insan", "al-Mursalat", "al-Naba'", "al-Nazi'at", "'Abasa",
  "al-Takwir", "al-Infitar", "al-Mutaffifin", "al-Inshiqaq", "al-Buruj", "al-Tariq", "al-A'la", "al-Ghashiya", "al-Fajr", "al-Balad",
  "al-Shams", "al-Layl", "al-Duha", "al-Sharh", "al-Tin", "al-'Alaq", "al-Qadr", "al-Bayyina", "al-Zalzala", "al-'Adiyat",
  "al-Qari'a", "al-Takathur", "al-'Asr", "al-Humaza", "al-Fil", "Quraysh", "al-Ma'un", "al-Kawthar", "al-Kafirun", "al-Nasr",
  "al-Masad", "al-Ikhlas", "al-Falaq", "al-Nas",
];

export function quranLabel(surah: number, a?: number | null, b?: number | null): string {
  const name = SURAHS[surah - 1] ?? `Surah ${surah}`;
  if (!a) return `${name} (${surah})`;
  return `${name} ${surah}:${a}${b && b !== a ? `–${b}` : ""}`;
}

export function quranUrl(surah: number, a?: number | null, b?: number | null): string {
  if (!a) return `https://quran.com/${surah}`;
  return `https://quran.com/${surah}/${a}${b && b !== a ? `-${b}` : ""}`;
}
