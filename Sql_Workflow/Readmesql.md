# SQL Workflow — GTZAN Music Genre Classification

## Goal

Built a metadata-only PostgreSQL database (`music_genre_db`) following 

## Database Schema

music_genre (1) ──→ (M) audio_track + vw_clean_tracks view

![ERD diagram](../music-genre-classification/image/Music_db_ERD.png)



**2 tables + 1 view:**
- `music_genre` → stores the 10 genre labels
- `audio_track` → stores metadata for each .wav file
- `vw_clean_tracks` → filters corrupted and duplicate tracks



## What Metadata We Store in database

**music_genre:**
- `genre_id`, `genre_name`

**audio_track:**
- `track_id`, `genre_id`, `file_name`, `file_path`
- `duration_sec`, `sample_rate`, `channels`, `file_size_bytes`
- `file_hash`, `duplicate_flagged`, `corrupted_flagged`
- `split`, `created_at`

---

## Decisions & Justification

**1. MD5 file hash**
Used to detect duplicate files. Found 28 tracks with identical 
audio content that could cause data leakage across splits.

**2. corrupted_flagged + duplicate_flagged**
Flags bad tracks without deleting them — keeps full audit trail 
of the original dataset.

**3. split = NULL for excluded tracks**
Corrupted and duplicate tracks are excluded from train/val/test 
by setting `split = NULL`. They remain in the database for transparency.

**4. Stratified 70/15/15 split with seed 42**
Applied per genre to ensure equal representation across splits.
Seed 42 guarantees all teammates get identical splits.

**5. vw_clean_tracks view**
Filters corrupted and duplicate tracks automatically so they 
never reach the model.

---

## Final Results

| Status   | Count |
|----------|-------|
| Train    | 677   |
| Val      | 141   |
| Test     | 153   |
| Excluded | 29    |
| **Total**| **1,000** |

