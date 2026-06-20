# GTZAN Music Genre Classification SQL 

## Project Overview

This project builds a **music genre classification system** using the **GTZAN dataset** from Kaggle. We extract audio features using Librosa, store them in PostgreSQL, and prepare the data for machine learning models.

**Team:** **`Master of Rhythms Team`**, [Winni], [Fanta], [Ian],[Ximena]

# Data Loading & Feature Extraction

1. **Downloaded GTZAN Dataset** (1000 .wav files, 10 genres, 100 per genre)
2. **Created PostgreSQL Database** with normalized schema (4 tables)
3. **Extracted Audio Features** using Librosa:
   - Tempo (BPM)
   - Chroma (pitch class energy)
   - MFCCs (13 Mel Frequency Cepstral Coefficients)
   - Spectral Centroid (brightness)
   - Spectral Rolloff (high-frequency cutoff)
   - Zero Crossing Rate (noise/percussion)
4. **Loaded Data** into PostgreSQL for analysis and modeling


### Dataset Statistics Summary

| Metric | Value |
|--------|-------|
| Total audio files | 1,000 |
| Genres | 10 |
| Files per genre | 100 |
| Audio duration | 30 seconds each |
| Sample rate | 22,050 Hz |
| Total features extracted | 999 tracks × (1 + 10 segments) |

![Schema database](/image/gtzan_schema_erd_cardinality.png)



### Table 1: `music_genre`

**Purpose:** Stores the 10 genre labels

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Unique genre ID (1-10) |
| `name` | VARCHAR(50) | Genre name (blues, jazz, rock, etc.) |

**Example:**
```
id | name
---|----------
1  | blues
2  | classical
3  | country
...
10 | rock
```

**Relationship:** Parent table. One genre has many audio tracks (1:M).

---

### Table 2: `audio_track`

**Purpose:** Records each physical audio file in the dataset

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Unique track ID (1-999) |
| `genre_id` | INT FK | Foreign key → `music_genre.id` |
| `filename` | VARCHAR(255) | e.g., `blues.00000.wav` |
| `file_path` | VARCHAR(512) | Full path on disk |

**Example:**
```
id  | genre_id | filename         | file_path
----|----------|------------------|------------------------------------------
1   | 1        | blues.00000.wav  | /Users/.../Music_Data/genres_original/...
2   | 1        | blues.00001.wav  | /Users/.../Music_Data/genres_original/...
... | ...      | ...              | ...
999 | 10       | rock.00099.wav   | /Users/.../Music_Data/genres_original/...
```

**Relationships:**
- Many-to-one with `music_genre` (each track belongs to one genre)
- One-to-one with `features_30_sec` (each track has exactly 1 feature set for full 30-sec)
- One-to-many with `features_3_sec` (each track has 10 feature sets, one per 3-sec segment)

**Indexes:**
- `idx_audio_track_genre_id` — speeds up queries filtering by genre

---

### Table 3: `features_30_sec`

**Purpose:** Stores extracted audio features for the **full 30-second track**

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Unique feature ID |
| `track_id` | INT FK | Foreign key → `audio_track.id` (UNIQUE) |
| `length` | INT | Number of audio samples (always 661,500 for 30-sec @ 22,050 Hz) |
| `tempo` | DOUBLE | Beats per minute (BPM) |
| `chroma_mean` | DOUBLE | Average pitch class energy (0-1) |
| `mfcc1_mean` to `mfcc13_mean` | DOUBLE (×13) | Mel Frequency Cepstral Coefficients |
| `spectral_centroid` | DOUBLE | Frequency center of mass (Hz) |
| `spectral_rolloff` | DOUBLE | Frequency where 85% of power is below (Hz) |
| `zero_crossing_rate` | DOUBLE | Average zero crossings per sample (0-1) |

**Example (first row):**
```
id  | track_id | length  | tempo   | chroma_mean | mfcc1_mean | ... | spectral_centroid | zero_crossing_rate
----|----------|---------|---------|-------------|------------|-----|-------------------|-------------------
1   | 1        | 661500  | 113.28  | 0.528       | -113.59    | ... | 1732.45           | 0.108
2   | 2        | 661500  | 89.32   | 0.612       | -207.52    | ... | 1530.26           | 0.056
... | ...      | ...     | ...     | ...         | ...        | ... | ...               | ...
```

**Relationships:**
- One-to-one with `audio_track` (enforced by UNIQUE constraint on `track_id`)
- Each .wav file has exactly ONE row here

**Indexes:**
- `UNIQUE idx_features_30_sec_track_id` — ensures 1:1 relationship

**Use Case:** Model training (each sample = one full 30-second track)

---

### Table 4: `features_3_sec`

**Purpose:** Stores extracted audio features for **each 3-second segment** of a track

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Unique feature ID |
| `track_id` | INT FK | Foreign key → `audio_track.id` |
| `segment_num` | INT | Segment number (0-9, one per 3 seconds) |
| `start_time_sec` | DOUBLE | Start time of segment in seconds (0, 3, 6, ..., 27) |
| `tempo`, `chroma_mean`, `mfcc1_mean`...`mfcc13_mean`, `spectral_centroid`, `spectral_rolloff`, `zero_crossing_rate` | DOUBLE | Same 18 features as `features_30_sec`, computed per segment |

**Example (segments of track 1):**
```
id  | track_id | segment_num | start_time_sec | tempo  | chroma_mean | ... | zero_crossing_rate
----|----------|-------------|----------------|--------|-------------|-----|-------------------
1   | 1        | 0           | 0.0            | 120.5  | 0.532       | ... | 0.110
2   | 1        | 1           | 3.0            | 115.2  | 0.531       | ... | 0.112
3   | 1        | 2           | 6.0            | 111.4  | 0.525       | ... | 0.105
... | 1        | ...         | ...            | ...    | ...         | ... | ...
11  | 1        | 9           | 27.0           | 118.9  | 0.540       | ... | 0.108
12  | 2        | 0           | 0.0            | 98.3   | 0.610       | ... | 0.055
... | ...      | ...         | ...            | ...    | ...         | ... | ...
```

**Row Count:** 999 tracks × 10 segments = 9,990 rows

**Relationships:**
- Many-to-one with `audio_track` (each track has 10 rows here)

**Indexes:**
- `idx_features_3_sec_track_id` — speeds up queries by track
- `idx_features_3_sec_segment_num` — speeds up queries by segment

**Use Case:** Temporal analysis (how features change within a 30-sec song)

## Schema Diagram (Text Format)

```
┌──────────────────┐
│   music_genre    │
├──────────────────┤
│ id (PK)          │◄─────┐
│ name             │      │
└──────────────────┘      │ 1:M
                          │
                    ┌─────┴──────────────┐
                    │                    │
        ┌───────────▼──────────┐   ┌────▼────────────────┐
        │   audio_track        │   │   features_30_sec   │
        ├──────────────────────┤   ├─────────────────────┤
        │ id (PK)              │   │ id (PK)             │
        │ genre_id (FK) ───────┤   │ track_id (FK) ◄─────┤ 1:1
        │ filename             │   │ length              │ (UNIQUE)
        │ file_path            │   │ tempo               │
        └────────┬─────────────┘   │ chroma_mean         │
                 │                 │ mfcc1_mean...       │
                 │ 1:M             │ mfcc13_mean         │
                 │                 │ spectral_centroid   │
                 │                 │ spectral_rolloff    │
                 │                 │ zero_crossing_rate  │
                 │                 └─────────────────────┘
                 │
        ┌────────▼──────────────┐
        │  features_3_sec       │
        ├───────────────────────┤
        │ id (PK)               │
        │ track_id (FK) ◄───────┤ 1:M (10 rows per track)
        │ segment_num (0-9)     │
        │ start_time_sec        │
        │ tempo                 │
        │ chroma_mean           │
        │ mfcc1_mean...         │
        │ mfcc13_mean           │
        │ spectral_centroid     │
        │ spectral_rolloff      │
        │ zero_crossing_rate    │
        └───────────────────────┘
```

---