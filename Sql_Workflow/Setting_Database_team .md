
# Step 1 

I download [https://postgresapp.com/](https://postgresapp.com/) Then I start  the server and click in postgres to create the database.

CREATE DATABASE music_genre_db;

Then I check my terminal user with the command **whoami** 

# Step 2 
I create a file called .env and I add my credentials like this 

DB_USER=user_postgres
DB_PASSWORD=Password_postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=music_genre_db


# Step 3 
conda activate Enviroment 
I install libraries pip install psycopg2-binary sqlalchemy python-dotenv pandas

# Step 4 

Download dataset **`GTZAN Dataset - Music Genre Classification`**

# Step 5 
Then we created the ESDs Schema

![Schema database](/image/gtzan_schema_erd_cardinality.png)

**Table: music\_genre** 

* **Purpose:** Stores the 10 genre categories used to classify your tracks  
* **Attributes:** `id` (Primary Key), `name` (e.g. `blues`, `jazz`, `metal`)  
* **Relationship:** Parent table every `audio_track` points back to one row here via `genre_id (1:M)`

**Table: audio\_track (your .wav files)**

* **Purpose:** Records each physical audio file in local  `Data_Music/genres_original` folder  
* **Attributes:** `id` (Primary Key), `filename` (e.g. `blues.00000.wav`), `file_path` (the local path on your Mac), `genre_id` (Foreign Key)  
* **Relationship:** Central node  connects to the math feature tables (`features_30_sec` and features\_3\_sec) via track\_id **(1:1)**

**Table: features\_30\_sec (the audio features for the whole 30-sec clip)**

* **Purpose:** Stores the calculated numeric features tempo, tone color, frequency content  extracted from the full track  
* **Attributes:** `id` (PK), `track_id` (FK), `tempo`, `chroma_mean`, `mfcc1_mean...mfcc13_mean`, `spectral_centroid`, `spectral_rolloff`, `zero_crossing_rate`  
* **Relationship:** One-to-one with audio\_track (1:1)

**Table: features\_3\_sec (the audio features for each 3-sec slice)**

* **Purpose:** Same calculated numeric features, but computed separately for each of the 10 three-second segments  
* **Attributes:** id (PK), track\_id (FK), segment\_num (0–9), same feature columns as above  
* **Relationship:** One-to-many with audio\_track  (1:M)


# Step 6

Run notebook Sql_workflow 