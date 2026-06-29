
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



## Code to delete the old tables in the database 

1. open you terminal write this 

/Applications/Postgres.app/Contents/Versions/latest/bin/psql -U user -d music_genre_db

then run 

DROP TABLE IF EXISTS features_3_sec CASCADE;
DROP TABLE IF EXISTS features_30_sec CASCADE;
DROP TABLE IF EXISTS audio_track CASCADE;
DROP TABLE IF EXISTS music_genre CASCADE;

