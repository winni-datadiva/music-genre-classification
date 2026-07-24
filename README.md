# Music Genre Classification    
A capstone project for The Knowledge House - Data Science Fellowship     
</br>


## **<div align="center">**RHYTHMX: Master Rhythm & Genre**</div>**

***<div align="center">***Team Members: Fanta, Ian, Winni, & Ximena***</div>***

| **Name**         | **Role**                                        |
| :--------------- | :---------------------------------------------- |
| Winni Paul       | Team Lead; Modeling (train, select, tuning)     |
| Ximena Rodriguez | Exploratory Data Analysis Lead; SQL database    |
| Ian McBride      | Data Transformation & Quality Assessment        |
| Fanta Bamba      | Background Researcher; Documentation Specialist |


---
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-parameters">Project Parameters</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#goal">Goal</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#project-scaffolding">Project Scaffolding</a></li>
    <li><a href="#sprint-1">Sprint 1</a></li>
    <li><a href="#sprint-2">Sprint 2</a></li>
    <li><a href="#sprint-3">Sprint 3</a></li>
    <li><a href="#sprint-4">Sprint 4</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---
<!-- PROJECT PARAMETERS -->    
## **Project Parameters:**    
Streaming platforms, DJs, and music curators constantly face the challenge of organizing and recommending music at scale. Training a model to recognize genre from audio is a foundational problem in music information retrieval with direct industry relevance.

<p align="right">(<a href="#readme-top">back to top</a>)</p> 

---

**Dataset:**    
Data was retrieved from Kaggle - **GTZAN Dataset - Music Genre Classification**

The GTZAN dataset is one of the most popular public dataset to evaluate machine learning (listening) for music genre recognition (MGR). The audio files were collected from 2000-2001 from a variety of sources. Audio data sources include personal CDs, microphone recordings, radio, in order to represent a variety of recording conditions. Key characteristics are: 

- 10 genres with 100 audio files each
- 30 seconds = length of audio files
- images original (visual representation of each audio file)
- 2 CSV files containing audio files features
  - mean and variance computed over multiple features (for each 30s song)
  - mean and variance computed over multiple features (each song split into 3s, 10x)


This dataset is licensed under a Creative Commons Attribution 4.0 International, (CC BY 4.0) license which allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given. View the dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

**Goal:**    
Build and train a model to classify music genre from audio clips using the GTZAN dataset. Some analytical questions and goals to consider include:    

● **Spectrogram/MFCC:** how do the spectrogram and MFCC representations differ across genres and what patterns emerge during EDA? <br>
● **Genres:** which genres are hardest for the model to distinguish and does that align with musical similarities between them? <br>
● **Interactive Demo:** how can you build an interactive demo where a user uploads an audio clip and the app returns the predicted genre? <br>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

**Citation:** 
Andrada. (2020). GTZAN Dataset - Music Genre Classification, Version 1 . Retrieved [month DD, YYYY] from https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---
<br/> 

**Project Scaffolding:**    
    
```
music-genre-classification/
    ├── .devcontainer/
    │   └── devcontainer.json
    ├── .vs/
    │   └── ProjectSettings.json
    ├── .vscode/
    │   ├── extensions.json
    │   ├── launch.json
    │   └── settings.json
    ├── Code/
    │   ├── __pycache__/
    │   │   ├── api_app_genre.cpython-311.pyc
    │   │   └── model.cpython-311.pyc
    │   ├── embeddings/
    │   │   ├── confusion_matrices_comparison.png
    │   │   ├── cost_efficiency_comparison.png
    │   │   ├── extraction_timing.csv
    │   │   ├── grad_cam_example.png
    │   │   ├── mert_test_labels.csv
    │   │   ├── mert_test.npy
    │   │   ├── mert_train_labels.csv
    │   │   ├── mert_train.npy
    │   │   ├── mert_val_labels.csv
    │   │   ├── mert_val.npy
    │   │   ├── musicnn_test_labels.csv
    │   │   ├── musicnn_test.npy
    │   │   ├── musicnn_train_labels.csv
    │   │   ├── musicnn_train.npy
    │   │   ├── musicnn_val_labels.csv
    │   │   └── musicnn_val.npy
    │   ├── api_app_genre.py
    │   ├── best_genre_cnn.pt
    │   ├── classification_report.csv
    │   ├── classification_report.md
    │   ├── classification_report.png
    │   ├── clean-preprocess.ipynb
    │   ├── CNN_Training.ipynb
    │   ├── EDA.ipynb
    │   ├── macro_f1_summary.png
    │   ├── mert_setup_notes.txt
    │   ├── model.py
    │   ├── musicnn_mert_results.md
    │   ├── preprocess_clean_final.ipynb
    │   ├── Sprint4_Comparison_Workflow.md
    │   ├── Sprint4_Embedding_Extraction.ipynb
    │   ├── Sprint4_Model_Comparison.ipynb
    │   ├── streamlit_app_genre.py
    │   └── train_norm_stats.json
    ├── Data_Music/
    │   └── .gitkeep
    ├── Doc/
    │   ├── ERD.md
    │   ├── image.png
    │   ├── project_scope.md
    │   ├── projscope.txt
    │   ├── READMEw.md
    │   ├── Research_FAQ.md
    │   └── Sprint1.md
    ├── image/
    │   ├── Audio_Duration_Distribution.png
    │   ├── confusion_matrix.png
    │   ├── Correlation_Heapmap.png
    │   ├── Mel_Spectrog.png
    │   ├── MFCCs.png
    │   ├── Music_db_ERD.png
    │   ├── Numberofaudioclipspergenre.png
    │   ├── Spectral_centroid.png
    │   ├── training_history.png
    │   └── Waveforms.png
    ├── Sql_Workflow/
    │   ├── __pycache__/
    │   │   └── api_app_genre.cpython-311.pyc
    │   ├── api_app_genre.py
    │   ├── genre_mlp_model.pt
    │   ├── MLP_Training.ipynb
    │   ├── PyTorch_Data_Pipeline.ipynb
    │   ├── Readmesql.md
    │   ├── Setting_Database_team .md
    │   ├── Sql_Workflow.ipynb
    │   └── streamlit_app_genre.py
    ├── .gitignore
    ├── cleaning.ipynb
    ├── CMakeLists.txt
    ├── Context_Music Genre Classification.txt
    ├── LICENSE
    ├── README.md
    └── requirements.txt
```

7 directories, 46 files
</br>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### **Sprint 1:**    
    
**GTZAN Dataset Quick Facts**
- 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
  - 100 audio files in each genre
- 30 second audio files (.wav files)
- images original folder contains visual representation for each audio file
- 2 .csv files
  - Features 30 sec (mean & variance for multiple features of the 30 second audio files)
  - Features 3 sec (songs split into 3 second audio files; mean & variance for multiple features)

<br/>

![Analog vs Digital](https://media.geeksforgeeks.org/wp-content/uploads/20260225170952274849/signals.webp)

**The Translation Problem: Infinity (Audio) vs Binary (Digital)**    
To the human ear, sound is a continuous wave of physical pressure. This acoustic experience is ***infinite*** (*continuous*) and unbroken. On the flip side, computers only understand *discrete* ***binary*** numbers. For the machine learning to work, we need to translate ***infinite sound*** into a ***finite mathematical language.***
</br>

<br/>

**Conversions: Making (sound)Waves & Taking Samples**    
A waveform is a simple snapshot - a 2D visual representation of sound, plotting the overall fluctuation of sound pressure over time.

![Spectrogram](https://media.geeksforgeeks.org/wp-content/uploads/20231006162533/Screenshot-2023-10-06-162421-min.png)

In the ***Sampling Process***, computer will capture continous waves by taking thousands of rapid, discrete "snapshots" per second. Thus the ***Sample Rate*** is the number of snapshots taken per second (i.e. 44.1 kHz = 44,100 samples per second.)

There's a limit to what the ***2D Waveform*** can capture. It only shows overall sound pressure aka mashing every concurrent frequency into a single, indistinguishable, and inseparable line. The computer can't tell apart a guitar from a flute if they're playing at the exact same time.

To level up from 2D to 3D, a ***Spectrogram*** uses a color spectrum to represent the third dimension (*Amplitude*). This separates instruments, analyzes pitch, pointing to the specific frequencies where the signal's acoustic *energy* is highest. This will extract the strength of a signal over a given frequency range, revealing exactly which frequencies are active at any given millisecond. Transforming a single line into a rich topographical *map of sound*.

Dimension 1 (X-axis): Time    
Dimension 2 (Y-axis): Frequency (pitch)    
Dimension 3 (Z-axis): Amplitude / Energy    

</br> 

<br/>

**Hypotheses at end of Phase 1**

***H1 - Genres are separable***
Spectrograms and MFCCs already look different per genre. The model has something real to learn from.    

***H2 - MFCCs will be the strongest signal***
Timbre is what makes a genre sound like itself - and MFCCs capture exactly that.    

**H3 = Accuracy is a fair metric** 
Because every genre has exactly 100 clips, accuracy won't be misleading.    

**H4 - Some genres will get confused** 
Blues, country, and rock share too many ingredients. Expect them to overlap in the confusion matrix.    

#### ***We found that genres have recipes: some are unique, some overlap and the model will be only as good as how different those recipes are.***    

</br> 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<br/> 

### **Sprint 2:** 

**Cleaning & Preprocessing**  
Cleaning and preprocessing were integral to Sprint 2 deliverables. Constants were set for the required directory. Dennited utilities for finding audio files. Metadata was checked for any corrupt files. For ease of use in the remaining sprints, metadata saved as a dataframe. Utilizing libraries such as ***librosa***, ***pydub***, and ***soundfile***, allowed us to pad, truncate, and/or copy files as needed for processing. Concluded with combining metadata and extracted features into one dataframe.  

**Extracted Features List**
- tempo
- chroma
- MFCCs (13 coefficients)
- spectral centroid
- spectral rolloff
- zero crossing rate

**SQL Metadata Datbase**
Built initial SQL metadata database using SQLite. Upon further research, PostgreSQL database was determined to be a better fit for our project.

**Dataset & Dataloader**
Custom PyTorch dataloader created to prevent data leakage during training, validation, and split.

</br> 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<br/> 

### **Sprint 3:**

**Model Architecture, Training, & Evaluation**

![Custom GenreCNN Classiciation Report](/Code/classification_report.png)

**Comparing Rhymthx's GenreCNN vs pretrained models (musicnn, MERT)**

![Macro F1 Summary](/Code/macro_f1_summary.png)

![Confusion Matrix Comparisons](/Code/embeddings/confusion_matrices_comparison.png) 

![Cost Efficiency Comparisons](/Code/embeddings/cost_efficiency_comparison.png) 

For both logistic regression and small MLP training, **musicnn** outperformed ***MERT***. The logistic regression results show musicnn doing better on both validation and test sets. Additionally, it uses slightly less model parameters. The MLP results show an even bigger gap, **musicnn achieved 86.27% test accuracy** compared to *74.51%* for MERT, while also using **1,920 fewer parameters**. **musicnn is the clear winner with a major performance advantage when using the MLP classifier**

</br> 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<br/> 

### **Sprint 4:**

**Deployment Decision**

 | **Question**                                                         | **Answer**                                                                                                                                                                                                                                                                                                                                  |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Which model wins on macro F1?                                        | **musicnn + MLP** with _**test macro F1 = 0.8644**_                                                                                                                                                                                                                                                                                         |
| Which model is cheapest to run at inference (latency + params)?      | **Rhythmx GenreCNN** is the cheapest with _**33,130 params & 0.076 s/clip**_. Compared with musicnn's _97,802 params & 1.649 s/clip_ and MERT's _99,722 params & 7.567 s/clip_                                                                                                                                                              |
| Does the winner justify the extra complexity/dependencies (TF + HF)? | Not for this project. Though _**musicnn**_ provides the highest accuracy but deployment requires maintaining TensorFlow, PyTorch (MLP), and feature extraction pipeline. This increases app size, startup time, dependency management, complexity, etc. The accuracy gain comes at a much higher inference cost.                            |
| Final choice for streamlit_app_genre.py / api_app_genre.py           | Deploy **Rhythmx GenreCNN** as it's a single end-to-end PyTorch model that requires no external embedding extraction, has the smallest model size, and performs inference about **22x faster** than musicnn _**(0.076 vs 1.649 s/clip)**_. Therefore it's the simplest, fastest, and most maintainable model for a production demo.<br><br> |

---

The **RhythmX GenreCNN** has the lowest computational cost, requiring only **33,130 parameters** and achieving an average inference time of **0.076 seconds per clip**, roughly **22 times faster** than ***musicnn*** and nearly **100 times faster** than ***MERT***.

- simplest deployment architecture (single PyTorch model)
- lowest inference latency
- smallest model footprint
- no TensorFlow or Hugging Face dependency chain
- easier maintenance and portability
best engineering choice for deployment
