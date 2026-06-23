## Background Research & Frequently Asked Questions 
---
### Project Foudation/ Exploratory Data Analysis
#### -What is a waveform and how is audio represented as a signal over time using amplitude and sample rate?

Waveforms are visual representations of sound as time (continuous) on the x-axis and amplitude(continuous) on the y-axis. Because it is difficult to analyze because of the infinite possibilities; it is transformed into digital signal(binary 0-1) .  
Audio is represented as a signal by taking thousands of rapid "snapshots" of an analog sound wave per second. A signal is a representation of a sound. This process—converting continuous acoustic waves into discrete binary numbers—uses sample rate to track time along the X-axis and amplitude to measure the height of the wave (physical size of the waveform) on the Y-axis.
Analog signal( a system, device, or signal that represents data using continuously variable physical quantities, such as voltage, frequency, or pressure) 

#### -What is a spectrogram is and how does it convert a waveform into a 2D visual representation of frequency over time?
A spectrogram is a graph that displays the strength of a signal over time for a given frequency range. Using a color spectrum, it points to the frequencies where the signal’s energy is highest and shows the energy variation over time.
A standard waveform only shows how overall sound pressure or voltage fluctuates over time. A spectrogram breaks down that single, 2D waveform into three core dimensions: Time, Frequency, and Amplitude. [1, 2, 3, 4]
Because a 2D image is limited, color is used to represent the third dimension

#### -What are MFCCs (Mel Frequency Cepstral Coefficients) and why are they the standard feature extraction method for audio classification tasks? 
 It’s a feature used in automatic speech and speaker recognition. Essentially, it’s a way to represent the short-term power spectrum of a sound which helps machines understand and process human speech more effectively.
Mel Frequency Cepstral Coefficients (MFCCs) are the standard for audio classification because they mimic human auditory perception. By compressing raw audio data into a compact, highly descriptive set of features, they isolate the most critical acoustic characteristics while ignoring background noise
MFCCs are a part of frequency- domain representation. Mel-Frequency Cepstral Coefficients (MFCC): Represent the short-term power spectrum of a sound, widely used in speech and audio processing due to their effectiveness in capturing the phonetically relevant characteristics of the audio signal. See the MFCC block in Edge Impulse.
It is a feature extraction technique that transforms raw audio into a compact representation that makes it easier for the machine learning model to recognize patterns in the sound.
Each coefficient captures a different aspect of the sound spectrum.
One song → many MFCC coefficients → summarize each coefficient → create columns such as mfcc1_mean, mfcc1_var, mfcc2_mean, etc. These summarized columns become the features used by the machine learning model.
Music genre classification: Sound classification can be used to identify the genre of a particular song based on its audio features.

#### -How does Librosa work as a Python library for loading and analyzing audio files? 
Librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

#### -What does a sample rate mean and why consistency across your dataset matters for preprocessing?  
Sample rate is the rate at which an audio signal is sampled. It indicates how many times per second an audio signal is sampled during the analog-to-digital conversion( the process of translating continuous, real-world analog signals (like sound or light) into discrete digital data (1s and 0s) that computers can process)  process. So lets just assume I’m looking at a waveform being sampled at 44,100hz 44.1 kHz = 44,100 samples per second. Lets also assume that this waveform’s duration is 1 second. Time will be represented on the X axis of this image.
A higher sample rate means more samples are taken per second, leading to a more accurate digital representation of the original analog signal.
For a music genre classification dataset, consistency is especially important because the model is trying to learn the characteristics that distinguish genres (Pop, Rock, Hip-Hop, Country, etc.). If the data is inconsistent, the model may learn mistakes instead of genre-related patterns.

FYI: If the graph places the higher sample rate waveform at the bottom, that's usually just a visual design choice so the reader can compare multiple waveforms without them overlapping. The vertical position of each example is not representing sample rate magnitude.
The author stacks them vertically for comparison. The bottom graph isn't "lower quality" or "lower sample rate"; it's just positioned lower on the page.
The key thing to observe is that the higher-sample-rate waveform has more sample points along the time axis, allowing it to follow the original analog curve more closely.

#### -What class imbalance looks like in an audio dataset and how can it affect training? 
Imbalanced data occurs when one class has far more samples than others, causing models to favour the majority class and perform poorly on the minority class. This often results in misleading accuracy when categorizing a genre to an audio. In audio classification, this means the model will often correctly recognize frequent sounds (e.g., standard speech) but completely fail to detect rarer acoustic events (e.g., specific bird calls or glass shattering), treating them as background noise.

### Cleaning/Preprocessing 

#### -How to use Librosa to load audio files, check sample rates, and extract MFCCs and spectrograms? 
The primary function for loading an audio file is librosa.load(). This single function handles many complexities, such as decoding different audio formats (like .wav, .mp3) and converting the signal into a standardized numerical format.

When you call librosa.load(), it returns two important values:

A time series (y): A NumPy array containing the audio signal's amplitude values. By default, Librosa normalizes the data so that the values range from -1.0 to 1.0. This is the digital signal 

A sampling rate (sr): The number of samples per second of audio

To check sample rate for any given file: librosa.get_samplerate 
librosa.featureFeature extraction and manipulation. This includes low-level feature extraction, such as chromagrams, Mel spectrogram, MFCC, and various other spectral and rhythmic features. Also provided are feature manipulation methods, such as delta features and memory embedding.

#### -What padding and truncating means for audio and how to standardize clip length across a dataset? 
A PAD (Passive Attenuation Device) is an attenuator circuit designed to reduce the strength of an incoming audio signal before it reaches the next stage of the audio chain.
Instead of boosting or shaping your sound like an equalizer or preamp, a PAD simply lowers the volume of the signal, often by 20, 26, or 30 decibels. This helps prevent distortion, clipping, and overloading, especially when you are working with high-level audio sources.

Truncating is shortening/trimming audio. 

#### -How to normalize MFCC arrays and why it matters before feeding them into a model? 
MFCC’s assist in feature extraction, reducing dimensionality while retaining crucial information for efficient data processing. 


#### List of Sources

https://medium.com/swlh/music-genre-classification-part-1-4c48a1a246ca
https://geraldclarkaudio.medium.com/audio-programming-sample-rate-and-bit-depth-dd68ae58f717
https://www.retekess.com/blog/understanding-sample-rates-in-digital-audio?
https://vibrationresearch.com/blog/what-is-a-spectrogram/
srsltid=AfmBOoo-4N5QO-kfZrKlGfruCvBTA1qcRXZ9cTjUBdf7SSufyXtQ3L1V
https://docs.edgeimpulse.com/knowledge/concepts/data-engineering/audio-feature-extraction
https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/
https://www.geeksforgeeks.org/data-analysis/preprocessing-the-audio-dataset/
https://www.proacousticsusa.com/blog/post/what-is-a-pad-in-audio-passive-attenuation-devices/?srsltid=AfmBOorW0OABdeEzg20j9I77stKz4_0YnQAblpNeoUAUEDztt4wqMgG_
https://librosa.org/doc/latest/tutorial.html
https://www.youtube.com/watch?v=TlQcd_ogsIU








