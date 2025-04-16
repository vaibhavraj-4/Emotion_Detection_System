#  Emotion Detection System
 This repository implements a **Multimodal Emotion Detection System** capable of analyzing both
 speech and text inputs to detect human emotions using deep learning and NLP techniques.
 ##  Features- **Speech-based Emotion Recognition** using a trained CNN model.- **Text-based Emotion Classification** using Hugging Face Transformers.- **Speech-to-Text Conversion** and language translation for non-English input.- **Combined Emotion Fusion** using weighted average of both modalities.- Intuitive **Streamlit-based UI**.- Supports **microphone input** or **manual text/audio upload**.- Uses **VADER**, **TextBlob**, and **transformers pipeline** for sentiment/emotion detection.- Displays emojis representing emotions.
 ##  Models Used- `testing10_model.h5`: Trained CNN model for speech emotion recognition.- `bhadresh-savani/distilbert-base-uncased-emotion`: Hugging Face model for text emotion
 classification.
 ##  Dataset Info- Based on RAVDESS dataset naming conventions.- Custom-processed folder `Actor_27` for unified audio processing.
##  Technologies & Libraries- `Streamlit`, `Librosa`, `Tensorflow`, `SpeechRecognition`- `googletrans`, `transformers`, `TextBlob`, `NLTK`, `Soundfile`- `streamlit_mic_recorder`
 ##  How It Works
 1. **Speech Input**: Record/upload `.wav` file  Extract MFCC features  Predict emotion.
 2. **Text Input**: Convert speech to text  Translate (if needed)  Classify using transformer model.
 3. **Fusion**: Combine predictions from both sources using a scoring mechanism.
 4. **Output**: Final emotion + emoji + confidence score.
 ##  Emotion Categories- Speech model: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`- Text model (mapped): `joy`, `sadness`, `anger`, `fear`, `surprise`, `love`  mapped to above
 ##  Project Structure
 ```
 app.py                # Streamlit main app
 test.py               # Backend logic and ML model handling
 testing10_model.h5    # CNN speech emotion model
 requirements.txt      # Dependencies
 Actor_27/             # Processed audio dataset
```
 ##  Installation
 ```bash
 git clone https://github.com/vaibhavraj-4/Emotion_Detection_System.git
 cd Emotion_Detection_System
 pip install -r requirements.txt
 streamlit run test.py
 ```
 ##  Use Cases- Customer service feedback analysis- Real-time emotion detection for virtual assistants- E-learning mood tracking
 ##  Note
 Ensure `testing10_model.h5` is present in the root directory.
 Microphone access is required for live speech input.
 ##  Author
 Developed by Vaibhav Raj  
GitHub: [vaibhavraj-4](https://github.com/vaibhavraj-4)
