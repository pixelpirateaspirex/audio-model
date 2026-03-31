#  Voice Language Identifier

## Project Overview
The **Voice Identifier** is ML designed to identify spoken audio as either **English** or **Hindi**. 

Rather than relying on conventional audio-based techniques, this approach treats the problem from a computer vision perspective. Audio samples are first transformed into Mel-Spectrograms, which are visual representations of sound frequencies over time. These images are then processed using a Convolutional Neural Network (CNN), which learns to recognize distinct visual patterns corresponding to each language and produces accurate classifications.

## Tech Stack Used
- **Frontend / Deployment**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: TensorFlow & Keras (CNN Architecture)
- **Audio Processing**: Librosa
- **Data Visualization & Image Processing**: Matplotlib, Pillow (PIL), NumPy
- **Environment**: Python 3.11

## Links
- **Deployment Link**: [Streamlit](https://audio-model.streamlit.app)
- **Colab Notebook Link**: [Colab](  https://colab.research.google.com/drive/1Cr3JELKzHyKMQjw79tYdF-qFJ4qdek4a?usp=sharing)
- **GitHub Repository**: [GitHub](https://github.com/pixelpirateaspirex/audio-model/main/README.md)
- **Dataset Used**: [Kaggle](https://www.kaggle.com/datasets/abhay242/english-hindi-audio-spectrograms)
