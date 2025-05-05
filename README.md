# 🖼️ Image Captioning with Visual Attention

This project generates textual descriptions for images using deep learning. It leverages convolutional neural networks (CNNs) and attention-based sequence models to generate accurate and meaningful captions for images. The architecture follows the "Show, Attend and Tell" approach.

---

## 🚀 Overview

- Uses **Flickr8k** dataset: 8,000 images with 5 captions each.
- Applies **Visual Attention** to focus on relevant image regions while generating captions.
- Inspired by the paper [Show, Attend and Tell (Xu et al., 2015)](https://arxiv.org/pdf/1502.03044.pdf).
- Converts predicted captions to speech using `gTTS`.

---

## 🧠 Problem Statement

Build a model that can:
- Understand visual features of an image.
- Generate natural language descriptions of the image.
- Use attention to focus on key parts of the image while generating each word.

---

## 📦 Dependencies

- Python 3.x  
- TensorFlow 2.x  
- gTTS (Google Text-to-Speech)

Install dependencies:

```bash
pip install tensorflow
pip install gTTS
```

---

## 📊 Dataset

- **Flickr8k_Dataset**: Contains 8,000 labeled images.
- **Flickr8k_text**: Each image has 5 human-annotated captions.
- The dataset can be downloaded from:  
  [Flickr8k Dataset Registration](https://forms.illinois.edu/sec/1713398)

---

## ⚙️ Run Instructions

1. Clone the repo and download the dataset:

```bash
git clone https://github.com/<your-repo>/image-captioning
cd image-captioning
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Preprocess data and extract features.

4. Train the model:

```bash
python train_caption_model.py
```

5. Generate captions for new images:

```bash
python generate_caption.py --image_path sample.jpg
```

6. Convert caption to speech (optional):

```bash
python speak_caption.py --caption "A man is riding a horse in a field."
```

---

## 📈 Evaluation

- Model is evaluated using **BLEU Score**, a popular metric in NLP.
- Score ranges from **0 (poor)** to **1 (perfect match)**.
- Higher BLEU = Better quality captions.

---

## 🧠 Model Architecture

Based on the **"Show, Attend and Tell"** paper:

- **Encoder**: Pre-trained CNN (e.g., InceptionV3 or VGG) to extract image features.
- **Attention Layer**: Learns to focus on different image regions per word.
- **Decoder**: RNN (LSTM) that generates the caption word-by-word.

---

## 📚 References

- TensorFlow Tutorial: https://www.tensorflow.org/tutorials/text/image_captioning  
- Show, Attend and Tell Paper: https://arxiv.org/pdf/1502.03044.pdf  
- BLEU Score: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/  
- CS231n Lectures by Andrej Karpathy: https://www.youtube.com/watch?v=NfnWJUyUJYU  
- Seq2Seq Models by Andrew Ng: https://www.youtube.com/watch?v=Q8ys8YnDRXM  
- Attention is All You Need: https://arxiv.org/pdf/1706.03762v5.pdf  
- Blog Tutorial: https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html

---

## 📝 License

This project is licensed under the MIT License.
