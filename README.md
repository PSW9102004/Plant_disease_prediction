# 🌿 Plant Disease Detection using CBAM-ResNet34

This project is a deep learning-based classifier designed to detect plant diseases from leaf images using a modified ResNet-34 architecture enhanced with CBAM (Convolutional Block Attention Module). It combines two popular datasets — **PlantDoc** and **PlantVillage** — to build a robust image classification model.

---

## 📦 Datasets Used

* **PlantDoc**: Real-world dataset with \~2.6k noisy, variable-quality leaf images across multiple crops.
* **PlantVillage**: Benchmark dataset with over 54k high-quality, labeled plant disease images.

> We combined both datasets and split them into training (80%), validation (10%), and test (10%) sets with class balance preserved.(**Not Combined here as during the training the Google colab's Compute units were not suffcient for such huge traning**)

---

## 🧠 Model Architecture

* **Backbone**: ResNet-34 pretrained on ImageNet
* **Attention**: CBAM (Channel and Spatial Attention)
* **Classifier**: AdaptiveAvgPool2D → Dropout → Fully Connected → Softmax
* **Total Number of Trainable Params: 21,385,955 (21M)**
* **Estimated Total Size (MB): 181.46MB**

---

## 🔥 Performance Overview

| Metric         | Score (Before Combining Datasets) |
| -------------- | --------------------------------- |
| Train Accuracy | > 90%                             |
| Val Accuracy   | \~67% (overfitting observed)      |

### ⚠️ Overfitting Analysis

Initially, the model trained only on PlantDoc showed signs of **overfitting**:

* Training accuracy soared to 90%+
* Validation accuracy stagnated at \~67%

### 🧪 Improvements To be Made(if more compute power available)

* ✅ Added **PlantVillage** dataset to increase data volume
* ✅ Introduced **extensive data augmentation**
* ✅ Applied **Dropout** and **L2 regularization**
* ✅ Implemented **stratified splitting** to maintain class balance
* ✅ Enabled **early stopping** and **checkpointing**

---

## 📊 Final Project Structure

```
project-root/
├── data/
│   ├── raw/
│   │   ├── PlantDoc/
│   │   │    ├── PlantDoc-Dataset/
│   │   │    │           ├── train
│   │   │    │           ├── split train
│   │   │    │           ├── test
│   │   │    │           ├── val
├── models/
│   ├── base_cnn.py
│   ├── model summary.txt
│   ├── efficientnet_b3                         
├── inference/
│   ├── predict.py
│   └── dashboard.py
├── data/
│   ├── transforms.py
│   └── download_datasets.py
├── cams/
│   ├── images
├── train.py
├── eval.py
├── gradcam.py
└── requirements.txt
```

---

## 🖼️ Grad-CAM Visualization

Use `gradcam.py` to visualize which parts of the image the model focuses on when making predictions.

```bash
python gradcam.py --image-path sample.jpg --checkpoint checkpoints/best_model.pth
```

---

## 🧪 Inference

Run batch predictions with human-readable class labels:

```bash
python inference/predict.py --dir test_images/ --checkpoint checkpoints/best_model.pth
```

---

## 💡 Future Improvements

* 🔍 Try **lighter backbones** like MobileNetV3 for mobile deployment
* 🧠 Fine-tune with **class-weighted loss** to handle class imbalance
* 🧾 Add **metadata-based features** (location, humidity) for richer context
* ⚙️ Integrate with a **real-time webcam/detection dashboard**

---

## 📌 Credits

* PlantDoc by M. H. M. H. B. et al. (CVPRW 2020)
* PlantVillage by Hughes and Salathé (arXiv 2015)

---

> Made with 💚 for sustainable agriculture and smarter farming
