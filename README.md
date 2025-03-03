Here's a **README.md** file for your image classification project:  

# 🖼️ AI-Powered Image Sorter

An AI-based image classification tool that automatically sorts images into "wanted" (important images) and "unwanted" (spam, screenshots, math problems, etc.). Uses a **deep learning model (MobileNetV2)** to classify images and move them to the correct folder. Includes a **GUI** for manual corrections.

## 🚀 Features
✅ **AI-based sorting** – Automatically classifies images using TensorFlow  
✅ **GUI interface** – View and manually override AI predictions  
✅ **Batch processing** – Processes all images in `dataset/new/`  
✅ **Easy training & customization** – Train on your own dataset  
✅ **Lightweight** – Uses MobileNetV2 for fast classification  

---

## 🏗️ Project Structure

```
image-sorter/
│── dataset/
│   ├── unclassified/    # Images to be labeled
│   ├── new/             # New images to classify
│   ├── wanted/          # AI-classified important images
│   ├── unwanted/        # AI-classified spam images
│   ├── train/           # Training dataset
│   ├── test/            # Test dataset
│── model/
│   ├── image_classifier.h5  # Trained AI model
│── scripts/
│   ├── image_labeler.py      # Manual labeling script
│   ├── train_model.py        # Model training script
│   ├── image_sorter_gui.py   # GUI-based sorting
│── .gitignore
│── README.md
│── requirements.txt
```

---

## ⚡ Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/image-sorter.git
cd image-sorter
```

2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Prepare your dataset**
- Place unclassified images in `dataset/unclassified/`
- Run `image_labeler.py` to manually sort images into `dataset/wanted/` and `dataset/unwanted/`
- Split data into `train/` and `test/`

4️⃣ **Train the model**
```bash
python scripts/train_model.py
```

5️⃣ **Run the GUI sorter**
```bash
python scripts/image_sorter_gui.py
```

---

## 🧠 How It Works

1️⃣ **Loads images from `dataset/new/`**  
2️⃣ **Uses AI to classify as "wanted" or "unwanted"**  
3️⃣ **Moves images to the correct folder**  
4️⃣ **Displays a GUI for manual correction**  

🔹 If AI **misclassifies an image**, you can override the decision.  

---

## 📌 Model Details

- **Architecture:** MobileNetV2 (pretrained on ImageNet)  
- **Input Size:** 128x128  
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Dataset Augmentation:** Rotation, flipping, brightness adjustment  

---

## 🔧 Customization

🔹 **Train on your own dataset** – Replace images in `dataset/train/` and `dataset/test/`  
🔹 **Change model architecture** – Modify `train_model.py`  
🔹 **Adjust image size** – Change `target_size=(128, 128)` in scripts  

---

## 📜 License
MIT License

---

## 🤝 Contributing
Pull requests are welcome! Open an issue for feature requests or bugs.

---

## ⭐ Credits
Developed with ❤️ using **TensorFlow, OpenCV, and Tkinter**.
