# 🚗 License Plate Detection Using OpenCV and CNN

## 📌 Project Description
This project focuses on detecting and extracting vehicle license plates from uploaded JPG/JPEG images using Python and OpenCV. It uses image processing techniques to identify potential license plate regions and segment individual characters for further recognition using deep learning techniques.

## 🎯 Objectives
- 📷 Upload car images in JPG/JPEG format.
- 🔍 Detect potential vehicle license plate regions.
- ✂️ Extract license plates from vehicle images.
- 🔠 Segment individual characters from license plates.
- 🤖 Explore CNN-based character classification.

## 🛠️ Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib
- TensorFlow / Keras
- Scikit-learn
- Google Colab

## ⚙️ Project Workflow
1. Upload a vehicle image.
2. Read the image using OpenCV.
3. Convert the image into grayscale.
4. Apply bilateral filtering and edge detection.
5. Detect contours to identify potential license plate regions.
6. Extract the detected license plate.
7. Segment individual characters.
8. Prepare character images for CNN classification.

## 📂 Project Structure
```text
License-Plate-Detection-Using-CNN/
│
├── license_plate_detection.ipynb
├── README.md
└── sample_images/
    └── car.jpg
```

## 📦 Installation

Install the required libraries:

```bash
pip install numpy opencv-python matplotlib tensorflow scikit-learn
```

## ▶️ How to Run
1. Open the notebook in Google Colab.
2. Run all the code cells.
3. Upload a JPG/JPEG vehicle image.
4. View the detected license plate.
5. Extract and visualize individual characters.

## 📊 Expected Output
- Original uploaded vehicle image.
- Detected license plate with a bounding rectangle.
- Cropped license plate image.
- Segmented character images.

## 🚀 Future Improvements
- Implement a trained CNN model for character recognition.
- Integrate YOLO for improved license plate detection.
- Recognize complete vehicle registration numbers using OCR.
- Support real-time license plate detection using video.
- Improve accuracy under different lighting and weather conditions.

## 👨‍💻 Author

**Nawaz Shareef Shaik**

GitHub: https://github.com/ShaikNawazShareef

## ⭐ Conclusion
This project demonstrates vehicle license plate detection and character segmentation using Python and OpenCV, providing a foundation for developing an intelligent automatic number plate recognition (ANPR) system.
