# Person Tracking Project

## Introduction
This project focuses on tracking people in video frames using the YOLOv8 object detection model. The code provides three main components: exploratory data analysis (EDA) on the COCO 2017 dataset, training the YOLOv8 model on the filtered dataset, and then using the trained model to track people in a video.

## Features
- Performed EDA on the COCO 2017 dataset to filter for the 'person' class.
- Trained YOLOv8 models (yolov8n, yolov8s, yolov8m) on the filtered COCO 2017 dataset.
- Implemented video tracking using the trained YOLOv8 models to detect and track people in a video.
- Saved the tracked video with bounding boxes and unique IDs for each person.
- Reported the total number of people detected in the video.

## Prerequisites
- Python 3.x
- Ultralytics YOLO library
- FiftyOne library
- OpenCV library
- Matplotlib library

## Installation
1. Clone the repository:
```
git clone https://github.com/your-username/person-tracking.git
```
2. Navigate to the project directory:
```
cd person-tracking
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```

## Exploratory Data Analysis
Before training the YOLOv8 models, we performed an exploratory data analysis (EDA) on the COCO 2017 dataset to prepare the data for training.

1. Loaded the COCO 2017 dataset using the FiftyOne library, focusing on the 'person' class.
2. Filtered the dataset to only include samples with 'person' detections in the ground truth.
3. Exported the filtered dataset in the YOLOv5 format, with the 'person' class as the only label, to the `./yolov5-coco-datasets` directory.

This process ensured that the training, validation, and test splits of the dataset only contained samples with 'person' detections, which was the focus of our person-tracking project.

### Sample Data

![image](https://github.com/insomnius/person-detection/assets/20650401/366d8415-0cf0-4e2c-bfbb-05c611e4ec5a)

![image](https://github.com/insomnius/person-detection/assets/20650401/fa1a7b04-3cb1-40c1-86eb-2d0c1e3fc27d)

![image](https://github.com/insomnius/person-detection/assets/20650401/d62f425f-ec3d-424d-ae32-09b1d254cddc)

## Usage
1. Run the EDA code:
```
python eda.py
```
This will load the COCO 2017 dataset, filter it for the 'person' class, and export the dataset in the YOLOv5 format.

2. Run the training code:
```
python train.py
```
This will train the YOLOv8 models on the filtered COCO 2017 dataset and save the trained weights in the `./training/` directory.

3. Run the tracking code:
```
python track.py
```
This will use the trained YOLOv8 model to detect and track people in the provided video, saving the output video in the current directory.

## Results

The trained YOLOv8 models achieved the following mean average precisions (mAP) on the COCO 2017 validation set:
- yolov8n: `0.61287`
- yolov8s: `0.56026`
- yolov8m: `0.59617`

### Detection

![image](https://github.com/insomnius/person-detection/assets/20650401/42914af4-b2c8-4de9-867d-fbf7b8b438d5)

### Tracking

The tracking code was able to detect and track number of people in the provided video.

<details open="" class="details-reset border rounded-2">
  <video src="https://github.com/insomnius/person-detection/assets/20650401/00f50d3a-13a8-4fdb-a4e1-f6cd5194224f" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px">
  </video>
</details>

<br>

<details open="" class="details-reset border rounded-2">
  <video src="https://github.com/insomnius/person-detection/assets/20650401/7f7ad14d-3566-4503-949a-784cf1b7ef49" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px">
  </video>
</details>

## Future Improvements
- Integrate more advanced tracking algorithms to improve the accuracy and robustness of the person tracking.
- Explore the use of other object detection models, such as YOLOv5 or Faster R-CNN, and compare their performance.
- Implement real-time person tracking on live video streams.
- Explore the integration of the person tracking system with other applications, such as people counting or activity recognition.

## License
This project is licensed under the [MIT License](LICENSE).