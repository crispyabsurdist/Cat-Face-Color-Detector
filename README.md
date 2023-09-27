# CatFace Palette (WIP)

A Python application for detecting cat faces and classifying their colors using OpenCV and a pre-trained deep learning model. This is a Work In Progress!
I still need to fine adjust the dataset and also create a larger dataset so that the classification model will get better.

## Overview

This project consists of three Python scripts for cat-related tasks:

- **maincoon.py:** Detects cat faces in a live video stream or a video file and classifies their colors using a pre-trained deep learning model.
- **cat_trainer.py:** Trains a cat color classification model using a custom dataset of black and white cat images.
- **cat_scraper.py:** Scrapes cat images from Pixabay based on a query and downloads them for building the dataset.

## Installation

### Dependencies
To run these scripts, you need Python 3.x installed on your system. Additionally, you'll need the following libraries:

```bash
pip3 install opencv-python-headless tensorflow numpy python-dotenv tqdm requests
``````

### .env file
Set up your Pixabay API key and other environment variables by creating a .env file in the project directory with the following content:
```env
PIXABAY_API_KEY=your_pixabay_api_key
URL_ENDPOINT=https://pixabay.com/api/
TRAIN_PATH=/path/to/training_data
VALID_PATH=/path/to/validation_data
```

## Usage

### Dataset

The dataset used for training the cat color classification model is available at https://www.kaggle.com/andrewmvd/cat-dataset. The dataset contains 25,000 images of cats with 12,500 images of each cat type (black and white). The dataset is split into training and validation sets with 20,000 and 5,000 images, respectively.

### maincoon.py

* Detects cat faces in a live video stream from your default webcam or a specified video file.
* Classifies the color of each detected cat face as "Black," "White," or "Unknown."
* Press the 'q' key to exit the script.

```bash
python3 maincoon.py
```

### cat_trainer.py

* Trains a cat color classification model using a custom dataset of black and white cat images.
* The trained model is saved as "cat_color_model.h5."

```bash
python3 cat_trainer.py
```

### cat_scraper.py

* You need a Pixabay API key to run this script. You can get one for free by signing up at https://pixabay.com/api/docs/.
* Scrapes cat images from Pixabay based on a query and downloads them for building the dataset.
* You need to change the query variable in the script to specify your desired cat type (e.g., "black," "white," etc.).

```bash
python3 cat_scraper.py
```
