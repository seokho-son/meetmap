# MeetMap Project

## Overview
MeetMap is a Python-based Flask web application that extracts room numbers and their coordinates from images. It utilizes image processing and Optical Character Recognition (OCR) to identify text within images and store the location information of the detected text. This application is particularly useful for mapping room numbers in building layouts or similar scenarios.

## Features
- Automated extraction of room numbers using OCR.
- Image processing for enhanced text detection.
- Flask web server with RESTful API for data management.
- Room number similarity analysis for approximate matches.
- Interactive image visualization with highlighted room numbers.
- Support for handling multiple floor images and complex room number formats.

# Architecture
- UML sequence diagram: https://github.com/seokho-son/meetmap/blob/main/sequence.uml

## Installation

### Prerequisites
- Python 3.x
- Flask, NumPy, Pillow, Pytesseract, and OpenCV libraries.

Install them using the following command:
```bash
# pip install Flask numpy Pillow pytesseract opencv-python-headless
# pip or pip3
pip install -r requirements.txt
```
For Windows users, additional steps are needed to use Pytesseract:
1. Install Tesseract OCR: Download the Tesseract executable from [this link](https://github.com/UB-Mannheim/tesseract/wiki) and install it. 


### Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/meetmap.git
cd meetmap
```
Add image files to ./image/ directory.
Example of the file name of an image file is `5.jpg` which means the fifth floor.

## Usage

### Running the Application

Run the application with the following command:

```bash
python meetmap.py <path_to_directory_containing_images>
```
Replace <path_to_directory_containing_images> with the path to the directory containing the images you want to process.

If a map.json file already exists, the program will ask whether to perform a new analysis or use the existing data. The analyzed data is saved in a temporary directory (tmp) and as a JSON file (map.json).

### API Endpoints
1. GET /api: Lists all available API endpoints.
1. GET /room: Retrieves a list of all extracted room numbers.
1. GET /view/<room_number>: Highlights and returns an image with the specified room number.
   - force=true: Forces the creation and combination of new images, ignoring any cached combined image.
   - Without force parameter or with force=false: Returns a previously generated combined image if it exists.
1. GET /room/<room_number>: Retrieves coordinate information for a specific room number.
1. POST /room: Adds new room number and coordinate information.
1. PUT /room/<room_number>: Updates coordinate information for an existing room number.
1. GET /validate: Combines multiple images into one and returns the combined image. 
   - testset=true: Generates and combines images for room numbers read from testset.txt.
   - force=true: Forces the creation and combination of new images, ignoring any cached combined image.
   - Without force parameter or with force=false: Returns a previously generated combined image if it exists.

### Example API Calls
To retrieve a list of room numbers:
```bash
curl http://localhost:1111/room
```

To retrieve coordinate information for a specific room number:
```bash
curl http://localhost:1111/room/551
```

To view an image with a specific room number highlighted:
```bash
curl http://localhost:1111/view/551
```

## License
This project is licensed under the Apache 2.0 License.