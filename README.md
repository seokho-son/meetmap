# MeetMap Project

## Overview
MeetMap is a Python-based Flask web application that extracts room numbers and their coordinates from images. It utilizes image processing and Optical Character Recognition (OCR) to identify text within images and store the location information of the detected text. This application is particularly useful for mapping room numbers in building layouts or similar scenarios.

## Features
- Extracts room numbers and their coordinates from images.
- Manages the extracted room number data through REST API endpoints.
- Adjusts the image's height to a consistent resolution for processing.
- Highlights extracted room numbers on the image.

## Installation
### Prerequisites
- Python 3.x
- Flask, NumPy, Pillow, Pytesseract, and OpenCV libraries. 

Install them using the following command:
```bash
pip install Flask numpy Pillow pytesseract opencv-python-headless
```

### Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/meetmap.git
cd meetmap
```

## Usage
### Running the Application

Run the application with the following command:

```bash
python meetmap.py <path_to_your_image.jpg>
```
Replace <path_to_your_image.jpg> with the path to the image you want to process.


### API Endpoints
- GET /room: Retrieves a list of all extracted room numbers.
- GET /room/<room_number>: Retrieves coordinate information for a specific room number.
- POST /room: Adds new room number and coordinate information.
- PUT /room/<room_number>: Updates coordinate information for an existing room number.
- GET /api: Lists all available API endpoints.
- GET /image: Returns the processed image.

### Example API Calls
To retrieve a list of room numbers:
```bash
curl http://localhost:5000/rooms
```

To retrieve coordinate information for a specific room number:
```bash
curl http://localhost:5000/room/551
```

## License
This project is licensed under the Apache 2.0 License.