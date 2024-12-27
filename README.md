# MeetMap Project (DIR service)

## Overview
MeetMap is a Python-based Flask web application that extracts room numbers and their coordinates from images. It utilizes image processing and Optical Character Recognition (OCR) to identify text within images and store the location information of the detected text. This application is particularly useful for mapping room numbers in building layouts or similar scenarios.

## Features
- Automated extraction of room numbers using OCR.
- Image processing for enhanced text detection.
- Flask web server with RESTful API for data management.
- Room number similarity analysis for approximate matches.
- Interactive image visualization with highlighted room numbers.
- Support for handling multiple floor images and complex room number formats.

## Installation

### Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/meetmap.git
cd meetmap
```

### Prerequisites
- Python 3.x
- Flask numpy Pillow opencv-python-headless easyocr torch torchvision

Install them using the following command:
```bash
# pip install Flask numpy Pillow pytesseract opencv-python-headless
# pip or pip3
pip install -r requirements.txt
```

## Usage

### Running the Application


Create the `./image/map/` directory and add image files to this directory.
An example of an image file name is `7-5.png`, which represents the fifth floor of building 7.

(Optional) Create the `./image/room/` directory and add pictures of rooms to this directory.
An example of a room image file name is `7-511.png`, which represents room 511 in building 7.

On Ubuntu:
```bash
mkdir -p ./image/map ./image/room
```
On Windows:
```cmd
mkdir image\map image\room
```

Update `alias.json` to provide your custom aliases for the rooms and buildings.


Run the application with the following command:

```bash
python meetmap.py
```

Script for Ubuntu users. (will automate installation)
```bash
./init.sh
```

If a map.json file already exists, the program will ask whether to perform a new analysis or use the existing data. The analyzed data is saved in a temporary directory (`tmp/`) and as a JSON file (`map.json`).

### API Endpoints
1. GET /: Lists all available API endpoints.
1. GET /room: Retrieves a list of all extracted room numbers.
1. GET /room/<room_number>: Highlights and returns an image with the specified room number.
   - force=true: Forces the creation and combination of new images, ignoring any cached combined image.
   - Without force parameter or with force=false: Returns a previously generated combined image if it exists.
1. GET /validate: Combines multiple images into one and returns the combined image. 
   - testset=true: Generates and combines images for room numbers read from testset.txt.
   - force=true: Forces the creation and combination of new images, ignoring any cached combined image.
   - Without force parameter or with force=false: Returns a previously generated combined image if it exists.

### Example API Calls
To retrieve a list of room numbers:
```bash
curl http://localhost:1111/room
```

To view an image with a specific room number highlighted:
```bash
curl http://localhost:1111/room/1-222
```


# Architecture
- Admin interections

```mermaid
graph TD
    %% Admin interactions
    A1[Admin] --> INIT[Start server and initialize]
    INIT --> CHECK_JSON[Check if map.json exists]
    CHECK_JSON -->|Yes| ANALYSIS_PROMPT[Perform new analysis?]
    ANALYSIS_PROMPT -->|Yes| NEW_ANALYSIS[Perform image analysis]
    NEW_ANALYSIS --> SAVE_JSON[Save results to map.json]
    SAVE_JSON --> JSON_FILE[Write to map.json]
    ANALYSIS_PROMPT -->|No| LOAD_JSON[Load existing map.json]
    LOAD_JSON --> JSON_FILE[Read map.json]

    %% Image analysis workflow
    NEW_ANALYSIS --> ITERATE[Iterate through image directory]
    ITERATE --> PROCESS_IMAGE[Analyze each image]
    PROCESS_IMAGE --> PREPROCESS[Preprocess image]
    PREPROCESS --> TMP_DIR[Save preprocessed images to tmp directory]
    PROCESS_IMAGE --> OCR[Run OCR using EasyOCR]
    OCR --> EXTRACT[Extract room numbers and coordinates]
    EXTRACT --> UPDATE_RESULTS[Update analyzed_results]
    PROCESS_IMAGE --> FONT[Use DejaVuSans.ttf for annotations]

    %% File relationships
    JSON_FILE[map.json]
    TMP_DIR["Temporary directory (tmp)"]
    FONT["DejaVuSans.ttf font file"]
    IMAGE_DIR["Image directory"]
    JSON_FILE --> LOAD_JSON
    TMP_DIR --> HIGHLIGHT
    TMP_DIR --> RETURN_MAP
    IMAGE_DIR --> ITERATE
```

- User interections

```mermaid
graph TD
    %% User interactions
    U1[User] --> API_CALL[Flask API Call]

    %% API Call Workflow
    API_CALL --> GET_ROOT[GET Root Endpoint]
    GET_ROOT --> LIST_APIS{{List all API endpoints}}

    API_CALL --> GET_ROOMS[GET Room List]
    GET_ROOMS --> LOAD_JSON[(Load map.json)]
    LOAD_JSON --> LIST_ROOMS{{List all available room numbers}}

    API_CALL --> GET_ROOM[GET Specific Room]
    GET_ROOM --> ROOM_CHECK{Check if room exists in map.json}
    ROOM_CHECK -->|Yes| HIGHLIGHT((Generate highlighted map))
    HIGHLIGHT --> TMP_DIR[/"Save to tmp directory"/]
    TMP_DIR --> RETURN_MAP[[Return highlighted map to user]]
    ROOM_CHECK -->|No| ERROR{{Return error message}}

    API_CALL --> VALIDATE[GET Validate Images]
    VALIDATE --> CHECK_TESTSET{Testset provided?}
    CHECK_TESTSET -->|Yes| COMBINE_TESTSET((Combine testset images))
    COMBINE_TESTSET --> TMP_DIR
    TMP_DIR --> RETURN_TESTSET[[Return testset image]]
    CHECK_TESTSET -->|No| COMBINE_ALL((Combine all highlighted images))
    COMBINE_ALL --> TMP_DIR
    TMP_DIR --> RETURN_COMBINED[[Return combined image]]
```




## License
This project is licensed under the Apache 2.0 License.
