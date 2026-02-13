# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright 2024 Seokho Son <https://github.com/seokho-son/meetmap>
# SPDX-License-Identifier: Apache-2.0

# Importing necessary libraries
import sys
import time
import numpy as np
import re
import os
import json
import threading
import atexit
import cv2  # OpenCV library for computer vision tasks
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from flask import Flask, send_from_directory, jsonify, request, send_file, Response
import logging

# Initializing Flask application
app = Flask(__name__)

# Suppress werkzeug default request logging and use custom selective logging
_SUPPRESS_PATHS = ('/assets/', '/static/', '/favicon')
logging.getLogger('werkzeug').setLevel(logging.ERROR)

@app.after_request
def _log_request(response):
    """Selectively log requests, skipping noisy paths like favicon/static."""
    path = request.path
    if path == '/' or any(path.startswith(p) for p in _SUPPRESS_PATHS):
        return response
    print(f'{request.remote_addr} - "{request.method} {request.full_path.rstrip("?")}" {response.status_code}')
    return response

# --- Access counter (timestamp-based, last 100 days) ---
ACCESS_COUNT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access-count.json')
ACCESS_SAVE_INTERVAL = 10  # save to file every N increments
ACCESS_WINDOW_DAYS = 100   # rolling window for Top 10 ranking

_access_log = {}   # {room_id: [unix_timestamp, ...]}
_access_lock = threading.Lock()
_access_dirty = 0  # increments since last save


def _prune_old_entries(log, cutoff):
    """Remove timestamps older than cutoff; drop empty entries."""
    return {rid: [t for t in ts if t > cutoff]
            for rid, ts in log.items()
            if any(t > cutoff for t in ts)}


def _load_access_counter():
    """Load persisted access log from JSON file."""
    global _access_log
    if os.path.exists(ACCESS_COUNT_FILE):
        try:
            with open(ACCESS_COUNT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data and isinstance(next(iter(data.values()), None), list):
                cutoff = time.time() - ACCESS_WINDOW_DAYS * 86400
                _access_log = _prune_old_entries(data, cutoff)
                total = sum(len(v) for v in _access_log.values())
                print(f"Loaded access log: {len(_access_log)} rooms, {total} recent accesses")
            else:
                # Old counter format (room_id: int) — no timestamps, start fresh
                print("Migrating from legacy counter format — starting fresh time-based tracking")
                _access_log = {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {ACCESS_COUNT_FILE}: {e}")
            _access_log = {}


def _save_access_counter():
    """Atomically persist access log to JSON, pruning old entries."""
    global _access_dirty
    with _access_lock:
        if _access_dirty == 0:
            return
        cutoff = time.time() - ACCESS_WINDOW_DAYS * 86400
        pruned = _prune_old_entries(_access_log, cutoff)
        _access_log.clear()
        _access_log.update(pruned)
        data = dict(_access_log)
        _access_dirty = 0
    tmp_path = ACCESS_COUNT_FILE + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, ACCESS_COUNT_FILE)  # atomic on most OS
    except IOError as e:
        print(f"Warning: Could not save access counts: {e}")


def increment_access_count(room_id):
    """Thread-safe: append current timestamp to room's access log."""
    global _access_dirty
    with _access_lock:
        if room_id not in _access_log:
            _access_log[room_id] = []
        _access_log[room_id].append(int(time.time()))
        _access_dirty += 1
        should_save = (_access_dirty >= ACCESS_SAVE_INTERVAL)
    if should_save:
        _save_access_counter()


def get_top_accessed(n=20):
    """Return top N rooms by access count within the last ACCESS_WINDOW_DAYS."""
    cutoff = time.time() - ACCESS_WINDOW_DAYS * 86400
    with _access_lock:
        counts = []
        for room_id, timestamps in _access_log.items():
            recent = sum(1 for t in timestamps if t > cutoff)
            if recent > 0:
                counts.append((room_id, recent))
    counts.sort(key=lambda x: x[1], reverse=True)
    return counts[:n]


# Load persisted log at startup
_load_access_counter()
# Save counts on normal shutdown
atexit.register(_save_access_counter)

# EasyOCR Reader - lazy loaded when OCR is actually needed
# Dependencies: Requires PyTorch
# Legal Notice: Enabling GPU mode (gpu=True) subjects this usage to NVIDIA EULA as it utilizes NVIDIA CUDA
#              CPU mode (gpu=False) operates without NVIDIA CUDA components
reader = None  # Will be initialized on first OCR use

def get_ocr_reader():
    """
    Lazily initializes and returns the EasyOCR reader.
    This allows the application to run without easyocr/pytorch if OCR is not needed.
    """
    global reader
    if reader is None:
        import easyocr  # EasyOCR library for text recognition
        reader = easyocr.Reader(['en'], gpu=True)  # GPU/CPU mode can be configured as needed
    return reader

# Get the directory where this script is located (for reliable path resolution)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Font configuration
# NanumSquareL.ttf Font License - https://help.naver.com/service/30016/contents/18088?osType=PC&lang=ko
FONT_FILENAME = "NanumSquareL.ttf"
font_path = os.path.join(SCRIPT_DIR, "assets", FONT_FILENAME)
# Copyright (c) 2010, NAVER Corporation (https://www.navercorp.com/) with Reserved Font Name Nanum, 
# Naver Nanum, NanumGothic, Naver NanumGothic, NanumMyeongjo, Naver NanumMyeongjo, NanumBrush, 
# Naver NanumBrush, NanumPen, Naver NanumPen, Naver NanumGothicEco, NanumGothicEco, 
# Naver NanumMyeongjoEco, NanumMyeongjoEco, Naver NanumGothicLight, NanumGothicLight, 
# NanumBarunGothic, Naver NanumBarunGothic, NanumSquareRound, NanumBarunPen, MaruBuri, NanumSquareNeo
# This Font Software is licensed under the SIL Open Font License, Version 1.1.
# This license is copied below, and is also available with a FAQ at: http://scripts.sil.org/OFL
# SIL OPEN FONT LICENSE
# Version 1.1 - 26 February 2007 

# Verify font file exists at startup
if not os.path.exists(font_path):
    print(f"ERROR: Font file not found: {font_path}")
    print(f"This font is required for Korean text display.")
    print(f"")
    print(f"To resolve this issue:")
    print(f"  1. Add a Korean-supporting font file to: {os.path.dirname(font_path)}")
    print(f"  2. Or update the 'FONT_FILENAME' variable in this script to use a different font file.")
    sys.exit(1)

def get_font(size):
    """
    Returns a font object for the specified size.
    Uses the font file specified by FONT_FILENAME in the assets folder.
    """
    try:
        return ImageFont.truetype(font_path, size)
    except Exception as e:
        print(f"ERROR: Failed to load font: {font_path}")
        print(f"Error details: {e}")
        raise RuntimeError(f"Cannot load required font file: {font_path}")

# Handle directory
tmp_directory = os.path.join(SCRIPT_DIR, "tmp")
if not os.path.exists(tmp_directory):
    os.makedirs(tmp_directory)

directory_path = os.path.join(SCRIPT_DIR, "image", "map")
if not os.path.exists(directory_path):
    print("image/map directory does not exist.")
    sys.exit(1)    

def load_json_file(file_path):
    """
    Loads a JSON file, handling different encodings (e.g., UTF-8, UTF-8 with BOM).
    :param file_path: Path to the JSON file.
    :return: Parsed JSON object or an empty dictionary if loading fails.
    """
    try:
        # First attempt: Read the file as UTF-8 (most common case)
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except UnicodeDecodeError:
        print(f"UTF-8 decoding failed for {file_path}. Retrying with UTF-8-sig...")
        try:
            # Second attempt: Handle UTF-8 with BOM
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                return json.load(file)
        except UnicodeDecodeError:
            print(f"UTF-8-sig decoding also failed for {file_path}. Retrying without encoding...")
            try:
                # Third attempt: Default system encoding (may work for cp949 or other encodings)
                with open(file_path, 'r') as file:
                    return json.load(file)
            except Exception as e:
                print(f"Error reading JSON file {file_path}: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decoding error in {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
    
    # Return an empty dictionary if all attempts fail
    return {}

# Load map-alias.json once at the start of the program
def load_alias_data():
    """
    Loads alias data from map-alias.json, handling different encodings.
    :return: Parsed alias data as a dictionary, or an empty dictionary if loading fails.
    """
    alias_file = 'map-alias.json'  # Define the file path
    if not os.path.exists(alias_file):
        print("map-alias.json file not found.")
        return {}
    
    alias_data = load_json_file(alias_file)  # Use the helper function
    if alias_data:
        return alias_data
    else:
        print("Failed to load map-alias.json or file is empty.")
        return {}

alias_data = load_alias_data()

# Load map-note.json once at the start of the program
def load_note_data():
    """
    Loads note data from map-note.json, handling different encodings.
    :return: Parsed note data as a dictionary, or an empty dictionary if loading fails.
    """
    note_file = 'map-note.json'  # Define the file path
    if not os.path.exists(note_file):
        print("map-note.json file not found.")
        return {}
    
    note_data = load_json_file(note_file)  # Use the helper function
    if note_data:
        return note_data
    else:
        print("Failed to load map-note.json or file is empty.")
        return {}

note_data = load_note_data()

def get_note_data(request_id, floor_id, building_id):
    """
    Retrieves the note data matching the given IDs.
    :param request_id: The request ID to search for.
    :param floor_id: The floor ID to search for.
    :param building_id: The building ID to search for.
    :return: The matching note data as a dictionary, or None if no match is found.
    """
    note_data = load_note_data()  # Load the latest note data

    if not note_data:
        return None

    # Check for exact match with request_id
    if request_id and request_id in note_data:
        return note_data[request_id]

    # Check for match with floor_id
    if floor_id and floor_id in note_data:
        return note_data[floor_id]

    # Check for match with building_id
    if building_id and building_id in note_data:
        return note_data[building_id]

    # No match found
    return None


# Function to analyze an image and extract room number information
def analyze_image(image_path, image_name):
    """
    Analyzes the image and extracts room number information.
    :param image_path: Path to the image to be analyzed.
    :param image_name: Name of the image file.
    :return: A dictionary with room numbers and their coordinates.
    """

    analyzed_results = {}
    image = Image.open(image_path).convert("RGB")
    
    # Enhancing image quality (increasing contrast)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)  # Increasing contrast
    image.save(os.path.join(tmp_directory, f'{image_name}-image_after_enhance.png'))

    # Resizing the image
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    new_height = 1400
    new_width = int(aspect_ratio * new_height)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_image.save(os.path.join(tmp_directory, f'{image_name}-map.png'))
    print(f"{image_name}")
    print(f"Original Size: {original_width}x{original_height}")
    print(f"Resized Size: {new_width}x{new_height}")

    ocr_image = cv2.imread(os.path.join(tmp_directory, f'{image_name}-map.png'), cv2.IMREAD_GRAYSCALE)

    # Step 1: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    ocr_image = clahe.apply(ocr_image)
    ocr_image = cv2.medianBlur(ocr_image, 1)

    # # Step 2: Apply bilateral filter for noise reduction while preserving edges
    # ocr_image = cv2.bilateralFilter(ocr_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Save or return preprocessed image
    preprocessed_path = os.path.join(tmp_directory, f'{image_name}-image_after_enhance.png')
    cv2.imwrite(preprocessed_path, ocr_image)

    # Function to run OCR on multiple page segmentation modes and combine results
    def run_ocr_on_all_modes(image):
        """
        Runs OCR on the image using EasyOCR and combines results to match the structure of combined_data.
        :param image: PIL Image object.
        :return: A dictionary with keys 'text', 'conf', 'left', 'top', 'width', 'height'.
        """
        # Initialize the combined_data dictionary
        combined_data = {'text': [], 'conf': [], 'left': [], 'top': [], 'width': [], 'height': []}
        seen = set()  # Track unique text elements to avoid duplicates

        # Convert the PIL image to a format compatible with EasyOCR
        image_np = np.array(image)  # Convert to numpy array (EasyOCR can use this directly)
        
        # Run EasyOCR (lazy load the reader)
        ocr_reader = get_ocr_reader()
        results = ocr_reader.readtext(image_np, allowlist ='0123456789LGB-', detail=1, paragraph=False)  # detail=1 includes bounding boxes and confidence

        for (bbox, text, confidence) in results:
            if confidence > 0.4:  # Only consider results with the confidence
                # Extract bounding box coordinates
                (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]
                identifier = (text.strip(), int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                if identifier not in seen:
                    seen.add(identifier)
                    combined_data['text'].append(text.strip())
                    combined_data['conf'].append(int(confidence * 100))  # Convert to percentage
                    combined_data['left'].append(int(x_min))
                    combined_data['top'].append(int(y_min))
                    combined_data['width'].append(int(x_max - x_min))
                    combined_data['height'].append(int(y_max - y_min))
        return combined_data
    
    # Run OCR using EasyOCR
    data = run_ocr_on_all_modes(ocr_image)

    draw = ImageDraw.Draw(resized_image)
    font_size = 12
    font = get_font(font_size)

    # sort by 'text'
    sorted_data = sorted(zip(data['text'], data['conf'], data['left'], data['top'], data['width'], data['height']), key=lambda x: x[0])
    data['text'], data['conf'], data['left'], data['top'], data['width'], data['height'] = zip(*sorted_data)

    floorName = image_name.split('-')[1]

    # Parsing the extracted text and coordinates
    # Drawing rectangles around detected text on the image
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        draw.text((data['left'][i], data['top'][i] + data['height'][i]), f"{text}/{data['conf'][i]}%", fill="gray", font=font)
        if int(data['conf'][i]) > 0.4:  # Using low confidence texts as well
            confColor = "red" if data['conf'][i] < 50 else "green"
            draw.text((data['left'][i], data['top'][i] + data['height'][i]), f"{text}/{data['conf'][i]}%", fill=confColor, font=font)

            if text.startswith(floorName):
                print(f"{image_name}: {text} (Confidence: {data['conf'][i]}%)")
                store_room_data(text, data, i, analyzed_results, image_name, resized_image.size)
            else:
                print(f"{image_name}: (Not stored) {text} (Confidence: {data['conf'][i]}%)")

    for text, coords in analyzed_results.items():
        # Convert ratios back to pixel values
        x = int(coords["x_ratio"] / 100 * resized_image.size[0])
        y = int(coords["y_ratio"] / 100 * resized_image.size[1])
        w = int(coords["w_ratio"] / 100 * resized_image.size[0])
        h = int(coords["h_ratio"] / 100 * resized_image.size[1])
        print(f"- {text}: Location: X: {x}, Y: {y}, Width: {w}, Height: {h}")

        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        draw_x = x - w // 2
        draw_y = y - h // 2
        
        draw.rectangle(((draw_x, draw_y), (draw_x + w, draw_y + h)), outline="red")

    # Saving the final image with highlighted text
    resized_image.save(os.path.join(tmp_directory, f'{image_name}-highlighted_image.png'))
    return analyzed_results


# Function to store data about a room
def store_room_data(room_id, data, index, analyzed_results, image_name, image_size):
    """
    Stores the room number data along with its coordinates.
    :param room_id: The identified room number from the image.
    :param data: The data dictionary obtained from pytesseract OCR.
    :param index: The current index in the OCR data list.
    :param analyzed_results: The dictionary where results are being stored.
    :param image_name: The name of the image being analyzed, used for labeling.
    :param image_size: The size of the image (width, height).
    """

    BuildingName = image_name.split('-')[0]
    room_id = BuildingName + "-" + room_id

    # Extract coordinates and dimensions of the detected text
    (x, y, width, height) = (data['left'][index], data['top'][index], data['width'][index], data['height'][index])
    # print(f"- {room_id}: Location: X: {x}, Y: {y}, Width: {width}, Height: {height}")

    # Convert coordinates and dimensions to ratios based on image size
    image_width, image_height = image_size
    width_ratio = width / image_width * 100
    height_ratio = height / image_height * 100

    x_ratio = (x / image_width * 100) + (width_ratio / 2)  # Center X coordinate
    y_ratio = (y / image_height * 100) + (height_ratio / 2)  # Center Y coordinate

    # Store the room information in the analyzed_results dictionary
    analyzed_results[room_id] = {
        "floor": image_name, "x_ratio": x_ratio, "y_ratio": y_ratio, "w_ratio": width_ratio, "h_ratio": height_ratio
    }

def split_floor_id(floor):
    """
    Splits the floor identifier into building_id and floor_only_id.
    :param floor: The floor identifier in the format 'building-floor'.
    :return: A tuple containing building_id and floor_only_id.
    """
    building_id, floor_only_id = floor.split("-", 1)
    return building_id, floor_only_id

def save_results_to_json(analyzed_results, json_file='map.json'):
    """
    Saves the analyzed room data to a JSON file.
    :param analyzed_results: The dictionary containing room data to be saved.
    :param json_file: The name of the JSON file to save data into.
    """
    try:
        with open(json_file, 'w', encoding='utf-8') as file:  # Use UTF-8 encoding
            json.dump(analyzed_results, file, indent=4, ensure_ascii=False)  # Ensure non-ASCII characters are preserved
        print(f"Data successfully saved to {json_file}")
    except Exception as e:
        print(f"Failed to save data to {json_file}: {e}")

def load_results_from_json(json_file='map.json', customization_file='map-customization.json'):
    """
    Loads analyzed room data from JSON files and merges them.
    :param json_file: The name of the main JSON file to load data from.
    :param customization_file: The name of the customization JSON file to load data from.
    :return: A dictionary containing the merged room data.
    """
    data = load_json_file(json_file)
    
    # Load customization JSON file and merge with main data
    try:
        if os.path.exists(customization_file):
            customization_data = load_json_file(customization_file)  # Use the helper function
            if customization_data:
                for key, value in customization_data.items():
                    if key in data:
                        data[key].update(value)
                        # print(f"Customization data for {key}{value} has been merged")
                        # print(f"Current data for {key}: {data[key]}")
                    else:
                        data[key] = value
                        # print(f"Customization data for {key}{value} has been added")
                        # print(f"Current data for {key}: {data[key]}")
                # print("Customization data has been merged successfully.")
                # print("Merged data:")
                # print(json.dumps(data, indent=4, ensure_ascii=False))
        else:
            print(f"File {customization_file} does not exist.")
    except Exception as e:
        print(f"Error loading {customization_file}: {e}")
    
    return data
    


# Drawing assets...

def draw_dashed_line(draw, start, end, interval=10, width=1, fill="red"):
    """
    Draws a dashed line between two points.
    :param draw: ImageDraw object.
    :param start: Starting point tuple (x, y).
    :param end: Ending point tuple (x, y).
    :param interval: Length of each dash.
    :param width: Line width.
    :param fill: Line color.
    """
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = (dx**2 + dy**2)**0.5
    dash_count = int(distance // interval)
    
    for i in range(dash_count):
        x_start = x1 + (i * dx) / dash_count
        y_start = y1 + (i * dy) / dash_count
        x_end = x1 + ((i + 1) * dx) / dash_count
        y_end = y1 + ((i + 1) * dy) / dash_count
        if i % 2 == 0:
            draw.line([(x_start, y_start), (x_end, y_end)], fill=fill, width=width)

def draw_arrow_head(draw, start, end, arrow_size=10, fill="red"):
    """
    Draws an arrow head at the end of a line.
    :param draw: ImageDraw object.
    :param start: Starting point tuple (x, y) of the line.
    :param end: Ending point tuple (x, y) where the arrow head will be drawn.
    :param arrow_size: Size of the arrow head.
    :param fill: Arrow color.
    """
    # Calculate the angle of the line
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = np.arctan2(dy, dx)

    # Calculate the points of the arrow head
    arrow_point1 = (end[0] - arrow_size * np.cos(angle - np.pi / 6),
                    end[1] - arrow_size * np.sin(angle - np.pi / 6))
    arrow_point2 = (end[0] - arrow_size * np.cos(angle + np.pi / 6),
                    end[1] - arrow_size * np.sin(angle + np.pi / 6))

    # Draw the arrow head
    draw.polygon([end, arrow_point1, arrow_point2], fill=fill)

def calculate_arrow_path(center_x, center_y, box_x, box_y, box_w, box_h):
    """
    Calculates the path for an arrow from the center of the image to the edge of a box.
    :param center_x, center_y: Center coordinates of the image.
    :param box_x, box_y, box_w, box_h: Coordinates and size of the box.
    :return: A list of points representing the path of the arrow.
    """
    path = [(center_x, center_y)] # arrow start

    box_center_x = box_x + box_w / 2
    box_center_y = box_y + box_h / 2

    # logic to set path of the arrow
    if box_x <= center_x <= box_x + box_w: 
        horizontal_target_x = box_center_x
        path.append((horizontal_target_x, center_y))

        if center_y > box_center_y:  
            vertical_target_y = box_y + box_h  
        else:  
            vertical_target_y = box_y 
        path.append((horizontal_target_x, vertical_target_y))
    else:  
        vertical_target_y = box_center_y
        path.append((center_x, vertical_target_y))

        if box_center_x > center_x:
            horizontal_target_x = box_x  
        else: 
            horizontal_target_x = box_x + box_w  

        path.append((horizontal_target_x, vertical_target_y))

    return path

def draw_label(draw, message, position, font_size=30, fill="yellow"):
    """
    Draws a question mark at the given position.
    :param draw: ImageDraw object.
    :param position: Tuple (x, y) for the position of the question mark.
    :param size: Size of the question mark.
    :param fill: Color of the question mark.
    """

    font = get_font(font_size)

    # Calculate the size of the text
    text_bbox = draw.textbbox((0, 0), message, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the rectangle coordinates based on the position and text size
    x, y = position
    y = y + 30
    rectangle_position = (x - 2, y, x + text_width + 5, y + text_height + 10)

    # Draw the rectangle
    draw.rectangle(rectangle_position, fill="black")

    # Draw the text on top of the rectangle
    draw.text((x, y), message, fill=fill, font=font)

def calculate_room_similarity(org_request_id, available_rooms):
    """
    Calculate similarity score between the requested room and available rooms.
    :param org_request_id: The requested room number as a string.
    :param available_rooms: List of available room numbers as strings.
    :return: The most similar room number.
    """
    def similarity_score(room1, room2):
        # Determine the length of the shorter room number
        min_length = min(len(room1), len(room2))
    
        # Initialize the similarity score
        score = 0
    
        # Compare each character with decreasing weights
        for i in range(min_length):
            weight = 4 ** (10 - i)
            if room1[i] == room2[i]:
                score += weight
            else:
                score -= abs(ord(room1[i]) - ord(room2[i])) * weight
            # # Debugging output for specific cases
            # if room2 in ["7-565-1", "7-566"]:
            #     print(f"Comparing {room1} and {room2}:")
            #     print(f"  Character {i}: {room1[i]} vs {room2[i]}")
            #     print(f"  Weight: {weight}")
            #     print(f"  Score: {score}")

        return score

    similarities = [(room, similarity_score(org_request_id, room)) for room in available_rooms]
    return max(similarities, key=lambda x: x[1])[0] if similarities else None


# Flask route definitions...

@app.route('/', methods=['GET'])
def menu():
    """
    Serve a directory menu page listing all available buildings, floors, and rooms.
    Users can browse and navigate to specific room locations.
    """
    menu_results = load_results_from_json()
    menu_alias_data = load_alias_data()
    menu_note_data = load_note_data()
    menu_tags_data = load_json_file(os.path.join(SCRIPT_DIR, 'map-tags.json')) if os.path.exists(os.path.join(SCRIPT_DIR, 'map-tags.json')) else {}
    search_query = request.args.get('q', '')
    top_accessed = get_top_accessed(10)

    # Create reverse alias mapping (room_id -> alias names)
    reverse_aliases = {}
    bldg_aliases = {}
    for alias_name, replacement in menu_alias_data.items():
        if replacement.endswith("-"):
            b_id = replacement.rstrip("-")
            if b_id not in bldg_aliases:
                bldg_aliases[b_id] = []
            bldg_aliases[b_id].append(alias_name)
        else:
            if replacement not in reverse_aliases:
                reverse_aliases[replacement] = []
            reverse_aliases[replacement].append(alias_name)

    # Group rooms by building and floor
    buildings = {}
    for room_id, room_info in menu_results.items():
        floor_id_full = room_info['floor']
        b_id, f_id = split_floor_id(floor_id_full)
        if b_id not in buildings:
            buildings[b_id] = {}
        if f_id not in buildings[b_id]:
            buildings[b_id][f_id] = []
        aliases = reverse_aliases.get(room_id, [])
        tags = menu_tags_data.get(room_id, [])
        room_display = room_id[len(b_id)+1:] if room_id.startswith(b_id + "-") else room_id
        buildings[b_id][f_id].append({
            'id': room_id, 'display': room_display, 'aliases': aliases, 'tags': tags
        })

    # Add floor-only entries from image_names (floors without specific rooms)
    for img_name in image_names:
        if '-' in img_name:
            b_id, f_id = split_floor_id(img_name)
            if b_id not in buildings:
                buildings[b_id] = {}
            if f_id not in buildings[b_id]:
                buildings[b_id][f_id] = []

    # Sort helpers
    def room_sort_key(room):
        d = room['display']
        parts = d.split('-')
        try:
            return (0, int(parts[0]), parts[1] if len(parts) > 1 else '')
        except (ValueError, IndexError):
            return (1, 0, d)

    for b_id in buildings:
        for f_id in buildings[b_id]:
            buildings[b_id][f_id].sort(key=room_sort_key)

    def get_building_name(b_id):
        if b_id in menu_note_data and 'building_name' in menu_note_data[b_id]:
            return menu_note_data[b_id]['building_name']
        return f"{b_id}동"

    def bldg_sort_key(b_id):
        try:
            return (0, int(b_id))
        except ValueError:
            return (1, 0)

    def floor_sort_key(f_id):
        order = {'B': -2, 'G': -1, 'L': 0}
        if f_id.upper() in order:
            return (order[f_id.upper()], '')
        try:
            return (1, int(f_id))
        except ValueError:
            return (2, f_id)

    sorted_bldg_ids = sorted(buildings.keys(), key=bldg_sort_key)
    total_rooms = len(menu_results)
    total_buildings = len(buildings)

    # Build HTML
    html = []
    html.append(f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MeetMap</title>
    <link rel="icon" href="/assets/favicon.ico" type="image/x-icon">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', Arial, sans-serif;
            background-color: #f0f2f5;
            color: #333;
            line-height: 1.5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a237e, #283593);
            color: white;
            padding: 20px 16px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .header h1 {{ font-size: 1.4em; margin-bottom: 2px; }}
        .header .subtitle {{ font-size: 0.85em; opacity: 0.8; }}
        .search-container {{
            max-width: 500px;
            margin: 12px auto 0;
            position: relative;
        }}
        .search-input {{
            width: 100%;
            padding: 10px 16px 10px 40px;
            border: none;
            border-radius: 25px;
            font-size: 15px;
            outline: none;
            background: rgba(255,255,255,0.95);
            color: #333;
            font-family: inherit;
        }}
        .search-input::placeholder {{ color: #999; }}
        .search-icon {{
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: #888;
            font-size: 16px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 12px;
        }}
        .stats {{
            text-align: center;
            padding: 8px;
            color: #666;
            font-size: 0.85em;
        }}
        .building {{
            margin-bottom: 10px;
            border-radius: 10px;
            overflow: hidden;
            background: white;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        .building-header {{
            padding: 14px 16px;
            background: #37474f;
            color: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
            -webkit-tap-highlight-color: transparent;
        }}
        .building-header:hover {{ background: #455a64; }}
        .building-header:active {{ background: #546e7a; }}
        .building-name {{ font-size: 1.05em; font-weight: 600; }}
        .building-aliases {{
            font-size: 0.78em;
            opacity: 0.7;
            margin-top: 2px;
        }}
        .building-meta {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }}
        .building-count {{
            font-size: 0.78em;
            background: rgba(255,255,255,0.2);
            padding: 2px 8px;
            border-radius: 12px;
        }}
        .toggle-icon {{
            font-size: 0.8em;
            transition: transform 0.2s;
        }}
        .building-content {{
            display: none;
        }}
        .building-content.open {{ display: block; }}
        .building.open .toggle-icon {{ transform: rotate(180deg); }}
        .floor-section {{
            border-bottom: 1px solid #eee;
        }}
        .floor-section:last-child {{ border-bottom: none; }}
        .floor-header {{
            padding: 10px 16px;
            background: #eceff1;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .floor-name {{
            font-weight: 600;
            font-size: 0.92em;
            color: #37474f;
        }}
        .floor-link {{
            font-size: 0.78em;
            color: #1565c0;
            text-decoration: none;
            padding: 4px 10px;
            border-radius: 12px;
            background: #e3f2fd;
        }}
        .floor-link:hover {{ background: #bbdefb; }}
        .room-grid {{
            padding: 10px 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .room-chip {{
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            padding: 7px 12px;
            background: #e3f2fd;
            color: #1565c0;
            border-radius: 8px;
            text-decoration: none;
            font-size: 0.88em;
            transition: background 0.15s, color 0.15s;
            border: 1px solid #bbdefb;
            min-width: 50px;
            text-align: center;
        }}
        .room-chip:hover {{
            background: #1565c0;
            color: white;
            border-color: #1565c0;
        }}
        .room-chip:active {{
            background: #0d47a1;
            color: white;
        }}
        .room-chip .alias {{
            font-size: 0.75em;
            color: #666;
            margin-top: 1px;
        }}
        .room-chip:hover .alias {{ color: rgba(255,255,255,0.8); }}
        .room-chip .tag {{
            display: inline-block;
            font-size: 0.65em;
            background: #e3f2fd;
            color: #1565c0;
            padding: 1px 5px;
            border-radius: 3px;
            margin-left: 4px;
            vertical-align: middle;
        }}
        .room-chip:hover .tag {{ background: rgba(255,255,255,0.25); color: white; }}
        .no-results {{
            text-align: center;
            padding: 40px 20px;
            color: #999;
            font-size: 0.95em;
            display: none;
        }}
        .empty-floor {{
            padding: 10px 16px;
            color: #aaa;
            font-size: 0.85em;
            font-style: italic;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #aaa;
            font-size: 0.75em;
        }}
        .footer a {{ color: #999; }}
        .hidden {{ display: none !important; }}
        .top-section {{
            margin-bottom: 12px;
            border-radius: 10px;
            overflow: hidden;
            background: white;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        .top-header {{
            padding: 14px 16px;
            background: #546e7a;
            color: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
            -webkit-tap-highlight-color: transparent;
        }}
        .top-header:hover {{ background: #607d8b; }}
        .top-title {{
            font-size: 1.05em;
            font-weight: 600;
        }}
        .top-subtitle {{
            font-size: 0.78em;
            opacity: 0.85;
            margin-top: 2px;
        }}
        .top-content {{
            display: none;
        }}
        .top-content.open {{ display: block; }}
        .top-section .toggle-icon {{
            font-size: 0.8em;
            transition: transform 0.2s;
            color: white;
            transform: rotate(180deg);
        }}
        .top-section.open .toggle-icon {{ transform: rotate(0deg); }}
        .top-grid {{
            padding: 10px 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .top-chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 10px 14px;
            background: white;
            color: #222;
            border-radius: 8px;
            text-decoration: none;
            font-size: 0.9em;
            transition: background 0.15s, color 0.15s, box-shadow 0.15s;
            border: 1px solid #ccc;
            min-width: 60px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        .top-chip:hover {{
            background: #1a237e;
            color: white;
            border-color: #1a237e;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}
        .top-chip:active {{
            background: #0d1642;
            color: white;
        }}
        .top-info {{
            display: flex;
            flex-direction: column;
            line-height: 1.5;
        }}
        .top-bldg {{
            font-weight: 600;
        }}
        .top-floor-room {{
        }}
        .top-alias {{
            color: #555;
            font-size: 0.85em;
        }}
        .top-chip:hover .top-alias {{ color: rgba(255,255,255,0.85); }}
        .top-empty {{
            padding: 16px;
            text-align: center;
            color: #aaa;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MeetMap <span style="font-size:0.7em; font-weight:300;">the Indoor Location Finder</span></h1>
        <div class="subtitle" style="font-size:0.75em; opacity:0.55;">presented by Ph.D., Seokho Son</div>
        <div class="search-container">
            <span class="search-icon">🔍</span>
            <input type="text" class="search-input" id="searchInput"
                   placeholder="Search building, floor, room number or name..."
                   autocomplete="off" value="{search_query}">
        </div>
    </div>
    <div class="container">
        <div class="stats">{total_buildings} Buildings · {total_rooms} Locations</div>
''')

    # Top 10 popular locations section
    if top_accessed:
        html.append('''<div class="top-section" id="topSection">
    <div class="top-header" onclick="toggleTop()">
        <div>
            <div class="top-title">Top 10</div>
            <div class="top-subtitle">Most viewed locations (last 100 days)</div>
        </div>
        <span class="toggle-icon">▲</span>
    </div>
    <div class="top-content" id="topContent">
        <div class="top-grid">''')
        for rank, (room_id, count) in enumerate(top_accessed, 1):
            # Build display name with building/floor context
            top_room = room_id
            top_bldg = ''
            top_floor = ''
            if room_id in menu_results:
                floor_full = menu_results[room_id]['floor']
                b_tmp, f_tmp = split_floor_id(floor_full)
                top_room = room_id[len(b_tmp)+1:] if room_id.startswith(b_tmp + "-") else room_id
                top_bldg = get_building_name(b_tmp)
                top_floor = f'{f_tmp}층'
            elif room_id in image_names:
                # Floor-level request
                b_tmp, f_tmp = split_floor_id(room_id)
                top_bldg = get_building_name(b_tmp)
                top_room = ''
                top_floor = f'{f_tmp}층'
            # Line 1: building
            bldg_html = f'<span class="top-bldg">{top_bldg}</span>' if top_bldg else ''
            # Line 2: floor + room
            floor_room_parts = [p for p in [top_floor, top_room] if p]
            floor_room_text = ' '.join(floor_room_parts)
            floor_room_html = f'<span class="top-floor-room">{floor_room_text}</span>' if floor_room_text else ''
            # Line 3: aliases
            alias_html = ''
            if room_id in reverse_aliases and reverse_aliases[room_id]:
                display_for_cmp = top_room if top_room else top_floor
                meaningful = [a for a in reverse_aliases[room_id] if a.replace(' ', '') != display_for_cmp.replace(' ', '')]
                if meaningful:
                    alias_html = f'<span class="top-alias">{" · ".join(meaningful)}</span>'
            html.append(f'            <a href="/r/{room_id}" class="top-chip" target="_blank"><span class="top-info">{bldg_html}{floor_room_html}{alias_html}</span></a>')
        html.append('        </div>\n    </div>\n</div>')
    else:
        html.append('''<div class="top-section">
    <div class="top-header" style="cursor: default;">
        <div>
            <div class="top-title">Top 10</div>
            <div class="top-subtitle">Most viewed locations (last 100 days)</div>
        </div>
    </div>
    <div class="top-content">
        <div class="top-empty">No access history yet</div>
    </div>
</div>''')

    html.append('''        <div id="buildingList">
''')

    for b_id in sorted_bldg_ids:
        building_name = get_building_name(b_id)
        b_al = bldg_aliases.get(b_id, [])

        # Filter out aliases that are redundant with building_name
        # An alias is redundant if every word is a substring of building_name,
        # or if every character of the alias (ignoring spaces) appears in order within building_name
        def _alias_redundant(alias, name):
            words = alias.split()
            if all(w in name for w in words):
                return True
            # Character-level check: all chars of alias found in name (for cases like 드론동 ⊂ 드론시험동)
            chars = alias.replace(' ', '')
            if all(ch in name for ch in chars):
                return True
            return False

        b_al_filtered = [a for a in b_al if not _alias_redundant(a, building_name)]

        # Remove words already in building_name from displayed aliases
        def _trim_alias(alias, name):
            return ' '.join(w for w in alias.split() if w not in name).strip()

        b_al_display = [_trim_alias(a, building_name) or a for a in b_al_filtered]

        floors = buildings[b_id]
        sorted_floors = sorted(floors.keys(), key=floor_sort_key)
        room_count = sum(len(floors[f]) for f in floors)

        # Build search data for this building
        search_parts = [b_id, building_name] + b_al
        for f_id in sorted_floors:
            search_parts.append(f_id)
            for room in floors[f_id]:
                search_parts.extend([room['id'], room['display']] + room['aliases'] + room.get('tags', []))
        search_data = ' '.join(search_parts).lower().replace('"', ' ')

        # Building-level text (for distinguishing building vs room match)
        bldg_text = ' '.join([b_id, building_name] + b_al).lower().replace('"', ' ')

        aliases_html = f'<div class="building-aliases">{", ".join(b_al_display)}</div>' if b_al_display else ''

        html.append(f'''<div class="building" data-search="{search_data}" data-bldg="{bldg_text}">
    <div class="building-header" onclick="toggleBuilding(this)">
        <div>
            <div class="building-name">{building_name}</div>
            {aliases_html}
        </div>
        <div class="building-meta">
            <span class="building-count">{room_count}</span>
            <span class="toggle-icon">▼</span>
        </div>
    </div>
    <div class="building-content">''')

        for f_id in sorted_floors:
            rooms = floors[f_id]
            floor_display = f'{f_id}층'
            floor_full_id = f'{b_id}-{f_id}'
            has_floor_image = floor_full_id in image_names
            floor_link = f'<a href="/r/{floor_full_id}" class="floor-link" target="_blank">Full floor view →</a>' if has_floor_image else ''

            html.append(f'''
        <div class="floor-section">
            <div class="floor-header">
                <span class="floor-name">{floor_display}</span>
                {floor_link}
            </div>''')

            if rooms:
                html.append('            <div class="room-grid">')
                for room in rooms:
                    alias_html = ''
                    if room['aliases']:
                        meaningful = [a for a in room['aliases'] if a.replace(' ', '') != room['display'].replace(' ', '')]
                        if meaningful:
                            alias_html = f'<span class="alias">{" · ".join(meaningful)}</span>'
                    tags_html = ''.join(f'<span class="tag">{t}</span>' for t in room.get('tags', []))
                    room_search = f"{room['id']} {room['display']} {' '.join(room['aliases'])} {' '.join(room.get('tags', []))}".lower().replace('"', ' ')
                    html.append(f'                <a href="/r/{room["id"]}" class="room-chip" target="_blank" data-room="{room_search}">{room["display"]}{tags_html}{alias_html}</a>')
                html.append('            </div>')
            else:
                html.append('            <div class="empty-floor">No individual locations (use full floor view)</div>')

            html.append('        </div>')

        html.append('    </div>\n</div>')

    html.append('''
        </div>
        <div class="no-results" id="noResults">No results found</div>
        <div class="footer">© Ph.D. Seokho Son · <a href="/api">API</a></div>
    </div>
    <script>
        function toggleTop() {
            var section = document.getElementById('topSection');
            var content = document.getElementById('topContent');
            if (section) {
                section.classList.toggle('open');
                content.classList.toggle('open');
            }
        }

        function toggleBuilding(header) {
            var building = header.closest('.building');
            var content = building.querySelector('.building-content');
            building.classList.toggle('open');
            content.classList.toggle('open');
        }

        var searchInput = document.getElementById('searchInput');
        var TAG_SYNONYMS = {
            '화상': '영상',
            '온라인': '영상'
        };
        searchInput.addEventListener('input', function() {
            var query = this.value.trim().toLowerCase();
            // Expand synonyms: also search canonical term
            var queries = [query];
            if (TAG_SYNONYMS[query]) {
                queries.push(TAG_SYNONYMS[query]);
            }
            var buildings = document.querySelectorAll('.building');
            var topSection = document.getElementById('topSection');
            var hasResults = false;

            if (!query) {
                if (topSection) topSection.classList.remove('hidden');
                buildings.forEach(function(b) {
                    b.classList.remove('hidden', 'open');
                    b.querySelector('.building-content').classList.remove('open');
                    b.querySelectorAll('.room-chip').forEach(function(r) { r.classList.remove('hidden'); });
                    b.querySelectorAll('.floor-section').forEach(function(f) { f.classList.remove('hidden'); });
                });
                document.getElementById('noResults').style.display = 'none';
                return;
            }

            if (topSection) topSection.classList.add('hidden');

            function matchesAny(text, qs) {
                for (var i = 0; i < qs.length; i++) {
                    if (text.indexOf(qs[i]) !== -1) return true;
                }
                return false;
            }

            buildings.forEach(function(building) {
                var searchData = building.getAttribute('data-search');
                if (!matchesAny(searchData, queries)) {
                    building.classList.add('hidden');
                    return;
                }

                building.classList.remove('hidden');
                building.classList.add('open');
                building.querySelector('.building-content').classList.add('open');
                hasResults = true;

                var bldgText = building.getAttribute('data-bldg');
                var isBuildingMatch = matchesAny(bldgText, queries);

                if (isBuildingMatch) {
                    building.querySelectorAll('.room-chip').forEach(function(r) { r.classList.remove('hidden'); });
                    building.querySelectorAll('.floor-section').forEach(function(f) { f.classList.remove('hidden'); });
                } else {
                    building.querySelectorAll('.floor-section').forEach(function(floor) {
                        var chips = floor.querySelectorAll('.room-chip');
                        var floorMatch = false;
                        chips.forEach(function(chip) {
                            var roomData = chip.getAttribute('data-room');
                            if (matchesAny(roomData, queries)) {
                                chip.classList.remove('hidden');
                                floorMatch = true;
                            } else {
                                chip.classList.add('hidden');
                            }
                        });
                        if (floorMatch) {
                            floor.classList.remove('hidden');
                        } else {
                            floor.classList.add('hidden');
                        }
                    });
                }
            });

            document.getElementById('noResults').style.display = hasResults ? 'none' : 'block';
        });

        // Trigger search if query parameter is present
        if (searchInput.value) {
            searchInput.dispatchEvent(new Event('input'));
        }
    </script>
</body>
</html>''')

    return Response(''.join(html), content_type='text/html; charset=utf-8')


@app.route('/api', methods=['GET'])
def list_apis():
    """
    list all available API endpoints.
    :return: HTML page with information about all routes.
    """
    # Collect routes, skip internal ones
    skip_endpoints = {'static', 'favicon'}
    grouped = {}   # endpoint_name -> {methods, urls[], description}
    for rule in app.url_map.iter_rules():
        ep = rule.endpoint
        if ep in skip_endpoints:
            continue
        url = str(rule)
        methods = sorted(rule.methods - {'OPTIONS', 'HEAD'})
        if ep not in grouped:
            doc = (app.view_functions[ep].__doc__ or '').strip()
            # Take only the first sentence/line as summary
            first_line = doc.split('\n')[0].strip()
            # Remove leading docstring artifacts
            for prefix in [':return:', ':param']:
                if first_line.startswith(prefix):
                    first_line = ''
                    break
            grouped[ep] = {
                'methods': methods,
                'urls': [],
                'description': first_line if first_line else ''
            }
        grouped[ep]['urls'].append(url)

    # Manual short descriptions (override verbose docstrings)
    desc_overrides = {
        'menu': 'Directory page — browse buildings, floors, and rooms (client-side search)',
        'list_apis': 'This page — list of all API endpoints',
        'list_room_ids': 'List all available room IDs and aliases (JSON)<br>'
            '<code>key</code> or <code>k</code> — filter by keyword',
        'get_alias': 'Reload and return alias mappings (JSON)',
        'view_room_highlighted': 'View a room location on the floor map.<br>'
            '<code>returnType=html</code> (default) — interactive web view<br>'
            '<code>returnType=file</code> — floor map image (PNG)<br>'
            '<code>returnType=json</code> — room location info (JSON)<br>'
            '&nbsp;&nbsp;└ <code>strict=true</code> — disable fuzzy match (404 if no exact match)<br>'
            '<code>note</code> / <code>n</code> / <code>label</code> / <code>title</code> — display note text on map<br>'
            '<code>x</code>, <code>y</code> — mark a custom point on the map',
        'validate_images': 'Generate and combine validation images for testing',
    }
    for ep, desc in desc_overrides.items():
        if ep in grouped:
            grouped[ep]['description'] = desc

    # Build HTML
    html = '''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MeetMap API</title>
<link rel="icon" href="/assets/favicon.ico">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',Arial,sans-serif;
     background:#f0f2f5;color:#222;line-height:1.6}
.wrap{max-width:700px;margin:0 auto;padding:16px}
h1{font-size:1.3em;color:#fff;background:linear-gradient(135deg,#37474f,#546e7a);
   padding:18px 20px;border-radius:10px;margin-bottom:16px;text-align:center}
h1 small{display:block;font-size:0.6em;font-weight:400;opacity:0.85;margin-top:4px}
.card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,0.08);
      margin-bottom:10px;padding:14px 16px;transition:box-shadow 0.2s}
.card:hover{box-shadow:0 2px 8px rgba(0,0,0,0.13)}
.ep-name{font-weight:600;font-size:1em;color:#37474f}
.ep-desc{font-size:0.88em;color:#555;margin:4px 0 8px}
.urls{display:flex;flex-wrap:wrap;gap:6px}
.url-tag{display:inline-block;background:#e8eaf6;color:#283593;font-family:monospace;
         font-size:0.82em;padding:3px 8px;border-radius:4px}
.method{display:inline-block;background:#e0f2f1;color:#00695c;font-size:0.75em;
        font-weight:600;padding:2px 6px;border-radius:3px;margin-right:6px;vertical-align:middle}
.footer{text-align:center;padding:14px;font-size:0.8em;color:#888}
.footer a{color:#546e7a;text-decoration:none}
</style></head><body><div class="wrap">
<h1>MeetMap API<small>Available Endpoints</small></h1>
'''
    # Preferred display order
    order = ['menu', 'view_room_highlighted', 'list_room_ids', 'get_alias', 'validate_images', 'list_apis']
    sorted_eps = [ep for ep in order if ep in grouped]
    sorted_eps += [ep for ep in grouped if ep not in order]

    for ep in sorted_eps:
        info = grouped[ep]
        methods_html = ''.join(f'<span class="method">{m}</span>' for m in info['methods'])
        urls_html = ''.join(f'<span class="url-tag">{u}</span>' for u in sorted(info['urls']))
        desc = info['description']
        html += f'''<div class="card">
<div class="ep-name">{methods_html}{ep}</div>
{f'<div class="ep-desc">{desc}</div>' if desc else ''}
<div class="urls">{urls_html}</div>
</div>
'''

    html += '''<div class="footer">© Ph.D. Seokho Son · <a href="/">Menu</a></div>
</div></body></html>'''
    return Response(html, content_type='text/html; charset=utf-8')

def sort_key(name):
    """
    Sort key function to sort room numbers based on the pattern '2-2', '10-1', etc.
    - Sorts numerically before and after the '-' character.
    """
    parts = name.split('-')
    
    # Convert the first part to an integer for numeric sorting
    first_part = int(parts[0])
    
    # Check if the second part is a number
    second_part = parts[1]
    if second_part.isdigit():
        second_part = (0, int(second_part))  # Numbers have higher priority (0)
    else:
        second_part = (1, second_part)  # Strings have lower priority (1)

    return (first_part, second_part)

@app.route('/room', methods=['GET'])
@app.route('/r', methods=['GET'])
def list_room_ids():
    """
    List all available room numbers and aliases.
    :return: JSON response with a list of all room numbers and aliases.
    """
    keyword = request.args.get('key', '').lower() or request.args.get('k', '').lower()
    analyzed_results = load_results_from_json()
    sorted_room_ids = sorted(analyzed_results.keys(), key=sort_key)
    
    if keyword:
        sorted_room_ids = [room_id for room_id in sorted_room_ids if keyword in room_id.lower()]
    
    global alias_data
    alias_data = load_alias_data()
    if keyword:
        filtered_alias_data = {alias: replacement for alias, replacement in alias_data.items() if keyword in alias.lower() or keyword in replacement.lower()}
    else:
        filtered_alias_data = alias_data

    # Convert filtered_alias_data to a list of dictionaries
    alias_list = [{"alias": alias, "replacement": replacement} for alias, replacement in filtered_alias_data.items()]

    print(f"Total Room Numbers: {len(sorted_room_ids)}")
    print(sorted_room_ids)
    response = {
        "room_count": len(sorted_room_ids),
        "rooms": sorted_room_ids,
        "alias_count": len(alias_list),
        "aliases": alias_list
    }
    return Response(
        json.dumps(response, ensure_ascii=False),  # Ensure UTF-8 encoding
        content_type="application/json; charset=utf-8",
        status=200
    )


@app.route('/alias', methods=['GET'])
def get_alias():
    """
    Reload map-alias.json and return the updated list.
    """
    global alias_data
    try:
        alias_data = load_alias_data()
        response = json.dumps(alias_data, ensure_ascii=False)
        return Response(response, content_type='application/json; charset=utf-8')
    except FileNotFoundError:
        return jsonify({"error": "map-alias.json file not found."}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding map-alias.json file."}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    
# @app.route('/room', methods=['POST'])
# def add_room():
#     """
#     add a new room number with its data.
#     :return: JSON response indicating success or error.
#     """
#     analyzed_results = load_results_from_json()
#     data = request.json
#     room_id = data.get('room_id')

#     if room_id and room_id not in analyzed_results:
#         analyzed_results[room_id] = data
#         save_results_to_json(analyzed_results)
#         return jsonify({"message": "Room added"}), 201
#     else:
#         return jsonify({"error": "Invalid request or room number already exists"}), 400

def expand_range(range_str):
    """
    Expands a range string into a list of all encompassed values.
    This function is used to expand a range string (e.g., "111~223") into a complete list of values within that range.
    :param range_str: A string representing a range of numbers.
    :return: A list of strings representing all values within the range.
    """
    # Use regular expression to extract the range's start and end numbers
    match = re.match(r'(\d+)(~|,|-)(\d+)', range_str)
    if match:
        start, _, end = match.groups()
        start, end = int(start), int(end)
        return [str(i) for i in range(start, end + 1)]
    else:
        return [range_str.split('-')[0]]  # If no range is specified, return the number itself

def is_number_in_range(num_str, range_str):
    """
    Check if a given number string is within a specified range.
    This function checks whether a number (in string format) falls within a specified range string (e.g., "100-200").
    :param num_str: The number string to check.
    :param range_str: The range string against which the number is to be checked.
    :return: Boolean indicating if the number is within the range.
    """
    num_str = str(num_str).split('-')[0]  # Convert the number to a string and ignore suffixes like '-5'
    expanded_range = expand_range(range_str)
    return num_str in expanded_range

@app.route('/assets/favicon.ico')
def favicon():
    return send_from_directory('assets', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/room/<request_id>', methods=['GET'])
@app.route('/view/<request_id>', methods=['GET'])
@app.route('/find/<request_id>', methods=['GET'])
@app.route('/dir/<request_id>', methods=['GET'])
@app.route('/r/<request_id>', methods=['GET'])
@app.route('/v/<request_id>', methods=['GET'])
@app.route('/f/<request_id>', methods=['GET'])
@app.route('/d/<request_id>', methods=['GET'])
def view_room_highlighted(request_id):
    """
        View an image with a specific room number highlighted.
        This endpoint sends an image with the specified room number highlighted, indicating its location.

        Query Parameters:
            :param request_id: The room number to be highlighted in the image.
            :param note: (str) Note text to display (optional)
            :param n: (str) Alternative parameter for note (optional)
            :param label: (str) Alternative parameter for note (optional) 
            :param title: (str) Alternative parameter for note (optional)
            :param returnType: (str) Response type - 'html' for web view or 'file' for direct image (optional)
        
        Returns:
            :return: HTML response with the image or file response based on returnType parameter.
            - HTML response: Rendered template with the image
            - File response: Direct image file
    """

    try:
        x_param = request.args.get('x')
        y_param = request.args.get('y')
        note_param = (
            request.args.get('note') or 
            request.args.get('n') or 
            request.args.get('label') or 
            request.args.get('title')
        )

        org_request_id = request_id
        # Replace multiple consecutive "-" with a single "-"
        request_id = re.sub(r'-+', '-', request_id)

        # Normalize request_id by removing spaces
        normalized_request_id = request_id.replace(" ", "")
        
        global alias_data
        alias_data = load_alias_data()
        analyzed_results = load_results_from_json()

        # Check alias_data for a matching key
        aliasReplacement = ""
        if request_id not in analyzed_results:
            for alias, replacement in alias_data.items():
                normalized_alias = alias.replace(" ", "")
                # print(f"Normalized Alias: {normalized_alias}")
                if normalized_alias in normalized_request_id:
                    aliasReplacement = replacement
                    request_id = re.sub(re.escape(normalized_alias), replacement, normalized_request_id, flags=re.IGNORECASE)
                    break    

        # Replace multiple consecutive "-" with a single "-"
        request_id = re.sub(r'-+', '-', request_id)
        
        # Replace the patterns with "-" only if aliasReplacement is empty and request_id is not in analyzed_results
        if request_id not in analyzed_results:
            # Define the patterns to be replaced
            patterns_to_replace = ["동"]
            if aliasReplacement == "" :
                for pattern in patterns_to_replace:
                    request_id = request_id.replace(pattern, "-")
                if request_id.endswith("호") or request_id.endswith("층"):
                    request_id = request_id[:-1]                
            else:
                # If aliasReplacement is not empty, check and remove the last "호" from request_id if necessary
                if not (aliasReplacement.endswith("호") or aliasReplacement.endswith("층")) and (request_id.endswith("호") or request_id.endswith("층")):
                    request_id = request_id[:-1]


        
        # Replace multiple consecutive "-" with a single "-"
        request_id = re.sub(r'-+', '-', request_id)
        request_id = request_id.replace(" ", "")

        request_id = request_id.upper()

        isFloorRequest = False
        if request_id in image_names:
            isFloorRequest = True

        # Check if request_id starts with any of the image names or if it exists in analyzed_results
        if request_id not in analyzed_results and not any(request_id.startswith(name.upper()) for name in image_names):
            response = {
                "Message": f"Map for {request_id} is not supported yet.",
                "Supported Maps": sorted(image_names, key=sort_key)
            }
            return Response(
                json.dumps(response, ensure_ascii=False),  # Ensure non-ASCII characters are preserved
                content_type="application/json; charset=utf-8",
                status=404
            )

        request_id, image_size, room_x, room_y, room_w, room_h, floor_id, building_id, floor_only_id, similar_room = search_and_get_room_info(request_id)

        # Record access for popularity tracking
        increment_access_count(request_id)

        if isFloorRequest:
            floor_id = request_id
            building_id, floor_only_id = split_floor_id(floor_id)

        floor_image_path = os.path.join(tmp_directory, f"{floor_id}-map.png")

        skyview_image_base64, room_image_base64 = get_suppliment_images(room_id=request_id)

        matched_note_data = get_note_data(request_id, floor_id, building_id)
        # if matched_note_data:
        #     print("Matched Note Data:", matched_note_data)
        # else:
        #     print("No matching note data found.")

        b_name = matched_note_data.get('building_name') if matched_note_data and matched_note_data.get('building_name') else None
        b_north_x = matched_note_data.get('north_x') if matched_note_data and matched_note_data.get('north_x') else None
        b_north_y = matched_note_data.get('north_y') if matched_note_data and matched_note_data.get('north_y') else None
        b_main_gate_x = matched_note_data.get('main_gate_x') if matched_note_data and matched_note_data.get('main_gate_x') else None
        b_main_gate_y = matched_note_data.get('main_gate_y') if matched_note_data and matched_note_data.get('main_gate_y') else None
        b_note = matched_note_data.get('note') if matched_note_data and matched_note_data.get('note') else None
        b_skyview_link = matched_note_data.get('skyview_link') if matched_note_data and matched_note_data.get('skyview_link') not in [None, ""] else None
        
        return_type = request.args.get('returnType', '').lower()

        destinationLabelText = f"{request_id} ?" if similar_room else request_id
        buildingText = f"{b_name}" if b_name else f"{building_id}동"
        noteText = f"{note_param}" if note_param else ""


        if return_type == "file":
            # Return image file directly
            return send_file(floor_image_path, mimetype='image/png')
        elif return_type == "json":
            # Return room location info as JSON for programmatic use
            strict_mode = request.args.get('strict', '').lower() in ('true', '1', 'yes')
            if strict_mode and similar_room:
                return Response(
                    json.dumps({
                        "found": False,
                        "request_id": org_request_id,
                        "message": f"No exact match for '{org_request_id}'."
                    }, ensure_ascii=False),
                    content_type="application/json; charset=utf-8",
                    status=404
                )
            result = {
                "found": True,
                "exact_match": not bool(similar_room),
                "request_id": org_request_id,
                "room_id": request_id,
                "building_id": building_id,
                "building_name": buildingText,
                "floor": floor_id,
                "floor_id": floor_only_id,
                "image_size": list(image_size) if image_size else None,
                "location": {
                    "x_ratio": room_x,
                    "y_ratio": room_y,
                    "w_ratio": room_w,
                    "h_ratio": room_h
                }
            }
            if similar_room:
                result["similar_room"] = similar_room
            return Response(
                json.dumps(result, ensure_ascii=False),
                content_type="application/json; charset=utf-8",
                status=200
            )
        else:
            # Use the Base64 encoded images in HTML
            html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Map:{request_id}</title>
                <link rel="icon" href="/assets/favicon.ico" type="image/x-icon">
                <style>
                    body {{
                        margin: 0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        background-color: #f0f0f0;
                        position: relative;
                        flex-direction: column;
                    }}
                    .image-container {{
                        position: relative;
                        display: inline-block;
                        max-width: 100%;
                        height: auto;
                        overflow: hidden;
                    }}
                    img {{
                        display: block;
                        width: 100%;
                        height: 100%;
                        object-fit: contain;
                        transition: all 0.3s ease-in-out;  
                        -webkit-user-drag: none;
                        -khtml-user-drag: none;
                        -moz-user-drag: none;
                        -o-user-drag: none;
                        user-drag: none;
                        pointerEvents = 'auto';
                        user-select: none;
                        -webkit-user-select: none;
                        -ms-user-select: none;
                    }}
                    .floor-identifier {{
                        position: absolute;
                        top: 1%;
                        left: 1%;
                        color: white;
                        background-color: rgba(0, 0, 0, 0.7);
                        padding: 0.5% 1%; 
                        border-radius: 0.5vw;
                        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', Arial, sans-serif;
                        font-size: 2.1vw; 
                        cursor: pointer;
                        display: flex;
                        align-items: baseline;
                        gap: 0.5vw;
                    }}
                    .floor-identifier .fi-floor {{
                        font-size: 1.3em;
                        font-weight: bold;
                    }}
                    .floor-identifier .fi-bldg {{
                        font-size: 0.75em;
                        opacity: 0.8;
                    }}
                    .button-container {{
                        position: absolute;
                        top: 1%;
                        right: 5%;
                        display: flex;
                        gap: 10%;
                        z-index: 10; 
                    }}
                    button {{
                        padding: 3% 6%; 
                        font-size: 1.6vw; 
                        border: none;
                        border-radius: 0.5vw;
                        cursor: pointer;
                        background-color: rgba(0, 0, 255, 0.7);
                        color: white;
                    }}
                    button:hover {{
                        background-color: #0056b3;
                    }}
                    .mouse-position {{
                        position: absolute;
                        bottom: 1%;                        
                        right: 1%;
                        color: white;
                        background-color: rgba(0, 0, 0, 0.7);
                        padding: 0.5% 1%; 
                        border-radius: 0.5vw;
                        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', Arial, sans-serif;
                        font-size: 1.4vw; 
                    }}
                    .author-label {{
                        position: absolute;
                        bottom: 1%;                        
                        left: 1%;
                        color: white;
                        background-color: rgba(0, 0, 0, 0);
                        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
                        padding: 0.5% 1%; 
                        border-radius: 0.5vw;
                        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', Arial, sans-serif;
                        font-size: 1.4vw; 
                        pointer-events: none;
                    }}                    
                    .source-box {{
                        position: absolute;
                        border: solid red;
                        border-radius: 50%;
                        borderWidth: 0.8vw;
                        background-color: rgba(255, 0, 0, 0.1);
                        pointer-events: none;
                        transform: translate(-50%, -50%);
                        aspect-ratio: 1;
                        animation: blink1 3s infinite;
                    }}
                    .destination-box {{
                        position: absolute;
                        border: solid red;
                        borderWidth: 0.8vw;
                        background-color: rgba(255, 0, 0, 0.1);
                        pointer-events: none;
                        transform: translate(-50%, -50%);
                        animation: blink2 3s infinite;
                    }}
                    .box-label {{
                        transform: translate(-50%, 0);
                        user-select: none;
                        pointer-events: none;
                        position: absolute;
                        color: white;
                        background-color: rgba(0, 0, 255, 0.7);
                        padding: 0.1% 0.2%;
                        border-radius: 0.5vw;
                        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', Arial, sans-serif;
                        font-size: 1.6vw; 
                    }}                      
                    .gate-label {{
                        transform: translate(-50%, -50%);
                        user-select: none;
                        pointer-events: none;
                        position: absolute;
                        color: white;
                        background-color: rgba(0, 0, 0, 0.5);
                        padding: 0.2% 0.4%; 
                        border-radius: 0.7vw;
                        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', Arial, sans-serif;
                        font-size: 1.4vw; 
                        animation: blink3 8s infinite;
                    }}                                          
                    @keyframes blink1 {{
                        0% {{ border-color: blue; background-color: rgba(255, 255, 255, 0.01); }}
                        50% {{ border-color: red; background-color: rgba(255, 255, 0, 0.2); }}
                        100% {{ border-color: blue; background-color: rgba(255, 255, 255, 0.01); }}
                    }}
                    @keyframes blink2 {{
                        0% {{ border-color: blue; background-color: rgba(255, 255, 255, 0.01); }}
                        50% {{ border-color: red; background-color: rgba(255, 255, 0, 0.2); }}
                        100% {{ border-color: blue; background-color: rgba(255, 255, 255, 0.01); }}
                    }}  
                    @keyframes blink3 {{
                        0% {{ background-color: rgba(0, 0, 0, 0.4); color: rgba(255, 255, 255, 1); }}
                        50% {{ background-color: rgba(0, 0, 0, 0.2); color: rgba(255, 255, 255, 0.6); }}
                        100% {{ background-color: rgba(0, 0, 0, 0.4); color: rgba(255, 255, 255, 1); }}
                    }}                        
                </style>
            </head>
            <body>
                <div class="image-container">
                    <img id="floorImage" src="data:image/png;base64,{convert_image_to_base64(floor_image_path)}" alt="Request: {org_request_id} ({request_id})" />
                    <div class="mouse-position" id="mousePosition">X: 0 / Y: 0</div>
                    <div class="author-label">© Ph.D. Seokho Son</div>                    
                    <div id="northLabel" class="gate-label" style="display: none;">북쪽</div>
                    <div id="mainGateLabel" class="gate-label" style="display: none;">주출입구방향</div>   
                    <div id="destinationLabel" class="box-label" style="display: none;">{destinationLabelText}</div>
                    <div id="sourceLabel" class="box-label" style="display: none;">X / Y</div>                                 
                    <div id="destinationBox" class="destination-box" style="display: none;"></div>
                    <div id="sourceBox" class="source-box" style="display: none;"></div>
                    <div class="button-container">
                        <button id='shareWindowButton' onclick='openShareWindow()'>share</button>
                        {"<button id='toggleSkyviewButton' onclick='toggleSkyview()'>sky</button>"}
                        {"<button id='toggleRoomviewButton' onclick='toggleRoomview()'>inside</button>" if room_image_base64 else ""}
                    </div>
                    <img id="skyviewImage" src="data:image/png;base64,{skyview_image_base64}" style="display: none; position: absolute; top: 50%; left: 75%; transform: translate(-50%, -50%); max-width: 45%; max-height: 80%; pointer-events: none;" />
                    <img id="roomviewImage" src="data:image/png;base64,{room_image_base64}" style="display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); max-width: 80%; max-height: 80%; pointer-events: none;" />
                    <div class="floor-identifier"><span class="fi-floor">{floor_only_id}층</span><span class="fi-bldg">{buildingText}</span></div>
                </div>
            
                <script>

                    const isFloorRequest = {str(isFloorRequest).lower()};
                    const bNorthX = {b_north_x if b_north_x else 'null'};
                    const bNorthY = {b_north_y if b_north_y else 'null'};
                    const bMainGateX = {b_main_gate_x if b_main_gate_x else 'null'};
                    const bMainGateY = {b_main_gate_y if b_main_gate_y else 'null'};
                    const bSkyviewLink = "{b_skyview_link if b_skyview_link else 'null'}";
                    const xParam = {x_param if x_param is not None else 'null'};
                    const yParam = {y_param if y_param is not None else 'null'};

                    function toggleSkyview() {{
                        const skyviewImage = document.getElementById('skyviewImage');
                        const roomviewImage = document.getElementById('roomviewImage');
                        if (bSkyviewLink !== 'null') {{
                            window.open(bSkyviewLink, '_blank');
                        }} else {{
                            if (skyviewImage.style.display === 'none') {{
                                skyviewImage.style.display = 'block';
                                roomviewImage.style.display = 'none';
                            }} else {{
                                skyviewImage.style.display = 'none';
                            }}
                        }}
                    }}

                    function openShareWindow() {{
                        const url = decodeURIComponent(window.location.href);
                        generateQRCode(url);      
                    }}

                    function toggleRoomview() {{
                        const roomviewImage = document.getElementById('roomviewImage');
                        const skyviewImage = document.getElementById('skyviewImage');
                        if (roomviewImage.style.display === 'none') {{
                            roomviewImage.style.display = 'block';
                            skyviewImage.style.display = 'none';
                        }} else {{
                            roomviewImage.style.display = 'none';
                        }}
                    }}                    

                    document.getElementById('floorImage').addEventListener('mousemove', function(event) {{
                        const img = event.target;
                        const rect = img.getBoundingClientRect();
                        const x = event.clientX - rect.left;
                        const y = event.clientY - rect.top;
                        const xPercent = (x / rect.width) * 100;
                        const yPercent = (y / rect.height) * 100;
                        document.getElementById('mousePosition').textContent = `X: ${{xPercent.toFixed(1)}} / Y: ${{yPercent.toFixed(1)}}`;
                    }});

                    document.getElementById('floorImage').addEventListener('contextmenu', function(event) {{
                            event.preventDefault();
                            alert("This image is protected and cannot be saved.");
                    }});

                    document.getElementById('floorImage').addEventListener('click', function(event) {{
                        const img = event.target;
                        const rect = img.getBoundingClientRect();
                        const x = event.clientX - rect.left;
                        const y = event.clientY - rect.top;
                        const xPercent = (x / rect.width) * 100;
                        const yPercent = (y / rect.height) * 100;

                        const sourceBox = document.getElementById('sourceBox');
                        sourceBox.style.left = `${{xPercent}}%`;
                        sourceBox.style.top = `${{yPercent}}%`;
                        sourceBox.style.width = '4%';
                        sourceBox.style.display = 'block';

                        const sourceLabel = document.getElementById('sourceLabel');
                        let labelContent = `X: ${{xPercent.toFixed(1)}}<br>Y: ${{yPercent.toFixed(1)}}`;
                        const urlParams = new URLSearchParams(window.location.search);
                        const noteText = urlParams.get('note');
                        if ("{noteText}") {{
                            labelContent = `{noteText}`;
                        }}
                        sourceLabel.innerHTML = labelContent;
                        sourceLabel.style.left = `${{xPercent}}%`;
                        sourceLabel.style.top = `${{yPercent + 5/2 + 1}}%`;
                        sourceLabel.style.display = 'block';
                        sourceLabel.style.backgroundColor = 'rgba(128, 0, 128, 0.6)';

                        // Update the URL with the new parameters
                        const newUrl = new URL(window.location.href);
                        newUrl.searchParams.set('x', xPercent.toFixed(1));
                        newUrl.searchParams.set('y', yPercent.toFixed(1));
                        window.history.pushState({{}}, '', newUrl);
                    }});

                    // Draw the destination box based on room coordinates
                    function drawDestinationBox() {{
                        const img = document.getElementById('floorImage');
                        const destinationBox = document.getElementById('destinationBox');

                        const roomX = {room_x};
                        const roomY = {room_y};
                        const roomW = {room_w + 0.1};
                        const roomH = {room_h + 0.1};

                        console.log('Drawing destination box with coordinates:', roomX, roomY, roomW, roomH); 

                        destinationBox.style.left = `${{roomX}}%`;
                        destinationBox.style.top = `${{roomY}}%`;
                        destinationBox.style.width = `${{roomW}}%`;
                        destinationBox.style.height = `${{roomH}}%`;
                        destinationBox.style.display = 'block';

                        const destinationLabel = document.getElementById('destinationLabel');
                        const orgId = `{org_request_id}`;
                        const destId = `{destinationLabelText}`;
                        let labelContent = orgId === destId ? orgId : `${{orgId}}<br>(${{destId}})`;
                        destinationLabel.innerHTML = labelContent;
                        destinationLabel.style.left = `${{roomX}}%`;
                        destinationLabel.style.top = `${{roomY + roomH /2 + 1}}%`;
                        destinationLabel.style.display = 'block';                        
                        destinationLabel.style.textAlign = 'center';
                    }}

                    // Draw the sourceBox based on query parameters
                    function drawsourceBoxFromQuery(x, y) {{
                        const sourceBox = document.getElementById('sourceBox');
                        sourceBox.style.left = `${{x}}%`;
                        sourceBox.style.top = `${{y}}%`;
                        sourceBox.style.width = '4%';
                        sourceBox.style.display = 'block';

                        const sourceLabel = document.getElementById('sourceLabel');
                        let labelContent = `X: ${{x.toFixed(1)}}<br>Y: ${{y.toFixed(1)}}`;
                        if ("{noteText}") {{
                            labelContent = `{noteText}`;
                        }}
                        sourceLabel.innerHTML = labelContent;
                        sourceLabel.style.left = `${{x}}%`;
                        sourceLabel.style.top = `${{y + 5/2 + 1}}%`;
                        sourceLabel.style.display = 'block';     
                        sourceLabel.style.backgroundColor = 'rgba(128, 0, 128, 0.6)';
                    }}

                    function initialize() {{
                        console.log('Initializing...');
                        if (!isFloorRequest) {{
                            drawDestinationBox();
                        }}

                        if (xParam !== null && yParam !== null) {{
                            drawsourceBoxFromQuery(xParam, yParam);
                        }}

                        console.log('Drawing building labels:', bNorthX, bNorthY, bMainGateX, bMainGateY); 

                        if (bNorthX !== null && bNorthY !== null) {{
                            const northLabel = document.getElementById('northLabel');
                            northLabel.style.left = `${{bNorthX}}%`;
                            northLabel.style.top = `${{bNorthY}}%`;
                            northLabel.style.display = 'block';
                        }}

                        if (bMainGateX !== null && bMainGateY !== null) {{
                            const mainGateLabel = document.getElementById('mainGateLabel');
                            mainGateLabel.style.left = `${{bMainGateX}}%`;
                            mainGateLabel.style.top = `${{bMainGateY}}%`;
                            mainGateLabel.style.display = 'block';
                        }}

                        updateBoxSize();

                    }}                

                    function updateBoxSize() {{
                        // Use image container's actual width instead of viewport width
                        // so fonts scale with the image, not the browser window
                        const container = document.querySelector('.image-container');
                        const cw = container.offsetWidth / 100; // 1% of container width in px

                        const sourceBox = document.getElementById('sourceBox');
                        const sourceLabel = document.getElementById('sourceLabel');
                        const floorIdentifier = document.querySelector('.floor-identifier');
                        const mousePosition = document.querySelector('.mouse-position');
                        const authorLabel = document.querySelector('.author-label');
                        const northLabel = document.getElementById('northLabel');
                        const mainGateLabel = document.getElementById('mainGateLabel');
                        const toggleSkyviewButton = document.getElementById('toggleSkyviewButton');
                        const toggleRoomviewButton = document.getElementById('toggleRoomviewButton');
                        const shareWindowButton = document.getElementById('shareWindowButton');

                        const boxSize = 4;
                        const fontSize = 1.6;
                        const padding = 0.5;
                        const borderThickness = 0.4;

                        sourceBox.style.width = `${{boxSize}}%`;
                        sourceBox.style.borderWidth = `${{cw * borderThickness}}px`;
                        
                        floorIdentifier.style.fontSize = `${{cw * (fontSize + 0.5)}}px`;
                        floorIdentifier.style.padding = `${{padding}}% 1%`;
                        floorIdentifier.style.borderRadius = `${{cw * 0.5}}px`;
                        mousePosition.style.fontSize = `${{cw * (fontSize - 0.2)}}px`;
                        mousePosition.style.borderRadius = `${{cw * 0.5}}px`;
                        if (authorLabel) {{
                            authorLabel.style.fontSize = `${{cw * (fontSize - 0.2)}}px`;
                        }}
                        sourceLabel.style.fontSize = `${{cw * fontSize}}px`;
                        sourceLabel.style.padding = `${{padding}}% 1%`;
                        sourceLabel.style.borderRadius = `${{cw * 0.5}}px`;

                        if (northLabel) {{
                            northLabel.style.fontSize = `${{cw * (fontSize - 0.2)}}px`;
                            northLabel.style.borderRadius = `${{cw * 0.7}}px`;
                        }}
                        if (mainGateLabel) {{
                            mainGateLabel.style.fontSize = `${{cw * (fontSize - 0.2)}}px`;
                            mainGateLabel.style.borderRadius = `${{cw * 0.7}}px`;
                        }}
                        
                        if (!isFloorRequest) {{
                            const destinationBox = document.getElementById('destinationBox');
                            const destinationLabel = document.getElementById('destinationLabel');
                            destinationBox.style.borderWidth = `${{cw * borderThickness}}px`;
                            destinationLabel.style.fontSize = `${{cw * fontSize}}px`;
                            destinationLabel.style.padding = `${{padding}}% 1%`;
                            destinationLabel.style.borderRadius = `${{cw * 0.5}}px`;
                        }}

                        if (toggleRoomviewButton) {{
                            toggleRoomviewButton.style.fontSize = `${{cw * fontSize}}px`;
                            toggleRoomviewButton.style.padding = `${{padding}}% 2%`;
                        }}
                        
                        if (toggleSkyviewButton) {{
                            toggleSkyviewButton.style.fontSize = `${{cw * fontSize}}px`;
                            toggleSkyviewButton.style.padding = `${{padding}}% 2%`;
                        }}

                        if (shareWindowButton) {{
                            shareWindowButton.style.fontSize = `${{cw * fontSize}}px`;
                            shareWindowButton.style.padding = `${{padding}}% 2%`;
                        }}

                    }}

                    // Call the function to draw the destination box when the image is loaded
                    document.addEventListener('DOMContentLoaded', function() {{
                        console.log('Document loaded');
                        const floorImage = document.getElementById('floorImage');
                        if (floorImage.complete) {{
                            console.log('Floor image already loaded');
                            initialize();
                        }} else {{
                            floorImage.addEventListener('load', function() {{
                                console.log('Floor image loaded');
                                initialize();
                            }});
                        }}
                    }});

                    window.addEventListener('resize', updateBoxSize);

                    // Add event listeners for floorIdentifier
                    const floorIdentifier = document.querySelector('.floor-identifier');
                    floorIdentifier.addEventListener('mouseenter', function() {{
                        floorIdentifier.innerHTML = '<span class="fi-floor">Copy URL</span>';
                    }});

                    floorIdentifier.addEventListener('mouseleave', function() {{
                        floorIdentifier.innerHTML = '<span class="fi-floor">{floor_only_id}층</span><span class="fi-bldg">{buildingText}</span>';
                    }});

                    floorIdentifier.addEventListener('click', function() {{
                        const url = decodeURIComponent(window.location.href);
                        if (navigator.clipboard) {{
                            navigator.clipboard.writeText(url).then(function() {{
                                if (confirm(`Copied following URL to clipboard.\n\n${{url}}\n\nDo you want to generate a QR code?`)) {{
                                    generateQRCode(url);
                                }}
                            }}, function(err) {{
                                console.error('Could not copy text: ', err);
                            }});
                        }} else {{
                            const textArea = document.createElement('textarea');
                            textArea.value = url;
                            document.body.appendChild(textArea);
                            textArea.select();
                            try {{
                                document.execCommand('copy');
                                if (confirm(`Copied following URL to clipboard.\n\n${{url}}\n\nDo you want to generate a QR code?`)) {{
                                    generateQRCode(url);
                                }}
                            }} catch (err) {{
                                console.error('Could not copy text: ', err);
                            }}
                            document.body.removeChild(textArea);
                        }}
                    }});

                    function generateQRCode(url) {{
                        const qrWindow = window.open('', '_blank', 'width=400,height=500');
                        const encodedUrl = encodeURI(url);
                        const escapedUrl = encodedUrl.replace(/'/g, "\\'");

                        const urlObj = new URL(url);
                        const hasLocation = urlObj.searchParams.has('x') && urlObj.searchParams.has('y');
                        const initialNote = urlObj.searchParams.get('note') || 
                                            urlObj.searchParams.get('n') || 
                                            urlObj.searchParams.get('label') || 
                                            urlObj.searchParams.get('title') || 
                                            '';  

                        const noteInputHtml = hasLocation ? `
                            <div class="note-container">
                                <input type="text" class="note-input" placeholder="Change the label for the location" value="${{initialNote}}">
                                <button class="update-button" onclick="updateQRWithNote()">Update</button>
                            </div>
                        ` : '';                
                        
                        const qrHtml = `
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <meta charset="UTF-8">
                                <title>QR Code</title>
                                <script>
                                    function loadQRCodeScript() {{
                                        const script = document.createElement('script');
                                        script.src = 'https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js';
                                        script.onload = () => {{
                                            console.log('QR Code library loaded successfully from jsdelivr CDN');
                                            initQR();
                                        }};
                                        script.onerror = () => {{
                                            console.log('jsdelivr CDN failed, trying cdnjs...');
                                            const backupScript = document.createElement('script');
                                            backupScript.src = 'http://cdnjs.cloudflare.com/ajax/libs/qrcode-generator/1.4.4/qrcode.min.js';
                                            backupScript.onload = () => {{
                                                console.log('QR Code library loaded successfully from cdnjs CDN');
                                                initQR();
                                            }};
                                            backupScript.onerror = () => {{
                                                console.error('Failed to load QR Code library from both CDNs');
                                                document.getElementById('qrcode').innerHTML = 'Failed to load QR Code library';
                                            }};
                                            document.head.appendChild(backupScript);
                                        }};
                                        document.head.appendChild(script);
                                    }}

                                    function downloadQRCode() {{
                                        const svg = document.querySelector('#qrcode svg');
                                        const svgData = new XMLSerializer().serializeToString(svg);
                                        const canvas = document.createElement('canvas');
                                        const ctx = canvas.getContext('2d');
                                        const img = new Image();
                                        
                                        img.onload = () => {{
                                            canvas.width = 256;
                                            canvas.height = 256;
                                            ctx.fillStyle = 'white';
                                            ctx.fillRect(0, 0, canvas.width, canvas.height);
                                            ctx.drawImage(img, 0, 0, 256, 256);
                                            
                                            const link = document.createElement('a');
                                            link.download = 'QR-{org_request_id}.png';
                                            link.href = canvas.toDataURL('image/png');
                                            link.click();
                                        }};
                                        
                                        img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
                                    }}

                                    function initQR() {{
                                        try {{
                                            const decodedUrl = decodeURI('${{escapedUrl}}');
                                            const typeNumber = 0;
                                            const errorCorrectionLevel = 'H';
                                            const qr = qrcode(typeNumber, errorCorrectionLevel);
                                            qr.addData('${{escapedUrl}}');
                                            qr.make();
                                            
                                            document.getElementById('qrcode').innerHTML = qr.createSvgTag({{
                                                cellSize: 4,
                                                margin: 4
                                            }});
                                            document.querySelector('.url-text').textContent = decodedUrl;
                                            document.getElementById('download-button').style.display = 'block';
                                            console.log('QR Code generated successfully');
                                            console.log('Encoded URL:', '${{escapedUrl}}');
                                            console.log('Decoded URL:', decodedUrl);
                                        }} catch(e) {{
                                            console.error('QR Code error:', e);
                                            document.getElementById('qrcode').innerHTML = 'Error: ' + e.message;
                                        }}
                                    }}
                                    
                                    function copyToClipboard() {{
                                        const url = document.querySelector('.url-text').textContent;
                                        if (navigator.clipboard) {{
                                            navigator.clipboard.writeText(url).then(() => {{
                                                const copyButton = document.getElementById('copy-button');
                                                copyButton.textContent = 'Copied!';
                                                setTimeout(() => {{
                                                    copyButton.textContent = 'Copy URL';
                                                }}, 2000);
                                            }});
                                        }} else {{
                                            const textArea = document.createElement('textarea');
                                            textArea.value = url;
                                            document.body.appendChild(textArea);
                                            textArea.select();
                                            try {{
                                                document.execCommand('copy');
                                                const copyButton = document.getElementById('copy-button');
                                                copyButton.textContent = 'Copied!';
                                                setTimeout(() => {{
                                                    copyButton.textContent = 'Copy URL';
                                                }}, 1500);
                                            }} catch (err) {{
                                                console.error('Could not copy text: ', err);
                                            }}
                                            document.body.removeChild(textArea);
                                        }}
                                    }}

                                    function updateQRWithNote() {{
                                        const noteInput = document.querySelector('.note-input');
                                        const newNote = noteInput.value.trim();
                                        const currentUrl = new URL('${{escapedUrl}}');
                                        
                                        currentUrl.searchParams.delete('note');
                                        currentUrl.searchParams.delete('n');
                                        currentUrl.searchParams.delete('label');
                                        currentUrl.searchParams.delete('title');
                                        
                                        if (newNote) {{
                                            currentUrl.searchParams.set('n', newNote);
                                        }}
                                        
                                        const newUrl = currentUrl.toString();
                                        const qr = qrcode(0, 'H');
                                        qr.addData(newUrl);
                                        qr.make();
                                        document.getElementById('qrcode').innerHTML = qr.createSvgTag({{
                                            cellSize: 4,
                                            margin: 4
                                        }});
                                        
                                        const decodedUrl = decodeURIComponent(currentUrl.href);
                                        document.querySelector('.url-text').textContent = decodedUrl;
                                    }}
                                <\\/script>
                                <style>
                                    body {{
                                        display: flex;
                                        flex-direction: column;
                                        justify-content: center;
                                        align-items: center;
                                        height: 100vh;
                                        margin: 0;
                                        background-color: #f0f0f0;
                                        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', Arial, sans-serif;
                                    }}
                                    #qrcode {{
                                        padding: 20px;
                                        background: white;
                                        border-radius: 10px;
                                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                                    }}
                                    #qrcode svg {{
                                        width: 256px;
                                        height: 256px;
                                    }}
                                    .url-text {{
                                        margin-top: 20px;
                                        padding: 10px;
                                        word-break: break-all;
                                        max-width: 300px;
                                        text-align: center;
                                        font-size: 12px;
                                        color: #666;
                                    }}
                                    .info-text {{
                                        margin-top: 20px;
                                        padding: 10px;
                                        word-break: break-all;
                                        max-width: 300px;
                                        text-align: center;
                                        font-size: 12px;
                                        color: black;
                                    }}                                    
                                    .button-container {{
                                        display: flex;
                                        gap: 10px;
                                        margin-top: 20px;
                                    }}
                                    
                                    #download-button, #copy-button {{
                                        padding: 10px 20px;
                                        border: none;
                                        border-radius: 5px;
                                        cursor: pointer;
                                        font-size: 14px;
                                    }}
                                    
                                    #download-button {{
                                        background-color: #007bff;
                                        color: white;
                                    }}
                                    
                                    #copy-button {{
                                        background-color: #28a745;
                                        color: white;
                                    }}
                                    
                                    #download-button:hover {{
                                        background-color: #0056b3;
                                    }}
                                    
                                    #copy-button:hover {{
                                        background-color: #218838;
                                    }}

                                    .note-container {{
                                        display: ${{hasLocation ? 'flex' : 'none'}};
                                        gap: 10px;
                                        align-items: center;
                                        margin-top: 10px;
                                        width: 100%;
                                        max-width: 300px;
                                    }}
                                    .note-input {{
                                        flex: 1;
                                        padding: 8px;
                                        border: 1px solid #ddd;
                                        border-radius: 4px;
                                        font-size: 12px;
                                    }}
                                    .update-button {{
                                        padding: 8px 15px;
                                        background-color: #6c757d;
                                        color: white;
                                        border: none;
                                        border-radius: 4px;
                                        cursor: pointer;
                                        font-size: 12px;
                                    }}
                                    .update-button:hover {{
                                        background-color: #5a6268;
                                    }}
                                </style>
                            </head>
                            <body>
                                <div id="qrcode"></div>
                                <div class="info-text">
                                    <div>Building: {buildingText} {floor_only_id}층</div>
                                    <div>Label: {org_request_id} ({destinationLabelText})</div>
                                    <div><span class="url-text"></span></div>
                                </div>
                                ${{noteInputHtml}}
                                <div class="button-container">
                                    <button id="download-button" onclick="downloadQRCode()">Download QR Code</button>
                                    <button id="copy-button" onclick="copyToClipboard()">Copy URL</button>
                                </div>
                                <script>
                                    loadQRCodeScript();
                                <\\/script>
                            </body>
                            </html>
                        `;
                        qrWindow.document.write(qrHtml);
                        qrWindow.document.close();
                    }}

                </script>
            </body>
            </html>
            """
            return html_template, 200, {'Content-Type': 'text/html'}

    except Exception as e:
        return jsonify({"Message": f"An error occurred: {e}"}), 500


def convert_image_to_base64(image_path):
    """
    Convert an image file to a Base64 string for embedding in HTML.
    :param image_path: Path to the image file.
    :return: Base64 encoded string.
    """
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def search_and_get_room_info(room_id):
    """
    Logic to get information about a specific room.
    :param room_id: The room number to get information about.
    :param force: Boolean flag to force regeneration of the information.
    :return: Tuple containing image size, room coordinates and dimensions, room_id, floor_id, building_id, floor_only_id, and similar_room flag.
    """

    analyzed_results = load_results_from_json()

    # Logic to find room information
    if room_id in analyzed_results:
        room_info = analyzed_results[room_id]
        similar_room = False  # Exact match found
    else:
        similar_room = calculate_room_similarity(room_id, analyzed_results.keys())
        if not similar_room or similar_room == room_id:  # Low similarity or no match found
            return None, None, None, None, None, None, None, None, None, False
        room_info = analyzed_results[similar_room]

    floor_id = room_info['floor']
    building_id, floor_only_id = split_floor_id(floor_id)

    # Load the floor image to get its size
    floor_image_path = os.path.join(tmp_directory, f"{floor_id}-map.png")
    if os.path.exists(floor_image_path):
        image = Image.open(floor_image_path)

        room_x = room_info['x_ratio']
        room_y = room_info['y_ratio']
        room_w = room_info['w_ratio']
        room_h = room_info['h_ratio']

        return room_id, image.size, room_x, room_y, room_w, room_h, floor_id, building_id, floor_only_id, similar_room
    else:
        return None, None, None, None, None, None, None, None, None, False

def get_suppliment_images(building_id=None, room_id=None):
    """
    Function to get skyview_image and room_image based on building_id or room_id.
    :param building_id: The building ID to find the skyview image.
    :param room_id: The room ID to find the room image.
    :return: Tuple containing the Base64 encoded skyview_image and room_image, or None if not found.
    """
    skyview_image_base64 = None
    room_image_base64 = None

    if room_id:
        building_id = room_id.split('-')[0]
        
    skyview_image_path = os.path.join("image/skyview", f"{building_id}.jpg")
    if os.path.exists(skyview_image_path):
        skyview_image_base64 = convert_image_to_base64(skyview_image_path)

    if room_id:
        room_image_path = os.path.join("image/room", f"{room_id}.jpg")
        if os.path.exists(room_image_path):
            room_image_base64 = convert_image_to_base64(room_image_path)

    return skyview_image_base64, room_image_base64

def search_and_highlight(room_id, force=False):
    """
    Logic to generate an image with a specific room number highlighted.
    :param room_id: The room number to be highlighted in the image.
    :param force: Boolean flag to force regeneration of the image.
    :return: Tuple containing the path to the generated image and its size, or (None, None) if not found.
    """

    highlighted_image_path = os.path.join(tmp_directory, f"{room_id}-highlighted.png")

    # Check if the image already exists and if force regeneration is not required
    if not force and os.path.exists(highlighted_image_path):
        with Image.open(highlighted_image_path) as img:
            return highlighted_image_path, img.size

    analyzed_results = load_results_from_json()

    # Logic to generate a new highlighted image
    if room_id in analyzed_results:
        room_info = analyzed_results[room_id]
        similar_room = None  # Exact match found
    else:
        similar_room = calculate_room_similarity(room_id, analyzed_results.keys())
        if not similar_room or similar_room == room_id:  # Low similarity or no match found
            return None, None
        room_info = analyzed_results[similar_room]

    floor_id = room_info['floor']
    building_id, floor_only_id = split_floor_id(floor_id)

    # Load the floor image and prepare for drawing
    floor_image_path = os.path.join(tmp_directory, f"{floor_id}-map.png")
    if os.path.exists(floor_image_path):
        image = Image.open(floor_image_path)
        draw = ImageDraw.Draw(image)

        # Calculate the center coordinates of the image
        image_center_x, image_center_y = image.size[0] // 2, image.size[1] // 2

        # Convert ratios back to pixel values
        room_center_x = int(room_info['x_ratio'] / 100 * image.size[0])
        room_center_y = int(room_info['y_ratio'] / 100 * image.size[1])
        room_width = int(room_info['w_ratio'] / 100 * image.size[0])
        room_height = int(room_info['h_ratio'] / 100 * image.size[1])

        # Enlarge the mark box
        x_expand = room_width * 0.3
        y_expand = room_height * 0.3
        x, y, w, h = room_center_x - x_expand, room_center_y - y_expand, room_width + 2 * x_expand, room_height + 2 * y_expand
        draw.rectangle(((x, y), (x + w, y + h)), outline="yellow", width=7)
        draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=3)

        # Save the highlighted image
        image.save(highlighted_image_path)

        # Check for additional images
        room_image_path = os.path.join("image/room", f"{room_id}.jpg")
        skyview_image_path = os.path.join("image/skyview", f"{building_id}.jpg")

        room_image = None
        skyview_image = None

        if os.path.exists(room_image_path):
            room_image = Image.open(room_image_path)

        if os.path.exists(skyview_image_path):
            skyview_image = Image.open(skyview_image_path)
            new_height = (2 * image.size[1]) // 3  # Adjust skyview_image to 2/3 height of the main image
            aspect_ratio = skyview_image.size[0] / skyview_image.size[1]
            new_width = int(new_height * aspect_ratio)
            skyview_image = skyview_image.resize((new_width, new_height))

        if room_image:
            # Adjust room_image to match skyview_image width
            new_width = skyview_image.size[0] if skyview_image else image.size[0] // 3
            aspect_ratio = room_image.size[1] / room_image.size[0]
            new_height = int(new_width * aspect_ratio)
            room_image = room_image.resize((new_width, new_height))

        # Combine images
        combined_width = image.size[0] + (skyview_image.size[0] if skyview_image else 0)
        combined_image_height = max(image.size[1], (room_image.size[1] if room_image else 0) + (skyview_image.size[1] if skyview_image else 0))
        combined_image = Image.new('RGB', (combined_width, combined_image_height))

        # Paste main image
        combined_image.paste(image, (0, 0))

        # Paste skyview image at the top right
        if skyview_image:
            if room_image:
                combined_image.paste(skyview_image, (image.size[0], 0))
            else:
                # Calculate y offset to center skyview_image vertically relative to image
                y_offset = (image.size[1] - skyview_image.size[1]) // 2
                # Paste skyview image at the right of the main image, centered vertically
                combined_image.paste(skyview_image, (image.size[0], y_offset))

        # Paste room image below skyview image and center vertically
        if room_image:
            y_offset = skyview_image.size[1] if skyview_image else 0
            remaining_space = combined_image_height - y_offset - room_image.size[1]
            centered_y = y_offset + (remaining_space // 2)
            combined_image.paste(room_image, (image.size[0], centered_y))

        # Save the combined image
        combined_image.save(highlighted_image_path)

        return highlighted_image_path, image.size
    else:
        return None, None




def combine_images(tmp_directory, image_name_postfix, image_name_prefix=None):
    """
    Combines multiple images matching the given patterns into a single image.
    :param tmp_directory: The directory where the images are stored.
    :param image_name_postfix: The primary pattern to match for image filenames.
    :param image_name_prefix: The secondary pattern to match for image filenames (optional).
    :return: Combined image object.
    """

    def sort_key(filename):
        match = re.match(r"(\d+)-(.+)", filename)
        if match:
            return (int(match.group(1)), match.group(2))
        return filename

    if image_name_prefix:
        image_files = sorted(
            [f for f in os.listdir(tmp_directory) if image_name_postfix in f and f.startswith(f"{image_name_prefix}-")],
            key=sort_key
        )
    else:
        image_files = sorted(
            [f for f in os.listdir(tmp_directory) if image_name_postfix in f],
            key=sort_key
        )
    # print("Sorted image files:", image_files)
    images = [Image.open(os.path.join(tmp_directory, img_file)) for img_file in image_files]

    # Assuming all images are of the same size for simplicity
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new image to combine all images
    combined_image = Image.new('RGB', (total_width, max_height))

    # Paste each image next to each other
    x_offset = 0
    for im in images:
        combined_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return combined_image

@app.route('/validate', methods=['GET'])
def validate_images():
    """
    combine multiple images into one and return the combined image.
    If a 'test' query parameter is provided, it reads room numbers from 'test.txt',
    generates images for each room number using view_room_highlighted, and combines these images.
    The 'force' query parameter can be used to force the regeneration of the combined image.

    - 'test=true': Generates and combines images for room numbers read from 'test.txt'.
    - 'target=<prefix>': Combines images with filenames starting with the given prefix.
    - 'force=true': Ignores any cached combined image and forces the creation and combination of new images.
    -  Without 'force' parameter or with 'force=false': If a previously generated combined image exists, it's returned.   

    :return: Response with the combined image file or an error message.
    """
    test = request.args.get('test')
    force = request.args.get('force')
    target = request.args.get('target')


    # If the test parameter is provided, perform validation on the test
    if test:
        combined_image_path = os.path.join(tmp_directory, 'combined_test_image.png')

        if not force:
            if os.path.exists(combined_image_path):
                return send_file(combined_image_path, mimetype='image/png')

        combined_images = []
        with open('testset.txt', 'r') as file:
            room_ids = file.read().splitlines()

        for room_id in room_ids:
            # Call the view_room_highlighted function and get the image path
            # We're calling the function directly, but it's better to refactor this logic into a common function
            # that both view_room_highlighted and this route can use.
            image_path, _ = search_and_highlight(room_id)
            if os.path.exists(image_path):
                combined_images.append(Image.open(image_path))

        # Combine all images into one
        if combined_images:
            # Assuming all images are the same size
            widths, heights = zip(*(i.size for i in combined_images))
            total_width = sum(widths)
            max_height = max(heights)
            combined_image = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for im in combined_images:
                combined_image.paste(im, (x_offset, 0))
                x_offset += im.size[0]

            
            combined_image.save(combined_image_path)
            return send_file(combined_image_path, mimetype='image/png')
        else:
            return jsonify({"error": "No images were generated from the test"}), 404
    else:
        combined_image_path = os.path.join(tmp_directory, 'combined_image.png')

        try:
            if target:
                combined_image_path = os.path.join(tmp_directory, f'{target}-combined_image.png')
                combined_image = combine_images(tmp_directory, '-highlighted_image.png', target)
            else:
                if not force:
                    if os.path.exists(combined_image_path):
                        return send_file(combined_image_path, mimetype='image/png')
                combined_image = combine_images(tmp_directory, '-highlighted_image.png')
        
            combined_image.save(combined_image_path)

            return send_file(combined_image_path, mimetype='image/png')
        except ValueError as e:
            return jsonify({"error": f"Cannot validate the requested building floors: {str(e)}"}), 404


analyzed_results = {}
image_names = []

def perform_image_analysis():
    # Handle tmp directory
    if os.path.exists(tmp_directory):
        for file in os.listdir(tmp_directory):
            os.remove(os.path.join(tmp_directory, file))

    results = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image_name = os.path.splitext(filename)[0]
            image_results = analyze_image(image_path, image_name)
            results.update(image_results)
    return results

# main
if __name__ == '__main__':
    json_file = 'map.json'

    # Check if the map.json file exists
    if os.path.exists(json_file):
        # Prompt the user to decide whether to perform new image analysis or use existing data
        user_input = input("The map.json file already exists. Do you want to proceed with new image analysis? (y/n): ")

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_name = os.path.splitext(filename)[0]
                image_names.append(image_name)

        if user_input.lower() != 'y':
            # If the user chooses not to perform new analysis, use the existing map.json file
            print("Using existing map.json file to run the server.")
            analyzed_results = load_results_from_json  # Load the analyzed results from the existing file
        else:
            # If the user chooses to perform new analysis
            analyzed_results = perform_image_analysis()  # Perform image analysis to extract room data
            save_results_to_json(analyzed_results, json_file)  # Save the new analyzed results to map.json
    else:
        # If the map.json file does not exist, perform new image analysis
        analyzed_results = perform_image_analysis()  # Perform image analysis to extract room data
        save_results_to_json(analyzed_results, json_file)  # Save the analyzed results to a new map.json file

    # Start the Flask server
    # app.run(debug=False, host='0.0.0.0', port=5000)  
    app.run(debug=False, host='0.0.0.0', port=80)  
