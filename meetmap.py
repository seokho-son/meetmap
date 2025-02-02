# Importing necessary libraries
import sys
import numpy as np
import re
import os
import json
import cv2  # OpenCV library for computer vision tasks
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from flask import Flask, jsonify, redirect, url_for, request, send_file, Response
import easyocr  # EasyOCR library for text recognition

# Initializing Flask application
app = Flask(__name__)

# Initialize EasyOCR Reader (supports multiple languages, e.g., ['en', 'ko'])
reader = easyocr.Reader(['en'], gpu=True)  # Use GPU if available

# NanumSquareL.ttf Font License - https://help.naver.com/service/30016/contents/18088?osType=PC&lang=ko
font_path = "assets/NanumSquareL.ttf"
# Copyright (c) 2010, NAVER Corporation (https://www.navercorp.com/) with Reserved Font Name Nanum, 
# Naver Nanum, NanumGothic, Naver NanumGothic, NanumMyeongjo, Naver NanumMyeongjo, NanumBrush, 
# Naver NanumBrush, NanumPen, Naver NanumPen, Naver NanumGothicEco, NanumGothicEco, 
# Naver NanumMyeongjoEco, NanumMyeongjoEco, Naver NanumGothicLight, NanumGothicLight, 
# NanumBarunGothic, Naver NanumBarunGothic, NanumSquareRound, NanumBarunPen, MaruBuri, NanumSquareNeo
# This Font Software is licensed under the SIL Open Font License, Version 1.1.
# This license is copied below, and is also available with a FAQ at: http://scripts.sil.org/OFL
# SIL OPEN FONT LICENSE
# Version 1.1 - 26 February 2007 

# Handle directory
tmp_directory = "tmp"
if not os.path.exists(tmp_directory):
    os.makedirs(tmp_directory)

directory_path = "image/map"
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
    image.save(os.path.join(tmp_directory, f'{image_name}-0-image_after_enhance.png'))

    # Resizing the image
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    new_height = 1400
    new_width = int(aspect_ratio * new_height)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_image.save(os.path.join(tmp_directory, f'{image_name}-map.png'))
    print("Image Information:")
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
    preprocessed_path = os.path.join(tmp_directory, f'{image_name}-0-image_after_enhance.png')
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
        
        # Run EasyOCR
        results = reader.readtext(image_np, allowlist ='0123456789LGB-', detail=1, paragraph=False)  # detail=1 includes bounding boxes and confidence

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
    font = ImageFont.truetype(font_path, font_size)
    font_size = 10
    fontSmall = ImageFont.truetype(font_path, font_size)

    # Parsing the extracted text and coordinates
    # Drawing rectangles around detected text on the image
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0.4:  # Using low confidence texts as well
            text = data['text'][i].strip()
            draw.text((data['left'][i], data['top'][i] + data['height'][i]), f"{text}/{data['conf'][i]}%", fill="green", font=font)
            if text:
                print(f"{text} (Confidence: {data['conf'][i]}%)")
                # Extract room number patterns
                matches = re.findall(r'\b((L|G|B)?\d{2,3}(-\d+)?(~\d+)?)\b', text)
                for match in matches:
                    # Check for range or list patterns
                    if '~' in match[0]:
                        # Split at '~' and save both parts
                        start, end = match[0].split('~')
                        half_width = data['width'][i] // 2
                        original_left = data['left'][i]  # Save the original 'left' value

                        # Adjust width for the start value and store
                        data['width'][i] = half_width
                        store_room_data(start.strip(), data, i, analyzed_results, image_name, resized_image.size)

                        # Adjust left and width for the end value and store
                        data['left'][i] = original_left + half_width
                        data['width'][i] = half_width
                        store_room_data(end.strip(), data, i, analyzed_results, image_name, resized_image.size)
                    elif ',' in match[0]:
                        # Split at ',' and save each part
                        split_patterns = match[0].split(',')
                        for room in split_patterns:
                            store_room_data(room.strip(), data, i, analyzed_results, image_name, resized_image.size)
                    elif '-' in match[0] and len(match[0].split('-')[0]) == len(match[0].split('-')[1]):
                        # Split at '-' and save both parts if they have the same length
                        start, end = match[0].split('-')
                        store_room_data(start.strip(), data, i, analyzed_results, image_name, resized_image.size)
                        store_room_data(end.strip(), data, i, analyzed_results, image_name, resized_image.size)
                    else:
                        store_room_data(match[0], data, i, analyzed_results, image_name, resized_image.size)

    for text, coords in analyzed_results.items():
        # Convert ratios back to pixel values
        x = int(coords["x_ratio"] / 100 * resized_image.size[0])
        y = int(coords["y_ratio"] / 100 * resized_image.size[1])
        w = int(coords["w_ratio"] / 100 * resized_image.size[0])
        h = int(coords["h_ratio"] / 100 * resized_image.size[1])
        print(f"- {text}: Location: X: {x}, Y: {y}, Width: {w}, Height: {h}")
        draw.rectangle(((x, y), (x + w, y + h)), outline="red")

    # Saving the final image with highlighted text
    resized_image.save(os.path.join(tmp_directory, f'{image_name}-7-highlighted_image.png'))
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
    # Replace the first character of room_id with image_name for unique identification
    room_id = image_name + room_id[1:]

    # Extract coordinates and dimensions of the detected text
    (x, y, width, height) = (data['left'][index], data['top'][index], data['width'][index], data['height'][index])
    print(f"- {room_id}: Location: X: {x}, Y: {y}, Width: {width}, Height: {height}")

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

    font = ImageFont.truetype(font_path, font_size)

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
def list_apis():
    """
    list all available API endpoints.
    :return: JSON response with information about all routes.
    """
    routes = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        endpoint = rule.endpoint
        url = str(rule)
        doc = app.view_functions[endpoint].__doc__ if app.view_functions[endpoint].__doc__ else "No description available"
        doc = doc.strip().replace('\n', ' ')
        routes.append({
            'endpoint': endpoint,
            'methods': methods,
            'url': url,
            'description': doc
        })
    return jsonify(routes)

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
    :param request_id: The room number to be highlighted in the image.
    :return: HTML response with the image or file response based on returnType parameter.
    """

    try:
        x_param = request.args.get('x')
        y_param = request.args.get('y')
        note_param = request.args.get('note')

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

        request_id = request_id.upper()

        isFloorRequest = False
        if request_id in image_names:
            isFloorRequest = True

        print(f"Original Request: {org_request_id} Adjusted Request: {request_id} (isFloorRequest: {isFloorRequest})") 

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

        if isFloorRequest:
            floor_id = request_id
            building_id, floor_only_id = split_floor_id(floor_id)

        print(f"floor_id: {floor_id}, building_id: {building_id}, floor_only_id: {floor_only_id}")
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
        
        return_type = request.args.get('returnType', '').lower()

        destinationLabelText = f"{request_id} ?" if similar_room else request_id
        buildingText = f"{b_name}" if b_name else f"{building_id}동" 
        noteText = f"{note_param}" if note_param else ""


        if return_type == "file":
            # Return image file directly
            return send_file(floor_image_path, mimetype='image/png')
        else:
            # Use the Base64 encoded images in HTML
            html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Map:{request_id}</title>
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
                        max-width: 100%;
                        max-height: 100%;
                        overflow: hidden;
                    }}
                    img {{
                        max-width: 100%;
                        max-height: 100%;
                        object-fit: contain;
                        transition: all 0.3s ease-in-out;
                    }}
                    .floor-identifier {{
                        position: absolute;
                        top: 1%;
                        left: 1%;
                        color: white;
                        background-color: rgba(0, 0, 0, 0.7);
                        padding: 0.5% 1%; 
                        border-radius: 0.5vw;
                        font-family: Arial, sans-serif;
                        font-size: 2.1vw; 
                        cursor: pointer;
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
                        font-family: Arial, sans-serif;
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
                        font-family: Arial, sans-serif;
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
                        font-family: Arial, sans-serif;
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
                        font-family: Arial, sans-serif;
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
                        0% {{ background-color: rgba(0, 0, 0, 0.5); color: rgba(255, 255, 255, 1); }}
                        50% {{ background-color: rgba(0, 0, 0, 0.1); color: rgba(255, 255, 255, 0.1); }}
                        100% {{ background-color: rgba(0, 0, 0, 0.5); color: rgba(255, 255, 255, 1); }}
                    }}                        
                </style>
            </head>
            <body>
                <div class="image-container">
                    <img id="floorImage" src="data:image/png;base64,{convert_image_to_base64(floor_image_path)}" alt="Request: {org_request_id} ({request_id})" />
                    <div id="northLabel" class="gate-label" style="display: none;">북쪽</div>
                    <div id="mainGateLabel" class="gate-label" style="display: none;">주출입구방향</div>                                      
                    <div id="sourceBox" class="source-box" style="display: none;"></div>
                    <div id="destinationBox" class="destination-box" style="display: none;"></div>
                    <div id="destinationLabel" class="box-label" style="display: none;">{destinationLabelText}</div>
                    <div id="sourceLabel" class="box-label" style="display: none;">X / Y</div>
                    <div class="button-container">
                        {"<button id='toggleSkyviewButton' onclick='toggleSkyview()'>sky</button>" if skyview_image_base64 else ""}
                        {"<button id='toggleRoomviewButton' onclick='toggleRoomview()'>room</button>" if room_image_base64 else ""}
                    </div>
                    <img id="skyviewImage" src="data:image/png;base64,{skyview_image_base64}" style="display: none; position: absolute; top: 50%; left: 75%; transform: translate(-50%, -50%); max-width: 45%; max-height: 80%;" />
                    <img id="roomviewImage" src="data:image/png;base64,{room_image_base64}" style="display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); max-width: 80%; max-height: 80%;" />
                    <div class="floor-identifier">{buildingText} {floor_only_id}층</div>
                    <div class="mouse-position" id="mousePosition">X: 0 / Y: 0</div>
                    <div class="author-label">© Ph.D. Seokho Son</div>
                </div>
            
                <script>

                    const isFloorRequest = {str(isFloorRequest).lower()};
                    const bNorthX = {b_north_x if b_north_x else 'null'};
                    const bNorthY = {b_north_y if b_north_y else 'null'};
                    const bMainGateX = {b_main_gate_x if b_main_gate_x else 'null'};
                    const bMainGateY = {b_main_gate_y if b_main_gate_y else 'null'};
                    const xParam = {x_param if x_param is not None else 'null'};
                    const yParam = {y_param if y_param is not None else 'null'};

                    function toggleSkyview() {{
                        const skyviewImage = document.getElementById('skyviewImage');
                        const roomviewImage = document.getElementById('roomviewImage');
                        if (skyviewImage.style.display === 'none') {{
                            skyviewImage.style.display = 'block';
                            roomviewImage.style.display = 'none';
                        }} else {{
                            skyviewImage.style.display = 'none';
                        }}
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
                        sourceBox.style.height = getComputedStyle(sourceBox).width;
                        sourceBox.style.borderRadius = '50%';
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
                        let labelContent = `{org_request_id}<br>({destinationLabelText})`;
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
                        sourceBox.style.height = getComputedStyle(sourceBox).width;
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
                        const sourceBox = document.getElementById('sourceBox');
                        
                        const sourceLabel = document.getElementById('sourceLabel');
                        const floorIdentifier = document.querySelector('.floor-identifier');
                        const mousePosition = document.querySelector('.mouse-position');
                        const toggleSkyviewButton = document.getElementById('toggleSkyviewButton');
                        const toggleRoomviewButton = document.getElementById('toggleRoomviewButton');

                        const boxSize = 4;
                        const fontSize = 1.6;
                        const padding = 0.5;
                        const borderThickness = 0.4;

                        sourceBox.style.width = `${{boxSize}}%`;
                        sourceBox.style.height = getComputedStyle(sourceBox).width;
                        sourceBox.style.borderWidth = `${{borderThickness}}vw`;
                        
                        floorIdentifier.style.fontSize = `${{fontSize + 0.5}}vw`;
                        floorIdentifier.style.padding = `${{padding}}% 1%`;
                        mousePosition.style.fontSize = `${{fontSize - 0.2}}vw`;
                        sourceLabel.style.fontSize = `${{fontSize}}vw`;
                        sourceLabel.style.padding = `${{padding}}% 1%`;
                        
                        if (!isFloorRequest) {{
                            const destinationBox = document.getElementById('destinationBox');
                            const destinationLabel = document.getElementById('destinationLabel');
                            destinationBox.style.borderWidth = `${{borderThickness}}vw`;
                            destinationLabel.style.fontSize = `${{fontSize}}vw`;
                            destinationLabel.style.padding = `${{padding}}% 1%`;                        
                        }}

                        if (toggleSkyviewButton) {{
                            toggleSkyviewButton.style.fontSize = `${{fontSize}}vw`;
                            toggleSkyviewButton.style.padding = `${{padding}}% 2%`;
                        }}

                        if (toggleRoomviewButton) {{
                            toggleRoomviewButton.style.fontSize = `${{fontSize}}vw`;
                            toggleRoomviewButton.style.padding = `${{padding}}% 2%`;
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
                    floorIdentifier.addEventListener('mouseover', function() {{
                        floorIdentifier.textContent = 'Copy URL';
                    }});

                    floorIdentifier.addEventListener('mouseout', function() {{
                        floorIdentifier.textContent = '{buildingText} {floor_only_id}층';
                    }});

                    floorIdentifier.addEventListener('click', function() {{
                        const url = decodeURIComponent(window.location.href);
                        if (navigator.clipboard) {{
                            navigator.clipboard.writeText(url).then(function() {{
                                alert(`Copied following URL to clipboard.\\n\\n${{url}}`);
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
                                alert(`Copied following URL to clipboard.\\n\\n${{url}}`);
                            }} catch (err) {{
                                console.error('Could not copy text: ', err);
                            }}
                            document.body.removeChild(textArea);
                        }}
                    }});

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




def combine_images(tmp_directory, image_pattern):
    """
    Combines multiple images matching the given pattern into a single image.
    :param tmp_directory: The directory where the images are stored.
    :param image_pattern: The pattern to match for image filenames.
    :return: Combined image object.
    """

    def sort_key(filename):
        match = re.match(r"(\d+)-(.+)", filename)
        if match:
            return (int(match.group(1)), match.group(2))
        return filename

    image_files = sorted([f for f in os.listdir(tmp_directory) if image_pattern in f], key=sort_key)
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
    - 'force=true': Ignores any cached combined image and forces the creation and combination of new images.
    -  Without 'force' parameter or with 'force=false': If a previously generated combined image exists, it's returned.   

    :return: Response with the combined image file or an error message.
    """
    test = request.args.get('test')
    force = request.args.get('force')


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

        if not force:
            if os.path.exists(combined_image_path):
                return send_file(combined_image_path, mimetype='image/png')

        # Original behavior for combining highlighted images
        combined_image = combine_images(tmp_directory, '-7-highlighted_image.png')
        combined_image.save(combined_image_path)

        return send_file(combined_image_path, mimetype='image/png')


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
        # Run the Flask app with debug mode on, accessible on all interfaces on port 80
    app.run(debug=True, host='0.0.0.0', port=80)  

