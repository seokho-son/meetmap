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

# Load alias.json once at the start of the program
def load_alias_data():
    """
    Loads alias data from alias.json, handling different encodings.
    :return: Parsed alias data as a dictionary, or an empty dictionary if loading fails.
    """
    alias_file = 'alias.json'  # Define the file path
    if not os.path.exists(alias_file):
        print("alias.json file not found.")
        return {}
    
    alias_data = load_json_file(alias_file)  # Use the helper function
    if alias_data:
        return alias_data
    else:
        print("Failed to load alias.json or file is empty.")
        return {}

alias_data = load_alias_data()

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
                        store_room_data(start.strip(), data, i, analyzed_results, image_name)

                        # Adjust left and width for the end value and store
                        data['left'][i] = original_left + half_width
                        data['width'][i] = half_width
                        store_room_data(end.strip(), data, i, analyzed_results, image_name)
                    elif ',' in match[0]:
                        # Split at ',' and save each part
                        split_patterns = match[0].split(',')
                        for room in split_patterns:
                            store_room_data(room.strip(), data, i, analyzed_results, image_name)
                    elif '-' in match[0] and len(match[0].split('-')[0]) == len(match[0].split('-')[1]):
                        # Split at '-' and save both parts if they have the same length
                        start, end = match[0].split('-')
                        store_room_data(start.strip(), data, i, analyzed_results, image_name)
                        store_room_data(end.strip(), data, i, analyzed_results, image_name)        
                    else:
                        store_room_data(match[0], data, i, analyzed_results, image_name)

    for text, coords in analyzed_results.items():
        draw.rectangle(((coords["x"], coords["y"]), (coords["x"] + coords["w"], coords["y"] + coords["h"])), outline="red")
        # draw.text((coords["x"], coords["y"] + coords["h"]), text, fill="green", font=font)

    # Saving the final image with highlighted text
    resized_image.save(os.path.join(tmp_directory, f'{image_name}-7-highlighted_image.png'))
    return analyzed_results


# Function to store data about a room
def store_room_data(room_number, data, index, analyzed_results, image_name):
    """
    Stores the room number data along with its coordinates.
    :param room_number: The identified room number from the image.
    :param data: The data dictionary obtained from pytesseract OCR.
    :param index: The current index in the OCR data list.
    :param analyzed_results: The dictionary where results are being stored.
    :param image_name: The name of the image being analyzed, used for labeling.
    """
    # Replace the first character of room_number with image_name for unique identification
    room_number = image_name + room_number[1:]

    # Extract coordinates and dimensions of the detected text
    (x, y, width, height) = (data['left'][index], data['top'][index], data['width'][index], data['height'][index])
    print(f"- {room_number}: Location: X: {x}, Y: {y}, Width: {width}, Height: {height}")
    
    # Store the room information in the analyzed_results dictionary
    analyzed_results[room_number] = {
        "floor": image_name, "x": x, "y": y, "w": width, "h": height
    }

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

def calculate_room_similarity(requested_room, available_rooms):
    """
    Calculate similarity score between the requested room and available rooms.
    :param requested_room: The requested room number as a string.
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

    similarities = [(room, similarity_score(requested_room, room)) for room in available_rooms]
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
def list_room_numbers():
    """
    List all available room numbers.
    :return: JSON response with a list of all room numbers.
    """
    analyzed_results = load_results_from_json()
    sorted_room_numbers = sorted(analyzed_results.keys(), key=sort_key)
    print(f"Total Room Numbers: {len(sorted_room_numbers)}")
    print(sorted_room_numbers)
    response = {
        "Total room numbers": len(sorted_room_numbers),
        "All identified Rooms": sorted_room_numbers
    }
    return Response(
        json.dumps(response, ensure_ascii=False),  # Ensure UTF-8 encoding
        content_type="application/json; charset=utf-8",
        status=200
    )


@app.route('/alias', methods=['GET'])
def get_alias():
    """
    Reload alias.json and return the updated list.
    """
    global alias_data
    try:
        alias_data = load_alias_data()
        response = json.dumps(alias_data, ensure_ascii=False)
        return Response(response, content_type='application/json; charset=utf-8')
    except FileNotFoundError:
        return jsonify({"error": "alias.json file not found."}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding alias.json file."}), 500
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
#     room_number = data.get('room_number')

#     if room_number and room_number not in analyzed_results:
#         analyzed_results[room_number] = data
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

@app.route('/room/<room_number>', methods=['GET'])
@app.route('/view/<room_number>', methods=['GET'])
@app.route('/find/<room_number>', methods=['GET'])
@app.route('/dir/<room_number>', methods=['GET'])
@app.route('/r/<room_number>', methods=['GET'])
@app.route('/v/<room_number>', methods=['GET'])
@app.route('/f/<room_number>', methods=['GET'])
@app.route('/d/<room_number>', methods=['GET'])
def view_room_highlighted(room_number):
    """
    View an image with a specific room number highlighted.
    This endpoint sends an image with the specified room number highlighted, indicating its location.
    :param room_number: The room number to be highlighted in the image.
    :return: HTML response with the image or file response based on returnType parameter.
    """

    try:
        # Replace multiple consecutive "-" with a single "-"
        room_number = re.sub(r'-+', '-', room_number)

        # Normalize room_number by removing spaces
        normalized_room_number = room_number.replace(" ", "")
        
        global alias_data
        alias_data = load_alias_data()

        # Check alias_data for a matching key
        for alias, replacement in alias_data.items():
            normalized_alias = alias.replace(" ", "")
            # print(f"Normalized Alias: {normalized_alias}")
            if normalized_alias in normalized_room_number:
                room_number = re.sub(re.escape(normalized_alias), replacement, normalized_room_number, flags=re.IGNORECASE)
                break    
    
        # Define the patterns to be replaced
        patterns_to_replace = [" 동 ", "동-", "동 ", " 동", "동", " - ", "- ", " -", " "]
        
        # Replace the patterns with "-"
        for pattern in patterns_to_replace:
            room_number = room_number.replace(pattern, "-")
        
        # Remove "호" from the room number
        room_number = room_number.replace("호", "")

        # Replace multiple consecutive "-" with a single "-"
        room_number = re.sub(r'-+', '-', room_number)

        room_number = room_number.upper()

        print(f"Requested Room Number: {room_number}")

        analyzed_results = load_results_from_json()

        # Check if room_number starts with any of the image names or if it exists in analyzed_results
        if room_number not in analyzed_results and not any(room_number.startswith(name.upper()) for name in image_names):
            response = {
                "Message": f"Map for {room_number} is not supported yet.",
                "Supported Maps": sorted(image_names, key=sort_key)
            }
            return Response(
                json.dumps(response, ensure_ascii=False),  # Ensure non-ASCII characters are preserved
                content_type="application/json; charset=utf-8",
                status=404
            )
        force = request.args.get('force', '').lower() == 'true'
        highlighted_image_path = view_room_highlighted_logic(room_number, force)

        if not os.path.exists(highlighted_image_path):
            return jsonify({"error": "Room number not found or image could not be created"}), 404

        return_type = request.args.get('returnType', '').lower()

        if return_type == "file":
            # Return image file directly
            return send_file(highlighted_image_path, mimetype='image/png')
        else:
            # Return HTML with image embedded and styled
            html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Find Room: {room_number}</title>
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
                    img {{
                        max-width: 100%;
                        max-height: 100%;
                        object-fit: contain;
                        transition: all 0.3s ease-in-out;
                    }}
                    .room-number {{
                        position: absolute;
                        bottom: 10px;
                        left: 10px;
                        color: white;
                        background-color: rgba(0, 0, 0, 0.7);
                        padding: 5px 10px;
                        border-radius: 5px;
                        font-family: Arial, sans-serif;
                        font-size: 14px;
                    }}
                    .button-container {{
                        position: absolute;
                        bottom: 10px;
                        right: 10px;
                    }}
                    button {{
                        padding: 10px 15px;
                        margin: 5px;
                        font-size: 14px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        background-color: #007BFF;
                        color: white;
                    }}
                    button:hover {{
                        background-color: #0056b3;
                    }}
                </style>
            </head>
            <body>
                <div class="button-container">
                    <button id="toggleSizeButton" onclick="toggleImageSize()">+</button>
                </div>
                <img id="highlightedImage" src="data:image/png;base64,{convert_image_to_base64(highlighted_image_path)}" alt="Highlighted Room: {room_number}">
                <div class="room-number">Requested Room: {room_number}</div>
                <script>
                    let isOriginalSize = false;

                    function toggleImageSize() {{
                        const img = document.getElementById('highlightedImage');
                        const button = document.getElementById('toggleSizeButton');

                        if (isOriginalSize) {{
                            img.style.maxWidth = "100%";
                            img.style.maxHeight = "100%";
                            img.style.width = "auto";
                            img.style.height = "auto";
                            button.textContent = "+";
                        }} else {{
                            img.style.maxWidth = "none";
                            img.style.maxHeight = "none";
                            img.style.width = "auto";
                            img.style.height = "auto";
                            button.textContent = "fit";
                        }}

                        isOriginalSize = !isOriginalSize;
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



def view_room_highlighted_logic(room_number, force=False):
    """
    Logic to generate an image with a specific room number highlighted.
    :param room_number: The room number to be highlighted in the image.
    :param force: Boolean flag to force regeneration of the image.
    :return: Path to the generated image, or None if not found.
    """

    highlighted_image_path = os.path.join(tmp_directory, f"{room_number}-highlighted.png")

    # Check if the image already exists and if force regeneration is not required
    if not force and os.path.exists(highlighted_image_path):
        return highlighted_image_path

    analyzed_results = load_results_from_json()

    # Logic to generate a new highlighted image
    if room_number in analyzed_results:
        room_info = analyzed_results[room_number]
        similar_room = None  # Exact match found
    else:
        similar_room = calculate_room_similarity(room_number, analyzed_results.keys())
        if not similar_room or similar_room == room_number:  # Low similarity or no match found
            return None
        room_info = analyzed_results[similar_room]

    # Load the floor image and prepare for drawing
    floor_image_path = os.path.join(tmp_directory, f"{room_info['floor']}-map.png")
    if os.path.exists(floor_image_path):
        image = Image.open(floor_image_path)
        draw = ImageDraw.Draw(image)

        # Calculate the center coordinates of the image
        image_center_x, image_center_y = image.size[0] // 2, image.size[1] // 2

        # Center of mark box
        room_center_x = room_info['x'] + room_info['w'] // 2
        room_center_y = room_info['y'] + room_info['h'] // 2

        # Enlarge the mark box
        x_expand = room_info['w'] * 0.3
        y_expand = room_info['h'] * 0.3
        x, y, w, h = room_info['x'] - x_expand, room_info['y'] - y_expand, room_info['w'] + 2 * x_expand, room_info['h'] + 2 * y_expand
        draw.rectangle(((x, y), (x + w, y + h)), outline="yellow", width=7)
        draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=3)

        # Draw arrow and text
        path = calculate_arrow_path(image_center_x, image_center_y, x, y, w, h)
        for i in range(len(path) - 1):
            draw_dashed_line(draw, path[i], path[i + 1], width=8, interval=12, fill="#32CD32")

        draw_arrow_head(draw, path[-2], path[-1], arrow_size=40, fill="#32CD32")
        message1 = f":{room_number}"
        if similar_room:
            message1 = f"? {room_number}"

        mark_pos = (path[-1][0], path[-1][1])
        draw_label(draw, message1 , mark_pos)

        # Extract building number from room_number
        building_number = room_number.split('-')[0]

        # Check for additional images
        room_image_path = os.path.join("image/room", f"{room_number}.jpg")
        skyview_image_path = os.path.join("image/skyview", f"{building_number}.jpg")

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

        return highlighted_image_path
    else:
        return None




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
            room_numbers = file.read().splitlines()

        for room_number in room_numbers:
            # Call the view_room_highlighted function and get the image path
            # We're calling the function directly, but it's better to refactor this logic into a common function
            # that both view_room_highlighted and this route can use.
            image_path = view_room_highlighted_logic(room_number)
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

