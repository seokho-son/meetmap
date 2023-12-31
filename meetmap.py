# Importing necessary libraries
import sys
import numpy as np
import re
import os
import json
import pytesseract  # Tesseract OCR library for text recognition
import cv2  # OpenCV library for computer vision tasks
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from flask import Flask, jsonify, request, send_file

# Initializing Flask application
app = Flask(__name__)

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
    enhanced_image = enhancer.enhance(1.01)  # Increasing contrast by 1.01 times

    # Increasing image sharpness
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer.enhance(1.0)  # Increasing sharpness, value greater than 1.0 makes the image sharper
    enhanced_image.save(os.path.join(tmp_directory, f'{image_name}-0-image_after_enhance.png'))

    # Resizing the image
    original_width, original_height = enhanced_image.size
    aspect_ratio = original_width / original_height
    new_height = 1000
    new_width = int(aspect_ratio * new_height)
    resized_image = enhanced_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print("Image Information:")
    print(f"Original Size: {original_width}x{original_height}")
    print(f"Resized Size: {new_width}x{new_height}")

    # Converting the image to a numpy array for processing
    image_data = np.array(resized_image)

    # Creating masks for red and blue channel adjustments
    red_mask = image_data[:, :, 0] > 90
    blue_mask = image_data[:, :, 2] > 90

    # Applying white color to specified pixels
    image_data[red_mask | blue_mask] = [255, 255, 255]

    # Saving the modified image
    image_after_red = Image.fromarray(image_data)
    image_after_red.save(os.path.join(tmp_directory, f'{image_name}-1-image_after_red.png'))

    # Image for map with enhanced contrast
    enhancer = ImageEnhance.Contrast(image_after_red)
    enhanced_image = enhancer.enhance(1.5) 
    enhanced_image.save(os.path.join(tmp_directory, f'{image_name}-map.png'))

    # Changing all pixels with a green channel below 20 to white
    green_threshold = 20
    green_mask = image_data[:, :, 1] < green_threshold
    image_data[green_mask] = [255, 255, 255]

    # Saving the modified image
    image_after_green = Image.fromarray(image_data)
    image_after_green.save(os.path.join(tmp_directory, f'{image_name}-2-image_after_green.png'))

    # Changing pixels to white if the deviation among RGB channels is less than 14
    max_deviation = np.max(image_data, axis=-1) - np.min(image_data, axis=-1)
    uniform_color_mask = max_deviation < 14
    image_data[uniform_color_mask] = [255, 255, 255]

    # Saving the modified image
    image_rgb_threshold = Image.fromarray(image_data)
    image_rgb_threshold.save(os.path.join(tmp_directory, f'{image_name}-3-image_after_rgb_threshold.png'))

    # Enhancing contrast for better text recognition
    enhancer = ImageEnhance.Contrast(image_rgb_threshold)
    image_rgb_threshold = enhancer.enhance(1.5)

    # Preprocessing the image for OCR
    # Setting a threshold value for binarization
    threshold_value = 100

    # Converting the image to grayscale
    gray_image = image_rgb_threshold.convert('L')
    gray_image.save(os.path.join(tmp_directory, f'{image_name}-4-gray_image.png'))

    # Binarizing the image
    _, binary_image = cv2.threshold(np.array(gray_image), threshold_value, 255, cv2.THRESH_BINARY)
    binary_image_save = Image.fromarray(binary_image)
    binary_image_save.save(os.path.join(tmp_directory, f'{image_name}-5-binary_image.png'))

    # Defining a sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  30, -1],
                                  [-1, -1, -1]])
    edges = cv2.Canny(binary_image, 100, 200)
    mask = edges != 0
    mask = mask.astype(np.uint8)

    # Applying sharpening to the image
    sharpened_image = cv2.filter2D(binary_image, -1, sharpening_kernel)
    binary_image[mask == 1] = sharpened_image[mask == 1]

    # Creating erosion and dilation to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    # Convert back to PIL Image for OCR
    image_for_ocr = Image.fromarray(dilated_image)
    image_for_ocr.save(os.path.join(tmp_directory, f'{image_name}-6-dilated_eroded_image.png'))

    # Using Tesseract OCR to extract text and their coordinates from the image
    # Function to run OCR on multiple page segmentation modes and combine results
    def run_ocr_on_all_modes(image, modes):
        combined_data = {'text': [], 'conf': [], 'left': [], 'top': [], 'width': [], 'height': []}
        seen = set()  # Set to track seen text and their coordinates

        for mode in modes:
            try:
                config = f'--psm {mode}'
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

                for i in range(len(data['text'])):
                    # Create a unique identifier for each text based on its content and position
                    identifier = (data['text'][i], data['left'][i], data['top'][i], data['width'][i], data['height'][i])

                    if identifier not in seen:
                        seen.add(identifier)
                        for key in combined_data.keys():
                            combined_data[key].append(data[key][i])

            except pytesseract.TesseractError as e:
                print(f"Error in PSM mode {mode}: {e}")
            except FileNotFoundError as e:
                print(f"FileNotFoundError in PSM mode {mode}: {e}")
                break     

        return combined_data


    # Define the list of PSM modes to use
    psm_modes = [3, 4, 6, 11]
    # psm_modes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    # Run OCR using multiple PSM modes and combine the results
    data = run_ocr_on_all_modes(image_for_ocr, psm_modes)

    draw = ImageDraw.Draw(image_after_red)
    font_size = 30 
    font = ImageFont.truetype("arial.ttf", font_size)
    font_size = 15 
    fontSmall = ImageFont.truetype("arial.ttf", font_size)

    # Parsing the extracted text and coordinates
    # Drawing rectangles around detected text on the image
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 1:  # Using low confidence texts as well
            text = data['text'][i].strip()
            draw.text((data['left'][i], data['top'][i] + data['height'][i]*2), f"{text}({data['conf'][i]}%)", fill="red", font=fontSmall)
            if text:
                print(f"{text} (Confidence: {data['conf'][i]}%)")
                # Extract room number patterns
                matches = re.findall(r'\b(L?\d{2,3}(-\d+)?(~\d+)?)\b', text)
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
        draw.text((coords["x"], coords["y"] + coords["h"]), text, fill="green", font=font)

    # Saving the final image with highlighted text
    image_after_red.save(os.path.join(tmp_directory, f'{image_name}-7-highlighted_image.png'))
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
    with open(json_file, 'w') as file:
        json.dump(analyzed_results, file, indent=4)

def load_results_from_json(json_file='map.json'):
    """
    Loads analyzed room data from a JSON file.
    :param json_file: The name of the JSON file to load data from.
    :return: A dictionary containing the loaded room data.
    """
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            return json.load(file)
    else:
        return {}
    
# Processing the image based on the user input path
if len(sys.argv) != 2:
    print("Usage: python script.py path_to_image")
    sys.exit(1)    

# Handle tmp directory
tmp_directory = "tmp"
if not os.path.exists(tmp_directory):
    os.makedirs(tmp_directory)

analyzed_results = {}

def perform_image_analysis():
    directory_path = sys.argv[1]
    results = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image_name = os.path.splitext(filename)[0]
            image_results = analyze_image(image_path, image_name)
            results.update(image_results)
    return results


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

def draw_question_mark(draw, position, size=50, fill="green"):
    """
    Draws a question mark at the given position.
    :param draw: ImageDraw object.
    :param position: Tuple (x, y) for the position of the question mark.
    :param size: Size of the question mark.
    :param fill: Color of the question mark.
    """
    font = ImageFont.truetype("arial.ttf", size)
    draw.text(position, "?", fill=fill, font=font)

def calculate_room_similarity(requested_room, available_rooms):
    """
    Calculate similarity score between the requested room and available rooms.
    :param requested_room: The requested room number as a string.
    :param available_rooms: List of available room numbers as strings.
    :return: The most similar room number.
    """
    def similarity_score(room1, room2):
        common_prefix_len = len(os.path.commonprefix([room1, room2]))

        num_part1 = re.sub("[^0-9]", "", room1)
        num_part2 = re.sub("[^0-9]", "", room2)

        num_difference = abs(int(num_part1) - int(num_part2)) if num_part1.isdigit() and num_part2.isdigit() else float('inf')

        return common_prefix_len * 10000 - num_difference

    similarities = [(room, similarity_score(requested_room, room)) for room in available_rooms]
    return max(similarities, key=lambda x: x[1])[0] if similarities else None


# Flask route definitions...
@app.route('/room/<room_number>', methods=['GET'])
def get_room_coordinates(room_number):
    """
    Flask route to get the coordinates of a specific room number.
    :param room_number: The room number requested by the client.
    :return: JSON response with the room's coordinates or an error message.
    """
    analyzed_results = load_results_from_json()
    if room_number in analyzed_results:
        return jsonify(analyzed_results[room_number])
    else:
        return jsonify({"error": "Room number not found"}), 404

@app.route('/room', methods=['GET'])
def list_room_numbers():
    """
    Flask route to list all available room numbers.
    :return: JSON response with a list of all room numbers.
    """
    analyzed_results = load_results_from_json()
    sorted_room_numbers = sorted(analyzed_results.keys())
    return jsonify(sorted_room_numbers)

@app.route('/room', methods=['POST'])
def add_room():
    """
    Flask route to add a new room number with its data.
    :return: JSON response indicating success or error.
    """
    analyzed_results = load_results_from_json()
    data = request.json
    room_number = data.get('room_number')

    if room_number and room_number not in analyzed_results:
        analyzed_results[room_number] = data
        save_results_to_json(analyzed_results)
        return jsonify({"message": "Room added"}), 201
    else:
        return jsonify({"error": "Invalid request or room number already exists"}), 400

@app.route('/room/<room_number>', methods=['PUT'])
def update_room(room_number):
    """
    Flask route to update the data of an existing room number.
    :param room_number: The room number to update.
    :return: JSON response indicating success or error.
    """
    analyzed_results = load_results_from_json()
    data = request.json

    if room_number in analyzed_results:
        analyzed_results[room_number] = data
        save_results_to_json(analyzed_results)
        return jsonify({"message": "Room updated"})
    else:
        return jsonify({"error": "Room number not found"}), 404

@app.route('/api', methods=['GET'])
def list_apis():
    """
    Flask route to list all available API endpoints.
    :return: JSON response with information about all routes.
    """
    routes = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        routes.append({'endpoint': rule.endpoint, 'methods': methods, 'url': str(rule)})
    return jsonify(routes)

@app.route('/image', methods=['GET'])
def get_image():
    """
    Flask route to retrieve a specific image.
    :return: Image file response.
    """
    return send_file(image_path, mimetype='image/png')

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

@app.route('/view/<room_number>', methods=['GET'])
@app.route('/v/<room_number>', methods=['GET'])
@app.route('/r/<room_number>', methods=['GET'])
def view_room_highlighted(room_number):
    """
    Flask route to view an image with a specific room number highlighted.
    This endpoint sends an image with the specified room number highlighted, indicating its location.
    :param room_number: The room number to be highlighted in the image.
    :return: Image response with the specified room number highlighted, or an error message if not found.
    """
    force = request.args.get('force', '').lower() == 'true'
    highlighted_image_path = view_room_highlighted_logic(room_number, force)

    if os.path.exists(highlighted_image_path):
        return send_file(highlighted_image_path, mimetype='image/png')
    else:
        return jsonify({"error": "Room number not found or image could not be created"}), 404

def view_room_highlighted_logic(room_number, force=False):
    """
    Logic to generate an image with a specific room number highlighted.
    :param room_number: The room number to be highlighted in the image.
    :param force: Boolean flag to force regeneration of the image.
    :return: Path to the generated image, or None if not found.
    """
    analyzed_results = load_results_from_json()
    highlighted_image_path = os.path.join(tmp_directory, f"{room_number}-highlighted.png")

    # Check if the image already exists and if force regeneration is not required
    if not force and os.path.exists(highlighted_image_path):
        return highlighted_image_path

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

        # Increasing image sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(5.0)  # Increasing sharpness, value greater than 1.0 makes the image sharper
        draw = ImageDraw.Draw(image)

        # Calculate the center coordinates of the image
        image_center_x, image_center_y = image.size[0] // 2, image.size[1] // 2

        # Center of mark box
        room_center_x = room_info['x'] + room_info['w'] // 2
        room_center_y = room_info['y'] + room_info['h'] // 2

        # Enlarge the mark box
        x_expand = room_info['w'] * 0.1
        y_expand = room_info['h'] * 0.1
        x, y, w, h = room_info['x'] - x_expand, room_info['y'] - y_expand, room_info['w'] + 2 * x_expand, room_info['h'] + 2 * y_expand
        draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=4)

        # Draw arrow and text
        path = calculate_arrow_path(image_center_x, image_center_y, x, y, w, h)
        for i in range(len(path) - 1):
            draw_dashed_line(draw, path[i], path[i + 1], fill="blue")

        draw_arrow_head(draw, path[-2], path[-1], fill="blue")
        if similar_room:
            question_mark_pos = (path[-1][0] + 20, path[-1][1])
            draw_question_mark(draw, question_mark_pos)
            font_size = 30 
            font = ImageFont.truetype("arial.ttf", font_size)
            message1 = f"No exact match: [{room_number}]"
            draw.text((image_center_x + 160, image_center_y - 30), message1, fill="green", font=font)
            message2 = f"Approximately similar: [{similar_room}]"
            draw.text((image_center_x + 160, image_center_y - 30 + 35), message2, fill="green", font=font)

        image.save(highlighted_image_path)
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
    image_files = [f for f in os.listdir(tmp_directory) if image_pattern in f]
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
    Flask route to combine multiple images into one and return the combined image.
    If a 'testset' query parameter is provided, it reads room numbers from 'testset.txt',
    generates images for each room number using view_room_highlighted, and combines these images.
    The 'force' query parameter can be used to force the regeneration of the combined image.

    - 'testset=true': Generates and combines images for room numbers read from 'testset.txt'.
    - 'force=true': Ignores any cached combined image and forces the creation and combination of new images.
    -  Without 'force' parameter or with 'force=false': If a previously generated combined image exists, it's returned.   

    :return: Response with the combined image file or an error message.
    """
    testset = request.args.get('testset')
    force = request.args.get('force')


    # If the testset parameter is provided, perform validation on the testset
    if testset:
        combined_image_path = os.path.join(tmp_directory, 'combined_testset_image.png')

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
            return jsonify({"error": "No images were generated from the testset"}), 404
    else:
        combined_image_path = os.path.join(tmp_directory, 'combined_image.png')

        if not force:
            if os.path.exists(combined_image_path):
                return send_file(combined_image_path, mimetype='image/png')

        # Original behavior for combining highlighted images
        combined_image = combine_images(tmp_directory, '-7-highlighted_image.png')
        combined_image.save(combined_image_path)

        return send_file(combined_image_path, mimetype='image/png')

# main
if __name__ == '__main__':
    json_file = 'map.json'

    # Check if the map.json file exists
    if os.path.exists(json_file):
        # Prompt the user to decide whether to perform new image analysis or use existing data
        user_input = input("The map.json file already exists. Do you want to proceed with new image analysis? (y/n): ")
        if user_input.lower() != 'y':
            # If the user chooses not to perform new analysis, use the existing map.json file
            print("Using existing map.json file to run the server.")
            with open(json_file, 'r') as f:
                analyzed_results = json.load(f)  # Load the analyzed results from the existing file
        else:
            # If the user chooses to perform new analysis
            analyzed_results = perform_image_analysis()  # Perform image analysis to extract room data
            save_results_to_json(analyzed_results, json_file)  # Save the new analyzed results to map.json
    else:
        # If the map.json file does not exist, perform new image analysis
        analyzed_results = perform_image_analysis()  # Perform image analysis to extract room data
        save_results_to_json(analyzed_results, json_file)  # Save the analyzed results to a new map.json file

    # Start the Flask server
        # Run the Flask app with debug mode on, accessible on all interfaces on port 1111
    app.run(debug=True, host='0.0.0.0', port=1111)  

