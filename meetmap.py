import sys
import numpy as np
import re
import os
import json
import pytesseract  # Tesseract OCR library for text recognition
import cv2  # OpenCV library for computer vision tasks
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

def analyze_image(image_path, image_name):
    """
    Analyzes the image and extracts room number information.
    :param image_path: Path to the image to be analyzed.
    :return: A dictionary with room numbers and their coordinates.
    """
    analyzed_results = {}
    image = Image.open(image_path).convert("RGB")

    # 이미지 화질 개선 (콘트라스트 증가)
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(1.01)  # 콘트라스트를 1.01배 증가
    enhanced_image.save(os.path.join(tmp_directory, f'{image_name}-0-image_after_enhance.png'))

    # resized_image
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

    # Image for map with enhanced
    enhancer = ImageEnhance.Contrast(image_after_red)
    enhanced_image = enhancer.enhance(1.5) 
    enhanced_image.save(os.path.join(tmp_directory, f'{image_name}-map.png'))

    # G 채널이 20보다 낮은 모든 픽셀을 흰색으로 변경
    green_threshold = 20
    green_mask = image_data[:, :, 1] < green_threshold
    image_data[green_mask] = [255, 255, 255]

    # 변경된 이미지를 저장
    image_after_green = Image.fromarray(image_data)
    image_after_green.save(os.path.join(tmp_directory, f'{image_name}-2-image_after_green.png'))

    # RGB 각 색상 채널 간의 편차가 14 이상 나지 않는 경우 흰색으로 변경
    max_deviation = np.max(image_data, axis=-1) - np.min(image_data, axis=-1)
    uniform_color_mask = max_deviation < 14
    image_data[uniform_color_mask] = [255, 255, 255]

    # 변경된 이미지를 저장
    image_rgb_threshold = Image.fromarray(image_data)
    image_rgb_threshold.save(os.path.join(tmp_directory, f'{image_name}-3-image_after_rgb_threshold.png'))

    # 콘트라스트를 높임
    enhancer = ImageEnhance.Contrast(image_rgb_threshold)
    image_rgb_threshold = enhancer.enhance(1.5)

    # 이미지 전처리 단계
    # 이미지 이진화를 위한 임계값 설정
    threshold_value = 100

    # 이미지를 그레이스케일로 변환
    gray_image = image_rgb_threshold.convert('L')
    gray_image.save(os.path.join(tmp_directory, f'{image_name}-4-gray_image.png'))

    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(np.array(gray_image), threshold_value, 255, cv2.THRESH_BINARY)
    binary_image_save = Image.fromarray(binary_image)
    binary_image_save.save(os.path.join(tmp_directory, f'{image_name}-5-binary_image.png'))

    # 샤프닝 커널 정의
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  30, -1],
                                  [-1, -1, -1]])
    edges = cv2.Canny(binary_image, 100, 200)
    mask = edges != 0
    mask = mask.astype(np.uint8)

    # 이미지에 샤프닝 적용
    sharpened_image = cv2.filter2D(binary_image, -1, sharpening_kernel)
    binary_image[mask == 1] = sharpened_image[mask == 1]

    # 결과 이미지 저장 또는 표시
    # cv2.imwrite(os.path.join(tmp_directory, f'{image_name}-6-sharpened_image.png'), binary_image)

    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)


    # Convert back to PIL Image to use with pytesseract
    image_for_ocr = Image.fromarray(eroded_image)
    image_for_ocr.save(os.path.join(tmp_directory, f'{image_name}-6-dilated_eroded_image.png')) 

    # Using Tesseract OCR to extract text and their coordinates from the image
    config = '--psm 6'
    data = pytesseract.image_to_data(image_for_ocr, config=config, output_type=pytesseract.Output.DICT)

    # Parsing the extracted text and coordinates
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 10:  # Using only high confidence text
            text = data['text'][i].strip()
            if text and re.match(r'\b(L?\d{3}(-\d+)?(~\d+)?)\b', text):
                (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                print(f"{text} (Confidence: {data['conf'][i]}%)")
                print(f"Location: X: {x}, Y: {y}, Width: {width}, Height: {height}")
                # Storing extracted data
                analyzed_results[text] = {
                    "floor": image_name, "x": x, "y": y, "w": width, "h": height
                }

    # Drawing rectangles around detected text on the image
    draw = ImageDraw.Draw(image_after_red)
    font = ImageFont.load_default()

    for text, coords in analyzed_results.items():
        draw.rectangle(((coords["x"], coords["y"]), (coords["x"] + coords["w"], coords["y"] + coords["h"])), outline="red")
        draw.text((coords["x"], coords["y"] + coords["h"]), text, fill="black", font=font)

    # Saving the final image with highlighted text
    image_after_red.save(os.path.join(tmp_directory, f'{image_name}-7-highlighted_image.png'))
    return analyzed_results

# Utilities
def save_results_to_json(analyzed_results, json_file='map.json'):
    """
    Saves the analyzed results to a JSON file.
    """
    with open(json_file, 'w') as file:
        json.dump(analyzed_results, file, indent=4)

def load_results_from_json(json_file='map.json'):
    """
    Loads analyzed results from a JSON file.
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

# tmp 디렉토리 확인 및 생성
tmp_directory = "tmp"
if not os.path.exists(tmp_directory):
    os.makedirs(tmp_directory)

analyzed_results = {}


# Drawing assets...

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
    path = [(center_x, center_y)]  # 시작점

    box_center_x = box_x + box_w / 2
    box_center_y = box_y + box_h / 2

    # 먼저 수직 방향으로 이동
    if center_y != box_center_y:
        path.append((center_x, box_center_y))

    # 수평 방향으로 이동하여 엣지 결정
    if box_center_x > center_x:  # 사각형이 오른쪽에 있음
        path.append((box_x, box_center_y))  # 왼쪽 엣지 중앙
    elif box_center_x < center_x:  # 사각형이 왼쪽에 있음
        path.append((box_x + box_w, box_center_y))  # 오른쪽 엣지 중앙
    else:  # 사각형이 수직선상에 있음
        if center_y > box_center_y:  # 사각형이 위에 있음
            path.append((center_x, box_y))  # 아래쪽 엣지 중앙
        elif center_y < box_center_y:  # 사각형이 아래에 있음
            path.append((center_x, box_y + box_h))  # 윗쪽 엣지 중앙

    return path

# Flask route definitions...

@app.route('/room/<room_number>', methods=['GET'])
def get_room_coordinates(room_number):
    analyzed_results = load_results_from_json()
    if room_number in analyzed_results:
        return jsonify(analyzed_results[room_number])
    else:
        return jsonify({"error": "Room number not found"}), 404

@app.route('/room', methods=['GET'])
def list_room_numbers():
    analyzed_results = load_results_from_json()
    sorted_room_numbers = sorted(analyzed_results.keys())
    return jsonify(sorted_room_numbers)

@app.route('/room', methods=['POST'])
def add_room():
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
    routes = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        routes.append({'endpoint': rule.endpoint, 'methods': methods, 'url': str(rule)})
    return jsonify(routes)

@app.route('/image', methods=['GET'])
def get_image():
    return send_file(image_path, mimetype='image/png')

def expand_range(range_str):
    """
    Expands a range string into a list of all encompassed values.
    :param range_str: A range string (e.g., "111~223", "111-223").
    :return: A list of strings representing all values within the range.
    """
    # 정규식을 사용하여 숫자 범위 추출
    match = re.match(r'(\d+)(~|,|-)(\d+)', range_str)
    if match:
        start, _, end = match.groups()
        start, end = int(start), int(end)
        return [str(i) for i in range(start, end + 1)]
    else:
        return [range_str.split('-')[0]]  # '-5'와 같은 접미사는 무시

def is_number_in_range(num_str, range_str):
    """
    Check if a given number string is within a specified range.
    :param num_str: The number string to check (as a string).
    :param range_str: The range string (e.g., "100-200", "L300~L350").
    :return: Boolean indicating if the number is within the range.
    """
    num_str = str(num_str).split('-')[0]  # Ensure num_str is a string, '-5'와 같은 접미사 무시
    expanded_range = expand_range(range_str)
    return num_str in expanded_range

@app.route('/view/<room_number>', methods=['GET'])
def view_room_highlighted(room_number):
    analyzed_results = load_results_from_json()
    numeric_part = int(re.sub("[^0-9]", "", room_number))  # Extract numeric part

    # Check for exact match first
    if room_number in analyzed_results:
        room_info = analyzed_results[room_number]
    else:
        # If not an exact match, check for range match
        for key, value in analyzed_results.items():
            if is_number_in_range(numeric_part, key):
                room_info = value
                break
        else:
            return jsonify({"error": "Room number not found or invalid format"}), 404

    floor_image_path = os.path.join(tmp_directory, f"{room_info['floor']}-map.png")
    if os.path.exists(floor_image_path):
        image = Image.open(floor_image_path)
        draw = ImageDraw.Draw(image)

        # 이미지 중앙 좌표 계산
        image_center_x, image_center_y = image.size[0] // 2, image.size[1] // 2

        # 사각형 중앙 좌표
        room_center_x = room_info['x'] + room_info['w'] // 2
        room_center_y = room_info['y'] + room_info['h'] // 2

        # 사각형 크기를 10% 키움 (중앙을 기준으로)
        x_expand = room_info['w'] * 0.05
        y_expand = room_info['h'] * 0.05
        x, y, w, h = room_info['x'] - x_expand, room_info['y'] - y_expand, room_info['w'] + 2 * x_expand, room_info['h'] + 2 * y_expand
        draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=4)

        # 화살표 경로 계산
        path = calculate_arrow_path(image_center_x, image_center_y, x, y, w, h)

        # 화살표 그리기
        for i in range(len(path) - 1):
            draw_dashed_line(draw, path[i], path[i + 1], fill="blue")

        # 화살표 끝 그리기
        draw_arrow_head(draw, path[-2], path[-1], fill="blue")

        highlighted_image_path = os.path.join(tmp_directory, f"{room_info['floor']}-highlighted.png")
        image.save(highlighted_image_path)
        return send_file(highlighted_image_path, mimetype='image/png')
    else:
        return jsonify({"error": "Floor image not found"}), 404



if __name__ == '__main__':
    json_file = 'map.json'

    # map.json 파일이 존재하는지 확인
    if os.path.exists(json_file):
        user_input = input("map.json 파일이 이미 존재합니다. 새로운 이미지 분석을 진행하시겠습니까? (y/n): ")
        if user_input.lower() != 'y':
            print("기존 map.json 파일을 사용하여 서버를 구동합니다.")
            with open(json_file, 'r') as f:
                analyzed_results = json.load(f)
        else:
            analyzed_results = perform_image_analysis()
            save_results_to_json(analyzed_results, json_file)
    else:
        # 파일이 존재하지 않는 경우, 이미지 분석 실행
        analyzed_results = perform_image_analysis()
        save_results_to_json(analyzed_results, json_file)

    # Flask 서버 구동
    app.run(debug=True, host='0.0.0.0', port=1111)
