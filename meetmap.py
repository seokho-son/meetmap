import pytesseract  # Tesseract OCR library for text recognition
import cv2  # OpenCV library for computer vision tasks
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import sys
import numpy as np
import re
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

def analyze_image(image_path):
    """
    Analyzes the image and extracts room number information.
    :param image_path: Path to the image to be analyzed.
    :return: A dictionary with room numbers and their coordinates.
    """
    analyzed_results = {}
    image = Image.open(image_path).convert("RGB")

    # resized_image
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    new_height = 1000
    new_width = int(aspect_ratio * new_height)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
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
    image_after_red.save('1-image_after_red.png')

    # G 채널이 20보다 낮은 모든 픽셀을 흰색으로 변경
    green_threshold = 20
    green_mask = image_data[:, :, 1] < green_threshold
    image_data[green_mask] = [255, 255, 255]

    # 변경된 이미지를 저장
    image_after_green = Image.fromarray(image_data)
    image_after_green.save('2-image_after_green.png')

    # RGB 각 색상 채널 간의 편차가 14 이상 나지 않는 경우 흰색으로 변경
    max_deviation = np.max(image_data, axis=-1) - np.min(image_data, axis=-1)
    uniform_color_mask = max_deviation < 14
    image_data[uniform_color_mask] = [255, 255, 255]

    # 변경된 이미지를 저장
    image_rgb_threshold = Image.fromarray(image_data)
    image_rgb_threshold.save('3-image_after_rgb_threshold.png')

    # 콘트라스트를 높임
    enhancer = ImageEnhance.Contrast(image_rgb_threshold)
    image_rgb_threshold = enhancer.enhance(1.5)

    # 이미지 전처리 단계
    # 이미지 이진화를 위한 임계값 설정
    threshold_value = 100

    # 이미지를 그레이스케일로 변환
    gray_image = image_rgb_threshold.convert('L')
    gray_image.save('4-gray_image.png')  # Save the gray image

    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(np.array(gray_image), threshold_value, 255, cv2.THRESH_BINARY)
    binary_image_save = Image.fromarray(binary_image)
    binary_image_save.save('5-binary_image.png')  # Save the binary image

    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=2)

    # Convert back to PIL Image to use with pytesseract
    image_for_ocr = Image.fromarray(eroded_image)
    image_for_ocr.save('6-dilated_eroded_image.png')  

    # Using Tesseract OCR to extract text and their coordinates from the image
    config = '--psm 6'
    data = pytesseract.image_to_data(image_for_ocr, config=config, output_type=pytesseract.Output.DICT)

    # Parsing the extracted text and coordinates
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 10:  # Using only high confidence text
            text = data['text'][i].strip()
            if text and re.match(r'\b\d{3}\b', text):  # Matching specific text pattern
                (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                print(f"{text} (Confidence: {data['conf'][i]}%)")
                print(f"Location: X: {x}, Y: {y}, Width: {width}, Height: {height}")
                # Storing extracted data
                analyzed_results[text] = {
                    "x": x, "y": y, "w": width, "h": height
                }

    # Drawing rectangles around detected text on the image
    draw = ImageDraw.Draw(image_after_red)
    font = ImageFont.load_default()

    for text, coords in analyzed_results.items():
        draw.rectangle(((coords["x"], coords["y"]), (coords["x"] + coords["w"], coords["y"] + coords["h"])), outline="red")
        draw.text((coords["x"], coords["y"] + coords["h"]), text, fill="black", font=font)

    # Saving the final image with highlighted text
    image_after_red.save('7-highlighted_image.png')

    return analyzed_results

# Processing the image based on the user input path
if len(sys.argv) != 2:
    print("Usage: python script.py path_to_image")
    sys.exit(1)

image_path = sys.argv[1]
analyzed_results = analyze_image(image_path)

# Flask route definitions...

@app.route('/room/<room_number>', methods=['GET'])
def get_room_coordinates(room_number):
    if room_number in analyzed_results:
        return jsonify(analyzed_results[room_number])
    else:
        return jsonify({"error": "Room number not found"}), 404

@app.route('/room', methods=['GET'])
def list_room_numbers():
    sorted_room_numbers = sorted(analyzed_results.keys())
    return jsonify(sorted_room_numbers)

@app.route('/room', methods=['POST'])
def add_room():
    data = request.json
    room_number = data.get('room_number')

    if room_number and room_number not in analyzed_results:
        analyzed_results[room_number] = data
        return jsonify({"message": "Room added"}), 201
    else:
        return jsonify({"error": "Invalid request or room number already exists"}), 400

@app.route('/room/<room_number>', methods=['PUT'])
def update_room(room_number):
    data = request.json

    if room_number in analyzed_results:
        analyzed_results[room_number] = data
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

if __name__ == '__main__':
    app.run(debug=True)