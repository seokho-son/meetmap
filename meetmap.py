import pytesseract
import cv2  # OpenCV library for computer vision tasks
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import sys
import numpy as np
import re

# 사용자 입력을 통해 이미지 경로 받기
if len(sys.argv) != 2:
    print("Usage: python script.py path_to_image")
    sys.exit(1)

image_path = sys.argv[1]

# 이미지 로드
image = Image.open(image_path).convert("RGB")

# 이미지 정보 추출
image_info = {
    "Size": image.size,
    "DPI": image.info.get('dpi')
}

# 이미지 정보 출력
print("Image Information:")
for key, value in image_info.items():
    print(f"{key}: {value}")

# 이미지 데이터를 numpy 배열로 변환
image_data = np.array(image)

# R 또는 B 채널이 90을 넘는 경우에 해당하는 마스크 생성
red_mask = image_data[:, :, 0] > 90
blue_mask = image_data[:, :, 2] > 90
# R 또는 B 채널 둘 중 하나라도 100을 넘는 경우에 해당하는 픽셀을 흰색으로 변경
image_data[red_mask | blue_mask] = [255, 255, 255]

# 변경된 이미지를 저장
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

# 최종 변경된 이미지를 저장
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

# # 원본 이미지의 크기를 구하고, 1/3로 줄임
# original_size = gray_image.size
# reduced_size = (original_size[0] // 3, original_size[1] // 3) 
# gray_image = gray_image.resize(reduced_size, Image.Resampling.LANCZOS)  # Image.Resampling.LANCZOS 사용


# Threshold the image to get a binary image
_, binary_image = cv2.threshold(np.array(gray_image), threshold_value, 255, cv2.THRESH_BINARY)
binary_image_save = Image.fromarray(binary_image)
binary_image_save.save('5-binary_image.png')  # Save the binary image

# 커널 크기를 (5, 5)로 증가
kernel = np.ones((2, 2), np.uint8)

# 반복 횟수를 2로 증가
dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
eroded_image = cv2.erode(dilated_image, kernel, iterations=2)

# Convert back to PIL Image to use with pytesseract
image_for_ocr = Image.fromarray(eroded_image)
image_for_ocr.save('6-dilated_eroded_image.png')  

# 이진화된 이미지를 사용하여 OCR 실행하고 위치 정보 추출
config = '--psm 6'
data = pytesseract.image_to_data(image_for_ocr, config=config, output_type=pytesseract.Output.DICT)

# 추출된 데이터에서 텍스트 및 위치 정보 출력
for i in range(len(data['text'])):
    if int(data['conf'][i]) > 10:  # 신뢰도가 10 이상인 데이터만 사용
        text = data['text'][i].strip()
        if text and re.match(r'\b\d{2}\b', text):  # 3자리 숫자를 포함하는 경우만 처리
            (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            print(f"{text} (Confidence: {data['conf'][i]}%)")
            print(f"Location: X: {x}, Y: {y}, Width: {width}, Height: {height}")

for i in range(len(data['text'])):
    if int(data['conf'][i]) > 10:  # 신뢰도가 10 이상인 데이터만 사용
        text = data['text'][i].strip()
        if text and re.match(r'\b\d{3}\b', text):  # 3자리 숫자를 포함하는 경우만 처리
            (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            print(f"{text} ")

# 이미지에 텍스트 위치를 나타내는 사각형을 그림
draw = ImageDraw.Draw(image_after_red)
font = ImageFont.load_default()  # 기본 폰트 사용

for i in range(len(data['text'])):
    if int(data['conf'][i]) > 10:  # 신뢰도가 50 이상인 데이터만 사용
        text = data['text'][i].strip()
        if text and re.match(r'\b\d{3}\b', text):  # 3자리 숫자를 포함하는 경우만 처리
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            draw.rectangle(((x, y), (x + w, y + h)), outline="red")
            draw.text((x, y + h), text, fill="black", font=font)

# 사각형이 그려진 이미지를 저장
image_after_red.save('7-highlighted_image.png')