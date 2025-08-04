import os
import cv2
import matplotlib.pyplot as plt
import easyocr
import re

# OCR configuration
OCR_LANGUAGES = ['en']
SAVE_DIR = "inference_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# Ask user for input image path
image_path = input("Enter path to image for inference: ").strip()
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load image and initialize OCR reader
image = cv2.imread(image_path)
original_image = image.copy()
reader = easyocr.Reader(OCR_LANGUAGES)

# Clean and normalize detected text for accurate matching
def clean_and_normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', '', text)      
    # Common OCR misread corrections
    replacements = {
        '1gps': 'igps',
        'lgps': 'igps',
        'igp5': 'igps',
        'igp$': 'igps',
        'icps': 'igps',
        'cps':  'gps',
    }
    for wrong, correct in replacements.items():
        if wrong in text:
            text = text.replace(wrong, correct)
    return text

# Identify if the detected text is strongly or partially "iGPS"
def classify_igps(text):
    cleaned = clean_and_normalize_text(text)
    if cleaned == 'igps':
        return 'strong'
    likely_partials = ['igp', 'gps']
    if cleaned in likely_partials:
        return 'partial'
    ambiguous = ['cps', 'icps']
    if cleaned in ambiguous:
        return 'ambiguous'
    # Check for fuzzy matches like "i.g.p.s"
    if re.search(r'i.{0,1}g.{0,1}p.{0,1}s', cleaned):
        return 'partial'
    return None

print("Running OCR on image...")
results = reader.readtext(image)

# Counters for each detection category
match_counts = {'strong': 0, 'partial': 0, 'ambiguous': 0}
color_map = {'strong': (0, 255, 0), 'partial': (0, 165, 255), 'ambiguous': (0, 0, 255)}

# Process each OCR result
for bbox, text, conf in results:
    match_type = classify_igps(text)
    if match_type:
        match_counts[match_type] += 1
        cleaned = clean_and_normalize_text(text)
        # Extract bounding box points
        top_left, top_right, bottom_right, bottom_left = bbox
        x1, y1 = map(int, top_left)
        x2, y2 = map(int, bottom_right)
        color = color_map[match_type]
        thickness = 3 if conf > 0.7 else 2
        # Draw rectangle around detected text
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"{match_type.upper()} iGPS: {cleaned} ({conf:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Convert image to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis("off")
title = f"iGPS OCR Detection - Strong: {match_counts['strong']} | Partial: {match_counts['partial']} | Ambiguous: {match_counts['ambiguous']}"
plt.title(title)
plt.show()

# Save annotated image to output directory
save_path = os.path.join(SAVE_DIR, f"ocr_igps_{os.path.basename(image_path)}")
cv2.imwrite(save_path, image)
print(f"Annotated image saved to: {save_path}")

# Print a detection summary
total_matches = sum(match_counts.values())
if total_matches:
    print(f"\niGPS-related text detected!")
    print(f"  - Strong Matches: {match_counts['strong']}")
    print(f"  - Partial Matches: {match_counts['partial']}")
    print(f"  - Ambiguous Matches: {match_counts['ambiguous']}")
else:
    print("\nNo iGPS-related text confidently detected.")

# Print all OCR detections with their cleaned versions
print("\nAll OCR Detections:")
for i, (bbox, text, conf) in enumerate(results):
    cleaned = clean_and_normalize_text(text)
    match_type = classify_igps(text)
    status = f"{match_type.upper()}" if match_type else "Other"
    print(f"  {i+1}. {status}: '{text}' → '{cleaned}' (conf: {conf:.3f})")
