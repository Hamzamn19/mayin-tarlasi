from __future__ import annotations

import glob
import os
import textwrap
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "landmine_flat", "test")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "runs",
    "detect",
    "Landmine_Detection_2026",
    "YOLO26_S_Standard",
    "weights",
    "best.pt",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "plots", "yolo_examples")


SAMPLE_NAMES = [
    "Jan_Jan_Afternoon_10_lwir_3.jpg",
    "May_May_Afternoon_20_lwir_0_lwir_20.jpg",
    "elevation_test_Jan_Afternoon_10m_0_lwir_22.jpg",
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def find_sample_images() -> list[str]:
    available = []
    for sample_name in SAMPLE_NAMES:
        candidate = os.path.join(TEST_DIR, sample_name)
        if os.path.exists(candidate):
            available.append(candidate)

    if available:
        return available

    fallback = sorted(glob.glob(os.path.join(TEST_DIR, "**", "*.jpg"), recursive=True))
    return fallback[:3]


def count_ground_truth_objects(image_path: str) -> int:
    xml_path = os.path.splitext(image_path)[0] + ".xml"
    if not os.path.exists(xml_path):
        return 0

    try:
        tree = ET.parse(xml_path)
        return len(tree.findall(".//object"))
    except ET.ParseError:
        return 0


def fit_image_to_canvas(image: np.ndarray, target_size: tuple[int, int]) -> Image.Image:
    target_width, target_height = target_size
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    fitted = Image.new("RGB", target_size, (13, 18, 28))
    pil_image.thumbnail((target_width - 24, target_height - 24), Image.Resampling.LANCZOS)
    offset_x = (target_width - pil_image.width) // 2
    offset_y = (target_height - pil_image.height) // 2
    fitted.paste(pil_image, (offset_x, offset_y))
    return fitted


def draw_footer(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str, font_title, font_body, accent: tuple[int, int, int]) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=26, fill=(12, 16, 24), outline=accent, width=3)
    draw.text((left + 24, top + 18), title, fill=(255, 255, 255), font=font_title)

    wrapped = textwrap.fill(body, width=94)
    draw.multiline_text(
        (left + 24, top + 74),
        wrapped,
        fill=(225, 230, 240),
        font=font_body,
        spacing=8,
    )


def create_comparison_panel(original: np.ndarray, annotated: np.ndarray, title: str, subtitle: str, stats_line: str) -> Image.Image:
    panel_width = 1220
    image_height = 860
    footer_height = 180
    spacing = 24
    canvas_height = image_height + footer_height + 96
    canvas = Image.new("RGB", (panel_width, canvas_height), (7, 10, 18))

    draw = ImageDraw.Draw(canvas)
    font_title = load_font(36)
    font_subtitle = load_font(22)
    font_body = load_font(20)
    font_small = load_font(18)

    draw.text((46, 26), title, fill=(255, 255, 255), font=font_title)
    draw.text((46, 70), subtitle, fill=(173, 183, 200), font=font_subtitle)

    half_width = (panel_width - (spacing * 3)) // 2
    top_y = 115
    left_canvas = fit_image_to_canvas(original, (half_width, image_height))
    right_canvas = fit_image_to_canvas(annotated, (half_width, image_height))

    canvas.paste(left_canvas, (spacing, top_y))
    canvas.paste(right_canvas, (half_width + (spacing * 2), top_y))

    draw.rounded_rectangle((spacing, top_y, spacing + half_width, top_y + image_height), radius=22, outline=(90, 110, 140), width=3)
    draw.rounded_rectangle((half_width + (spacing * 2), top_y, panel_width - spacing, top_y + image_height), radius=22, outline=(114, 193, 255), width=3)

    draw.rectangle((spacing + 18, top_y + 18, spacing + 190, top_y + 58), fill=(55, 82, 120))
    draw.rectangle((half_width + (spacing * 2) + 18, top_y + 18, half_width + (spacing * 2) + 190, top_y + 58), fill=(36, 130, 88))
    draw.text((spacing + 34, top_y + 26), "Original image", fill=(255, 255, 255), font=font_small)
    draw.text((half_width + (spacing * 2) + 34, top_y + 26), "YOLO inference", fill=(255, 255, 255), font=font_small)

    footer_top = top_y + image_height + 28
    draw_footer(
        draw,
        (spacing, footer_top, panel_width - spacing, footer_top + footer_height),
        "Why this sample is useful",
        stats_line,
        font_title=load_font(26),
        font_body=font_body,
        accent=(114, 193, 255),
    )

    return canvas


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"YOLO weights not found: {MODEL_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    sample_images = find_sample_images()
    if not sample_images:
        raise FileNotFoundError(f"No JPG images found under {TEST_DIR}")

    generated = []
    for index, image_path in enumerate(sample_images, start=1):
        original = cv2.imread(image_path)
        if original is None:
            continue

        results = model.predict(source=original, verbose=False)
        result = results[0]
        annotated = result.plot()

        boxes = result.boxes
        detection_count = int(len(boxes)) if boxes is not None else 0
        best_conf = float(boxes.conf.max().item()) if detection_count and boxes.conf is not None else 0.0
        gt_count = count_ground_truth_objects(image_path)
        image_name = os.path.basename(image_path)

        stats_line = (
            f"Image: {image_name} | Ground-truth objects: {gt_count} | YOLO detections: {detection_count} | "
            f"Best confidence: {best_conf:.2f}. The right panel shows model boxes and labels, while the left panel keeps the raw thermal frame for visual comparison."
        )

        panel = create_comparison_panel(
            original,
            annotated,
            title=f"YOLO26 Detection Example #{index}",
            subtitle="Landmine imagery with an explanatory comparison layout for README usage",
            stats_line=stats_line,
        )

        output_name = f"yolo_demo_{index:02d}_{os.path.splitext(image_name)[0]}.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        panel.save(output_path, quality=95)
        generated.append(output_path)
        print(f"Saved: {output_path}")

    if not generated:
        raise RuntimeError("No YOLO demo images were generated.")


if __name__ == "__main__":
    main()