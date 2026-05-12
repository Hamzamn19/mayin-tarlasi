from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
PNG_PATH = os.path.join(OUTPUT_DIR, "vote_flow_cards.png")
GIF_PATH = os.path.join(OUTPUT_DIR, "vote_flow_cards.gif")


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def lerp_color(color_a, color_b, factor: float):
    return tuple(int(color_a[i] + (color_b[i] - color_a[i]) * factor) for i in range(3))


def draw_vertical_gradient(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], top_color, bottom_color) -> None:
    left, top, right, bottom = box
    height = max(1, bottom - top)
    for offset in range(height):
        factor = offset / (height - 1 if height > 1 else 1)
        color = lerp_color(top_color, bottom_color, factor)
        draw.line((left, top + offset, right, top + offset), fill=color)


def rounded_box(base: Image.Image, box: tuple[int, int, int, int], fill, outline, radius: int = 30, width: int = 4):
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    base.alpha_composite(overlay)


def add_shadow(base: Image.Image, box: tuple[int, int, int, int], radius: int = 30) -> None:
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, 150))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    base.alpha_composite(shadow)


def draw_card(base: Image.Image, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str, stage: str, accent: str, active: bool) -> None:
    left, top, right, bottom = box
    accent_rgb = hex_to_rgb(accent)
    title_font = load_font(32, bold=True)
    body_font = load_font(18)
    stage_font = load_font(16, bold=True)

    # soft glass-like card
    draw.rounded_rectangle(box, radius=34, fill=(17, 24, 42, 235), outline=accent_rgb + (255,), width=4)

    # top ribbon
    ribbon = (left + 18, top + 18, right - 18, top + 58)
    draw.rounded_rectangle(ribbon, radius=18, fill=accent_rgb + (255,))
    draw.text((left + 36, top + 27), stage, fill=(10, 15, 28), font=stage_font)

    # icon glow
    glow = Image.new("RGBA", draw.im.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    center_x = int((left + right) / 2)
    glow_draw.ellipse((center_x - 48, top + 82, center_x + 48, top + 178), fill=accent_rgb + (90,))
    glow = glow.filter(ImageFilter.GaussianBlur(16))
    base.alpha_composite(glow)

    draw.ellipse((center_x - 34, top + 96, center_x + 34, top + 164), outline=accent_rgb + (255,), width=5)
    draw.text((center_x, top + 128), title[0], fill=(255, 255, 255), font=load_font(30, bold=True), anchor="mm")

    draw.text((left + 26, top + 196), title, fill=(255, 255, 255), font=title_font)
    draw.multiline_text((left + 26, top + 250), body, fill=(224, 231, 245), font=body_font, spacing=8)

    if active:
        highlight = (left + 8, top + 8, right - 8, bottom - 8)
        draw.rounded_rectangle(highlight, radius=30, outline=(255, 255, 255), width=5)


def add_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str, width: int = 9) -> None:
    draw.line((start, end), fill=hex_to_rgb(color), width=width, joint="curve")
    ex, ey = end
    sx, sy = start
    dx, dy = ex - sx, ey - sy
    length = max((dx ** 2 + dy ** 2) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    left = (ex - int(ux * 20) + int(-uy * 10), ey - int(uy * 20) + int(ux * 10))
    right = (ex - int(ux * 20) + int(uy * 10), ey - int(uy * 20) + int(-ux * 10))
    draw.polygon([end, left, right], fill=hex_to_rgb(color))


def build_canvas(active_index: int | None = None) -> Image.Image:
    width, height = 2200, 1280
    canvas = Image.new("RGBA", (width, height), hex_to_rgb("#07111f") + (255,))
    bg_draw = ImageDraw.Draw(canvas)
    draw_vertical_gradient(bg_draw, (0, 0, width, height), hex_to_rgb("#07111f"), hex_to_rgb("#0d1b2d"))

    # Decorative background panels
    for x in range(0, width, 280):
        bg_draw.rounded_rectangle((x + 12, 80, x + 220, 1180), radius=38, fill=(255, 255, 255, 7), outline=(255, 255, 255, 18), width=2)

    title_font = load_font(56, bold=True)
    subtitle_font = load_font(24)
    note_font = load_font(22)

    bg_draw.text((110, 70), "Hybrid Voting Flow", fill=(255, 255, 255), font=title_font)
    bg_draw.text((110, 140), "A refined card sequence showing how the system moves from candidate detection to final approval.", fill=(184, 198, 220), font=subtitle_font)

    card_top = 260
    card_height = 660
    card_width = 560
    gap = 90
    left = 100
    card_boxes = [
        (left, card_top, left + card_width, card_top + card_height),
        (left + card_width + gap, card_top, left + card_width * 2 + gap, card_top + card_height),
        (left + card_width * 2 + gap * 2, card_top, left + card_width * 3 + gap * 2, card_top + card_height),
    ]

    labels = [
        ("YOLO26", "Finds candidate mine regions\nProduces bounding boxes and confidence\nKeeps recall high at the first gate", "STAGE 1", "#4fc3f7"),
        ("RF", "Reads the crop features\nChecks circularity, thermal contrast, and edge density\nSuppresses false positives", "STAGE 2", "#7ee787"),
        ("VOTE", "Combines YOLO confidence\nwith Random Forest probability\nApplies the final ensemble rule", "STAGE 3", "#ffb86b"),
    ]

    # Shadows first
    for box in card_boxes:
        add_shadow(canvas, box)

    draw = ImageDraw.Draw(canvas)
    for index, (box, label) in enumerate(zip(card_boxes, labels)):
        title, body, stage, accent = label
        is_active = active_index == index
        if active_index is None:
            is_active = True
        draw_card(canvas, draw, box, title, body, stage, accent, is_active)

    # Arrows / connectors
    arrow_y = card_top + card_height // 2
    add_arrow(draw, (card_boxes[0][2] + 26, arrow_y), (card_boxes[1][0] - 26, arrow_y), "#4fc3f7")
    add_arrow(draw, (card_boxes[1][2] + 26, arrow_y), (card_boxes[2][0] - 26, arrow_y), "#7ee787")

    # Bottom rule bar
    rule_box = (110, 1000, width - 110, 1185)
    draw.rounded_rectangle(rule_box, radius=36, fill=(10, 17, 31, 245), outline=hex_to_rgb("#35557d") + (255,), width=3)
    draw.text((width // 2, 1040), "Voting rule in deployment", fill=(255, 255, 255), font=load_font(30, bold=True), anchor="mm")
    draw.text((width // 2, 1100), "final_score = (YOLO_confidence + RF_probability) / 2", fill=(224, 238, 255), font=load_font(28), anchor="mm")
    draw.text((width // 2, 1148), "if final_score >= 0.50  →  mine detected   |   else  →  reject candidate", fill=(156, 203, 255), font=note_font, anchor="mm")

    # Final highlight badge
    badge = (850, 190, 1350, 238)
    draw.rounded_rectangle(badge, radius=22, fill=(20, 35, 63, 255), outline=hex_to_rgb("#79b8ff") + (255,), width=2)
    draw.text((1100, 214), "High recall first, precise decision second", fill=(238, 245, 255), font=load_font(20, bold=True), anchor="mm")

    return canvas


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    final_canvas = build_canvas(active_index=None)
    final_canvas.save(PNG_PATH)

    frames = []
    for active_index in [0, 1, 2, 1, 2]:
        frame = build_canvas(active_index=active_index)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))

    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=[900, 900, 900, 700, 1000],
        loop=0,
        optimize=False,
    )

    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {GIF_PATH}")


if __name__ == "__main__":
    main()