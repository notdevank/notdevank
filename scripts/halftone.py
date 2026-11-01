#!/usr/bin/env python3
"""Halftone / dot-matrix asset generator for the GitHub profile README.

Every asset is drawn as a grid of dots: large type is halftoned from a font,
small labels come from a hand-built 5x7 dot-matrix alphabet, and screenshots
and dividers are drawn from the same grid.

    python3 scripts/halftone.py all

Requires Pillow and numpy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

PAPER = "#EDE7DA"
INK = "#17130F"
ACCENT = "#E0472B"
MUTED = "#8A8172"
HAIRLINE = "#CFC6B4"

def resolve_font(candidates: list[str]) -> str:
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError("none of these fonts exist: " + ", ".join(candidates))


SANS_BOLD = resolve_font([
    "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansCondensed-Bold.ttf",
])

# Classic 5x7 dot-matrix glyphs, one string per row of five cells.
FONT_5X7 = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "·": ("00000", "00000", "00000", "01100", "01100", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "#": ("01010", "01010", "11111", "01010", "11111", "01010", "01010"),
    "%": ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
}


# ---------------------------------------------------------------- dot-matrix


def bitmap_layout(text: str, cell: int, tracking: int = 1) -> tuple[list[tuple[int, int]], int, int]:
    """Return dot cells (col, row) plus the rendered width and height in px."""
    dots: list[tuple[int, int]] = []
    cursor = 0
    for char in text.upper():
        glyph = FONT_5X7.get(char, FONT_5X7[" "])
        for row, line in enumerate(glyph):
            for col, bit in enumerate(line):
                if bit == "1":
                    dots.append((cursor + col, row))
        cursor += 5 + tracking
    width = max(cursor - tracking, 0) * cell
    return dots, width, 7 * cell


def bitmap_svg(text: str, cell: int, x: float, y: float, color: str,
               tracking: int = 1, radius: float = 0.34, opacity: float = 1.0) -> str:
    dots, _, _ = bitmap_layout(text, cell, tracking)
    r = cell * radius
    body = "".join(
        f'<circle cx="{x + (col + 0.5) * cell:.1f}" cy="{y + (row + 0.5) * cell:.1f}" r="{r:.1f}"/>'
        for col, row in dots
    )
    alpha = f' opacity="{opacity}"' if opacity < 1 else ""
    return f'<g fill="{color}"{alpha}>{body}</g>'


def bitmap_size(text: str, cell: int, tracking: int = 1) -> tuple[int, int]:
    _, width, height = bitmap_layout(text, cell, tracking)
    return width, height


# ---------------------------------------------------------------- raster masks


def text_mask(text: str, font_path: str, size: int, tracking: float = 0.0) -> np.ndarray:
    """Render text to a grayscale mask, cropped tight to its ink."""
    font = ImageFont.truetype(font_path, size)
    widths = [font.getlength(char) + tracking for char in text]
    width = int(math.ceil(sum(widths) - tracking)) + 4
    ascent, descent = font.getmetrics()
    canvas = Image.new("L", (width, ascent + descent + 4), 0)
    draw = ImageDraw.Draw(canvas)
    x = 2.0
    for char, advance in zip(text, widths):
        draw.text((x, 2), char, font=font, fill=255)
        x += advance
    arr = np.array(canvas)
    rows = np.where(arr.max(axis=1) > 8)[0]
    cols = np.where(arr.max(axis=0) > 8)[0]
    if len(rows) == 0 or len(cols) == 0:
        return arr
    return arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def downsample(mask: np.ndarray, cell: int, invert: bool = False) -> np.ndarray:
    height, width = mask.shape
    rows, cols = height // cell, width // cell
    if rows == 0 or cols == 0:
        raise ValueError(f"mask {width}x{height} is smaller than one {cell}px cell")
    grid = mask[: rows * cell, : cols * cell].reshape(rows, cell, cols, cell).mean(axis=(1, 3)) / 255.0
    return 1.0 - grid if invert else grid


def radial_mask(size: int, radius: float, power: float = 1.0) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    centre = size / 2
    dist = np.sqrt((x - centre) ** 2 + (y - centre) ** 2) / radius
    return np.clip(1.0 - dist, 0.0, 1.0) ** power


def dots(grid: np.ndarray, cell: int, ox: float, oy: float, max_radius: float,
         gamma: float = 1.0, floor: float = 0.12) -> str:
    parts = []
    rows, cols = grid.shape
    for y in range(rows):
        for x in range(cols):
            value = float(grid[y, x]) ** gamma
            if value <= floor:
                continue
            radius = max_radius * math.sqrt(value)
            if radius < 0.35:
                continue
            parts.append(
                f'<circle cx="{ox + x * cell + cell / 2:.1f}" cy="{oy + y * cell + cell / 2:.1f}" r="{radius:.1f}"/>'
            )
    return "".join(parts)


def svg_open(width: int, height: int, background: str | None) -> list[str]:
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    ]
    if background:
        head.append(f'<rect width="{width}" height="{height}" fill="{background}"/>')
    return head


def write_svg(name: str, parts: list[str]) -> Path:
    parts.append("</svg>")
    path = ASSETS / name
    path.write_text("".join(parts))
    return path


# ---------------------------------------------------------------- compositions


def build_banner() -> Path:
    width, height = 1280, 400
    parts = svg_open(width, height, PAPER)

    # halftone sphere, with a dashed registration ring
    sphere = radial_mask(340, 168, power=1.25)
    grid = downsample((sphere * 255).astype(np.uint8), 8)
    parts.append(f'<g fill="{ACCENT}">{dots(grid, 8, 900, 30, 4.6, gamma=0.85, floor=0.06)}</g>')
    parts.append(
        f'<circle cx="1076" cy="200" r="178" fill="none" stroke="{INK}" '
        f'stroke-width="1.5" stroke-dasharray="2 7" opacity="0.75"/>'
    )

    parts.append(bitmap_svg("PORTFOLIO / 2026", 3, 64, 38, ACCENT, tracking=1))

    for index, line in enumerate(("DEVANK", "GUPTA")):
        mask = text_mask(line, SANS_BOLD, 132, tracking=2)
        cell = 7
        grid = downsample(mask, cell)
        parts.append(f'<g fill="{INK}">{dots(grid, cell, 62, 86 + index * 108, 4.1, gamma=1.25, floor=0.1)}</g>')

    parts.append(bitmap_svg("LOCAL-FIRST SOFTWARE / AGENTIC TOOLING", 3, 64, 336, INK, tracking=1))

    for cx, cy in ((44, 40), (1236, 40), (44, 360), (1236, 360)):
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="2.2" fill="{MUTED}"/>')

    return write_svg("banner.svg", parts)


def build_heading(text: str, name: str, cell: int = 5) -> Path:
    label_width, label_height = bitmap_size(text, cell)
    pad_left, pad_y = 46, 13
    width, height = label_width + pad_left + 12, label_height + pad_y * 2

    parts = svg_open(width, height, None)
    parts.append(f'<circle cx="16" cy="{height / 2:.1f}" r="7" fill="{ACCENT}"/>')
    parts.append(
        f'<circle cx="16" cy="{height / 2:.1f}" r="12" fill="none" stroke="{ACCENT}" '
        f'stroke-width="1" opacity="0.4"/>'
    )
    parts.append(bitmap_svg(text, cell, pad_left, pad_y, INK))
    return write_svg(name, parts)


def build_divider(name: str = "divider.svg", width: int = 1200) -> Path:
    height, cell = 24, 6
    parts = svg_open(width, height, None)
    circles = []
    for index in range(width // cell):
        t = index / max(width // cell - 1, 1)
        radius = 1.0 + 2.1 * math.sin(math.pi * t) ** 0.55
        circles.append(f'<circle cx="{index * cell + cell / 2:.1f}" cy="{height / 2:.1f}" r="{radius:.1f}"/>')
    parts.append(f'<g fill="{MUTED}">{"".join(circles)}</g>')
    return write_svg(name, parts)


def build_stats() -> Path:
    data = json.loads((ASSETS / "stats.json").read_text())
    width, height = 1100, 300
    parts = svg_open(width, height, PAPER)

    parts.append(bitmap_svg("GITHUB // SNAPSHOT", 3, 44, 30, ACCENT))
    parts.append(f'<line x1="44" y1="76" x2="{width - 44}" y2="76" stroke="{HAIRLINE}" '
                 f'stroke-width="2" stroke-dasharray="1 6"/>')

    metrics = [
        ("REPOS", str(data["repos"])),
        ("STARS", str(data["stars"])),
        ("FOLLOWERS", str(data["followers"])),
    ]
    x = 44
    for label, value in metrics:
        parts.append(bitmap_svg(value, 8, x, 108, INK))
        parts.append(bitmap_svg(label, 2, x + 2, 186, MUTED))
        x += 170

    divider_x = 566
    parts.append(f'<line x1="{divider_x}" y1="104" x2="{divider_x}" y2="216" stroke="{HAIRLINE}" '
                 f'stroke-width="2" stroke-dasharray="1 6"/>')

    parts.append(bitmap_svg("LANGUAGES", 3, 610, 104, ACCENT))
    languages = data["languages"][:3]
    y = 148
    total = sum(item["percent"] for item in languages) or 1
    for index, item in enumerate(languages):
        share = round(item["percent"] / total * 100)
        color = INK if index == 0 else ACCENT if index == 1 else MUTED
        parts.append(bitmap_svg(item["name"].upper(), 2, 610, y, color))
        parts.append(bitmap_svg(f"{share}%", 2, 1032, y, MUTED))

        filled = max(1, round(share / 100 * 30))
        for cell_index in range(30):
            radius = 3.0 if cell_index < filled else 1.1
            opacity = 1 if cell_index < filled else 0.28
            parts.append(
                f'<circle cx="{750 + cell_index * 9:.1f}" cy="{y + 7:.1f}" r="{radius}" '
                f'fill="{color}" opacity="{opacity}"/>'
            )
        y += 44

    parts.append(bitmap_svg(f"SYNCED {data['updated']}", 2, 44, 252, MUTED))
    return write_svg("stats.svg", parts)


def build_all() -> None:
    ASSETS.mkdir(exist_ok=True)
    written = [
        build_banner(),
        build_divider(),
        build_heading("ABOUT", "head-about.svg"),
        build_heading("STACK", "head-stack.svg"),
        build_heading("TELEMETRY", "head-telemetry.svg"),
        build_heading("CONTACT", "head-contact.svg"),
    ]
    if (ASSETS / "stats.json").exists():
        written.append(build_stats())

    for path in written:
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", default="all")
    args = parser.parse_args()

    ASSETS.mkdir(exist_ok=True)
    if args.target == "all":
        build_all()
    elif args.target == "banner":
        print(build_banner())
    elif args.target == "divider":
        print(build_divider())
    elif args.target == "stats":
        print(build_stats())
    else:
        parser.error(f"unknown target {args.target!r}")


if __name__ == "__main__":
    main()
