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

PALETTES = {
    "dark": {
        "paper": "#0B0908",
        "ink": "#F2ECE1",
        "accent": "#FF4A2A",
        "muted": "#9C9284",
        "hairline": "#332D26",
    },
    "light": {
        "paper": "#EDE7DA",
        "ink": "#17130F",
        "accent": "#D63A1E",
        "muted": "#6E6557",
        "hairline": "#C9BFAC",
    },
}

PAPER = PALETTES["dark"]["paper"]
INK = PALETTES["dark"]["ink"]
ACCENT = PALETTES["dark"]["accent"]
MUTED = PALETTES["dark"]["muted"]
HAIRLINE = PALETTES["dark"]["hairline"]

OUT_DIR = ASSETS


def use_theme(theme: str) -> None:
    """Swap the palette and the output folder."""
    global PAPER, INK, ACCENT, MUTED, HAIRLINE, OUT_DIR
    palette = PALETTES[theme]
    PAPER, INK, ACCENT = palette["paper"], palette["ink"], palette["accent"]
    MUTED, HAIRLINE = palette["muted"], palette["hairline"]
    OUT_DIR = ASSETS if theme == "dark" else ASSETS / theme
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def bitmap_svg_right(text: str, cell: int, right: float, y: float, color: str,
                     tracking: int = 1, radius: float = 0.34) -> str:
    width, _ = bitmap_size(text, cell, tracking)
    return bitmap_svg(text, cell, right - width, y, color, tracking, radius)


def grain(width: int, height: int, count: int, color: str, opacity: float, seed: int) -> str:
    """Sparse noise, like ink sitting unevenly on the sheet."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, width, count)
    ys = rng.uniform(0, height, count)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1"/>' for x, y in zip(xs, ys))
    return f'<g fill="{color}" opacity="{opacity}">{dots}</g>'


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
    path = OUT_DIR / name
    path.write_text("".join(parts))
    return path


# ---------------------------------------------------------------- compositions


def build_banner() -> Path:
    width, height = 1400, 500
    parts = svg_open(width, height, PAPER)
    parts.append(grain(width, height, 1100, INK, 0.05, seed=11))
    parts.append(f'<rect x="0" y="0" width="6" height="{height}" fill="{ACCENT}"/>')

    parts.append(bitmap_svg("SPECIMEN / 01", 3, 64, 40, ACCENT))
    parts.append(bitmap_svg_right("MMXXVI", 3, width - 64, 40, MUTED))

    for index, line in enumerate(("DEVANK", "GUPTA")):
        cell = 9
        grid = downsample(text_mask(line, SANS_BOLD, 170, tracking=3), cell)
        y = 96 + index * 122
        parts.append(
            f'<g fill="{ACCENT}" opacity="0.78">'
            f'{dots(grid, cell, 73, y + 7, 4.6, gamma=1.25, floor=0.1)}</g>'
        )
        parts.append(f'<g fill="{INK}">{dots(grid, cell, 68, y, 4.6, gamma=1.25, floor=0.1)}</g>')

    parts.append(bitmap_svg("LOCAL-FIRST SOFTWARE / AGENTIC TOOLING", 3, 68, 378, INK))

    parts.append(f'<line x1="900" y1="96" x2="900" y2="392" stroke="{HAIRLINE}" stroke-width="1"/>')
    for tick in range(0, 297, 24):
        parts.append(f'<line x1="893" y1="{96 + tick}" x2="900" y2="{96 + tick}" stroke="{HAIRLINE}" stroke-width="1"/>')
    parts.append(bitmap_svg("45 LPI", 2, 916, 140, MUTED))
    parts.append(bitmap_svg("2 INKS", 2, 916, 182, MUTED))
    parts.append(bitmap_svg("NO.001", 2, 916, 224, MUTED))

    size = 470
    sphere = radial_mask(size, 232, power=1.2)
    grid = downsample((sphere * 255).astype(np.uint8), 10)
    parts.append(
        f'<g fill="{ACCENT}">{dots(grid, 10, 1330 - size // 2, 322 - size // 2, 5.4, gamma=0.85, floor=0.05)}</g>'
    )
    parts.append(
        f'<circle cx="1330" cy="322" r="250" fill="none" stroke="{INK}" stroke-width="1.2" '
        f'stroke-dasharray="2 8" opacity="0.55"/>'
    )

    parts.append(f'<line x1="64" y1="430" x2="1000" y2="430" stroke="{HAIRLINE}" stroke-width="1"/>')
    for x in range(64, 999, 44):
        parts.append(f'<line x1="{x}" y1="430" x2="{x}" y2="438" stroke="{HAIRLINE}" stroke-width="1"/>')
    parts.append(bitmap_svg("HALFTONE / DOT MATRIX", 2, 64, 452, MUTED))

    return write_svg("banner.svg", parts)


def build_section(number: str, title: str, name: str, total: str = "04") -> Path:
    width, height = 1400, 92
    parts = svg_open(width, height, PAPER)
    parts.append(bitmap_svg(number, 4, 64, 34, ACCENT))
    parts.append(bitmap_svg(title, 4, 154, 34, INK))

    title_width = bitmap_size(title, 4)[0]
    x = 154 + title_width + 26
    while x < width - 220:
        parts.append(f'<circle cx="{x}" cy="48" r="1.8" fill="{HAIRLINE}"/>')
        x += 11
    parts.append(bitmap_svg_right(f"{number} / {total}", 2, width - 64, 41, MUTED))
    return write_svg(name, parts)


def build_stats() -> Path:
    data = json.loads((ASSETS / "stats.json").read_text())
    width, height = 1400, 318
    parts = svg_open(width, height, PAPER)
    parts.append(grain(width, height, 500, INK, 0.04, seed=5))

    parts.append(bitmap_svg("GITHUB // SNAPSHOT", 3, 40, 28, ACCENT))
    parts.append(bitmap_svg_right(f"SYNCED {data['updated']}", 2, width - 40, 36, MUTED))
    parts.append(f'<line x1="40" y1="76" x2="{width - 40}" y2="76" stroke="{HAIRLINE}" stroke-width="1"/>')

    metrics = [
        ("REPOS", data["repos"]),
        ("STARS", data["stars"]),
        ("FOLLOWERS", data["followers"]),
    ]
    x = 40
    for label, value in metrics:
        parts.append(bitmap_svg(str(value), 10, x, 104, INK))
        parts.append(bitmap_svg(label, 2, x + 2, 202, MUTED))
        x += 260

    divider_x = 830
    parts.append(
        f'<line x1="{divider_x}" y1="98" x2="{divider_x}" y2="244" stroke="{HAIRLINE}" '
        f'stroke-width="1" stroke-dasharray="2 7"/>'
    )

    parts.append(bitmap_svg("LANGUAGES", 3, 870, 100, ACCENT))
    languages = data["languages"][:3]
    total = sum(item["percent"] for item in languages) or 1
    y = 146
    for index, item in enumerate(languages):
        share = round(item["percent"] / total * 100)
        color = INK if index == 0 else ACCENT if index == 1 else MUTED
        parts.append(bitmap_svg(item["name"].upper(), 2, 870, y, color))
        parts.append(bitmap_svg_right(f"{share}%", 2, width - 40, y, MUTED))

        filled = max(1, round(share / 100 * 30))
        for cell_index in range(30):
            radius = 3.0 if cell_index < filled else 1.1
            opacity = 1 if cell_index < filled else 0.24
            parts.append(
                f'<circle cx="{1020 + cell_index * 9:.1f}" cy="{y + 7:.1f}" r="{radius}" '
                f'fill="{color}" opacity="{opacity}"/>'
            )
        y += 46

    for index in range(0, width - 80, 12):
        t = index / (width - 80)
        radius = 1.2 + 2.2 * abs(math.sin(t * math.pi * 3))
        parts.append(
            f'<circle cx="{40 + index}" cy="292" r="{radius:.1f}" fill="{ACCENT}" opacity="0.55"/>'
        )

    return write_svg("stats.svg", parts)


def build_colophon() -> Path:
    width, height = 1400, 124
    parts = svg_open(width, height, PAPER)

    parts.append(bitmap_svg("COLOPHON", 3, 40, 22, ACCENT))
    parts.append(bitmap_svg_right("EDITION 2026 / NO. 001", 2, width - 40, 29, MUTED))
    parts.append(f'<line x1="40" y1="58" x2="{width - 40}" y2="58" stroke="{HAIRLINE}" stroke-width="1"/>')

    for index, color in enumerate((INK, ACCENT, MUTED)):
        parts.append(f'<circle cx="{52 + index * 26}" cy="88" r="8" fill="{color}"/>')
    parts.append(bitmap_svg("PALETTE", 2, 40, 104, MUTED))

    parts.append(bitmap_svg("DEJAVU SANS CONDENSED / 5X7 DOT MATRIX / TWO INKS", 3, 200, 80, INK))

    return write_svg("colophon.svg", parts)


def build_all() -> None:
    written = [
        build_banner(),
        build_section("01", "ABOUT", "sec-about.svg"),
        build_section("02", "STACK", "sec-stack.svg"),
        build_section("03", "TELEMETRY", "sec-telemetry.svg"),
        build_section("04", "CONTACT", "sec-contact.svg"),
        build_colophon(),
    ]
    if (ASSETS / "stats.json").exists():
        written.append(build_stats())

    for path in written:
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", default="all")
    parser.add_argument("--theme", choices=["dark", "light", "both"], default="both")
    args = parser.parse_args()

    ASSETS.mkdir(exist_ok=True)
    themes = ["dark", "light"] if args.theme == "both" else [args.theme]

    for theme in themes:
        use_theme(theme)
        print(f"[{theme}]", end=" ")
        if args.target == "all":
            build_all()
        elif args.target == "banner":
            print(build_banner())
        elif args.target == "stats":
            print(build_stats())
        else:
            parser.error(f"unknown target {args.target!r}")


if __name__ == "__main__":
    main()
