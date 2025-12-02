from datetime import datetime
import io
import logging
import os
import math

from flask import send_file
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim
import matplotlib

# Use a non-interactive backend for servers without display support
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from PIL import ImageDraw  # noqa: E402
from PIL import ImageFont  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    SimpleDocTemplate,
)
import requests
from io import BytesIO
import numpy as np
import traceback
from typing import Optional

import google.generativeai as genai  # noqa: E402


logger = logging.getLogger(__name__)

FOOTER_TEXT = "krishizest powered by TerrAqua UAV"

def crop_to_content(img: PILImage.Image, padding: int = 20) -> PILImage.Image:
    """
    Aggressively crops the image to the non-transparent content.
    This ensures the field appears large in the PDF.
    """
    try:
        # Ensure we are working with alpha channel
        img = img.convert("RGBA")
        alpha = img.split()[3]
        
        # Get bounding box of non-transparent pixels
        bbox = alpha.getbbox()
        
        # If no alpha bbox found, try finding difference from white/black
        if not bbox:
            bg = PILImage.new(img.mode, img.size, img.getpixel((0,0)))
            diff = ImageChops.difference(img, bg)
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox()

        if bbox:
            left, upper, right, lower = bbox
            width, height = img.size
            
            # Add padding without going out of bounds
            left = max(0, left - padding)
            upper = max(0, upper - padding)
            right = min(width, right + padding)
            lower = min(height, lower + padding)
            
            return img.crop((left, upper, right, lower))
            
        return img
    except Exception as e:
        logger.warning(f"Cropping failed: {e}")
        return img


def process_map_image(image_buffer: BytesIO, target_width: int = 1600) -> BytesIO:
    """
    Standardizes map images: 
    1. Crops to content (removes empty space).
    2. Adds White Background (removes black box).
    3. Resizes for high-quality print.
    """
    try:
        image_buffer.seek(0)
        img = PILImage.open(image_buffer)
        
        # 1. Crop to the actual field
        img = crop_to_content(img)
        
        # 2. Handle Transparency -> Composite on White
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            background = PILImage.new("RGB", img.size, (255, 255, 255))
            # Paste using alpha channel as mask
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert("RGB")

        # 3. Resize (High Quality)
        w_percent = (target_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        
        # Don't upscale if it's already smaller than target to avoid blur, 
        # unless it's tiny (less than 300px)
        if img.size[0] > target_width or img.size[0] < 300:
             img = img.resize((target_width, h_size), PILImage.LANCZOS)

        # 4. Save
        out_buffer = BytesIO()
        img.save(out_buffer, format='PNG', optimize=True)
        out_buffer.seek(0)
        return out_buffer

    except Exception as e:
        logger.error(f"Error processing map image: {e}")
        return image_buffer


class HR(Flowable):
    """Helper flowable used for drawing horizontal rules."""

    def __init__(self, thickness: int = 1):
        super().__init__()
        self.thickness = thickness

    def draw(self) -> None:
        width = self.canv._pagesize[0] - 40 * mm  # type: ignore[attr-defined]
        self.canv.setLineWidth(self.thickness)
        self.canv.line(20 * mm, 0, 20 * mm + width, 0)


def draw_footer(canvas, doc) -> None:
    """Draw footer and page border on each page."""

    canvas.saveState()

    # Border
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(0.5)
    margin = 10 * mm
    canvas.rect(
        margin,
        margin,
        doc.pagesize[0] - 2 * margin,
        doc.pagesize[1] - 2 * margin,
        stroke=1,
        fill=0,
    )

    # Footer label
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(doc.pagesize[0] / 2.0, 12 * mm, FOOTER_TEXT)

    canvas.restoreState()


def create_high_quality_tile_mosaic(aoi, start_date, end_date, layer_type, 
                                   backend_url="http://localhost:5000", 
                                   zoom=15, grid_size=3, final_size=2400):
    """
    Create a HIGH-QUALITY tile mosaic with:
    - Higher zoom level (15-16 instead of 14)
    - Larger grid for better coverage
    - Higher resolution output
    - Better stitching and cropping
    """
    try:
        logger.info(f"Creating HIGH-QUALITY {layer_type} tile mosaic...")
        
        # Get tile URL template from backend
        response = requests.post(
            f"{backend_url}/get_gee_tile",
            json={
                'aoi': aoi,
                'start_date': start_date,
                'end_date': end_date,
                'layer': layer_type
            },
            timeout=60
        )
        
        if response.status_code != 200:
            logger.warning(f"Failed to get {layer_type} tile template")
            return None, None
        
        data = response.json()
        tile_url_template = data.get('urlFormat')
        stats = data.get('stats')
        
        if not tile_url_template:
            return None, None
        
        # Calculate AOI center
        coords = aoi['geometry']['coordinates'][0] if isinstance(aoi['geometry']['coordinates'][0][0], list) else aoi['geometry']['coordinates']
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        
        # Convert to tile coordinates
        center_x, center_y = lonlat_to_tile_xy(center_lon, center_lat, zoom)
        
        # Fetch tiles in a grid
        tile_size = 256  # Standard tile size
        tiles = {}
        successful = 0
        
        radius = grid_size // 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                tile_x = int(center_x) + dx
                tile_y = int(center_y) + dy
                
                try:
                    tile_url = tile_url_template.replace('{x}', str(tile_x)).replace('{y}', str(tile_y)).replace('{z}', str(zoom))
                    if '{s}' in tile_url:
                        tile_url = tile_url.replace('{s}', 'a')
                    
                    tile_response = requests.get(tile_url, timeout=15)
                    
                    if tile_response.status_code == 200:
                        tile_img = PILImage.open(BytesIO(tile_response.content)).convert('RGBA')
                        tiles[(dx, dy)] = tile_img
                        successful += 1
                        
                except Exception as e:
                    logger.debug(f"Failed to fetch tile ({tile_x}, {tile_y}): {e}")
                    continue
        
        if not tiles:
            logger.warning(f"No tiles fetched for {layer_type}")
            return None, None
        
        logger.info(f"Successfully fetched {successful} tiles for {layer_type}")
        
        # Create mosaic canvas
        mosaic_width = grid_size * tile_size
        mosaic_height = grid_size * tile_size
        mosaic = PILImage.new('RGBA', (mosaic_width, mosaic_height), (255, 255, 255, 0))
        
        # Stitch tiles
        for (dx, dy), tile_img in tiles.items():
            x_pos = (dx + radius) * tile_size
            y_pos = (dy + radius) * tile_size
            mosaic.paste(tile_img, (x_pos, y_pos))
        
        # Crop to content (remove transparent areas)
        mosaic = crop_to_content(mosaic, padding=10)
        
        # Convert transparency to white background
        if mosaic.mode == 'RGBA':
            background = PILImage.new('RGB', mosaic.size, (255, 255, 255))
            background.paste(mosaic, mask=mosaic.split()[3])
            mosaic = background
        
        # Resize to final high-quality size
        aspect_ratio = mosaic.size[0] / mosaic.size[1]
        if aspect_ratio > 1:
            final_width = final_size
            final_height = int(final_size / aspect_ratio)
        else:
            final_height = final_size
            final_width = int(final_size * aspect_ratio)
        
        mosaic = mosaic.resize((final_width, final_height), PILImage.LANCZOS)
        
        # Add colormap legend only if not already present
        # (Prevent double legend by not calling here if called elsewhere)
        # mosaic = add_colormap_legend(mosaic, layer_type)
        
        # Save to buffer with high quality and increased DPI for small areas
        buffer = BytesIO()
        # If area is small, set higher DPI
        dpi_value = 300 if mosaic.size[0] < 1200 or mosaic.size[1] < 1200 else 150
        mosaic.save(buffer, format='PNG', optimize=False, compress_level=1, dpi=(dpi_value, dpi_value))
        buffer.seek(0)
        
        logger.info(f"✅ Created high-quality {layer_type} mosaic: {final_width}x{final_height}px")
        return buffer, stats
        
    except Exception as e:
        logger.error(f"Error creating high-quality tile mosaic: {e}")
        traceback.print_exc()
        return None, None

def lonlat_to_tile_xy(lon, lat, zoom):
    """Convert longitude/latitude to tile coordinates."""
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    
    return x, y


def add_colormap_legend(mosaic_pil: PILImage, layer_type: str, title: str = None) -> PILImage:
    """
    Add a professional colormap legend and title to a tile mosaic image.
    Mimics Earth Engine visualization style with centered legend at bottom.
    """
    try:
        # Define colormaps and ranges for each layer type
        colormap_specs = {
            'ndvi': {
                'title': title or 'NDVI (Normalized Difference Vegetation Index)',
                'subtitle': 'Normalized Difference Vegetation Index (NDVI)',
                'min': -1.0,
                'max': 1.0,
                'colors': ['#FFFFFF', '#CE7E45', '#DF923D', '#F1B555', '#FCD163', '#99B718', '#74A901', '#66A000', '#529400', '#3E8601', '#207401', '#056201', '#004C00', '#023B01', '#012E01', '#011D01', '#011301'],
                'labels': ['-1.000', '-0.333', '0.333', '1.000']
            },
            'soilmoisture': {
                'title': title or 'Soil Moisture',
                'subtitle': 'Soil Moisture (%)',
                'min': 0,
                'max': 100,
                'colors': ['#8b4513', '#d2691e', '#daa520', '#90ee90', '#006400'],
                'labels': ['0', '25', '50', '75', '100']
            },
            'cwr': {
                'title': title or 'Crop Water Requirement (mm)',
                'subtitle': 'Crop Water Requirement',
                'min': 0,
                'max': 10,
                'colors': ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
                'labels': ['0', '2.5', '5', '7.5', '10']
            },
            'lst': {
                'title': title or 'Land Surface Temperature (°C)',
                'subtitle': 'Land Surface Temperature',
                'min': 0,
                'max': 50,
                'colors': ['#0000ff', '#00ffff', '#ffff00', '#ff7f00', '#ff0000'],      
                'labels': ['0°C', '25°C', '50°C']
            },
            'etc': {
                'title': title or 'Evapotranspiration (mm)',
                'subtitle': 'Evapotranspiration',
                'min': 0,
                'max': 10,
                'colors': ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494'],
                'labels': ['0', '2.5', '5', '7.5', '10']
            },
            'ndti': {
                'title': title or 'NDTI (Normalized Difference Turbidity Index)',
                'subtitle': 'Normalized Difference Turbidity Index (NDTI)',
                'min': -1.0,
                'max': 1.0,
                'colors': ['#0000ff', '#1e90ff', '#00ffff', '#ffff00', '#ff7f00', '#ff0000'],
                'labels': ['-1.000', '-0.333', '0.333', '1.000']
            }
        }
        
        spec = colormap_specs.get(layer_type, colormap_specs['ndvi'])
        
        # Create new image with space for title and legend
        mosaic_w, mosaic_h = mosaic_pil.size
        title_h = 35        # Space for main title
        subtitle_h = 20     # Space for subtitle
        legend_h = 55       # Space for legend bar and labels
        padding = 15        # Minimal padding around content
        
        new_w = mosaic_w + padding * 2
        new_h = mosaic_h + title_h + subtitle_h + legend_h + padding * 2
        
        # Create white background
        new_img = PILImage.new('RGB', (new_w, new_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(new_img)
        
        # Paste original mosaic centered (with minimal padding)
        paste_x = padding
        paste_y = title_h + subtitle_h + padding
        new_img.paste(mosaic_pil, (paste_x, paste_y))
        
        # Setup fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 16)
            subtitle_font = ImageFont.truetype("arial.ttf", 12)
            label_font = ImageFont.truetype("arial.ttf", 10)
        except:
            # Fallback to default fonts
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
        
        # Draw main title at top (centered)
        title_text = spec['title']
        if hasattr(title_font, 'getbbox'):
            bbox = draw.textbbox((0, 0), title_text, font=title_font)
            title_width = bbox[2] - bbox[0]
        else:
            title_width = len(title_text) * 10  # Rough estimate
        title_x = (new_w - title_width) // 2
        draw.text((title_x, 10), title_text, fill=(0, 0, 0), font=title_font)
        
        # Draw subtitle below title
        subtitle_y = title_h - 5
        subtitle_text = spec['subtitle']
        if hasattr(subtitle_font, 'getbbox'):
            bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
            subtitle_width = bbox[2] - bbox[0]
        else:
            subtitle_width = len(subtitle_text) * 8
        subtitle_x = (new_w - subtitle_width) // 2
        draw.text((subtitle_x, subtitle_y), subtitle_text, fill=(80, 80, 80), font=subtitle_font)
        
        # Draw colormap legend bar at bottom (centered)
        legend_y = paste_y + mosaic_h + 10
        legend_width = mosaic_w - 60  # Leave some margin
        legend_x_start = (new_w - legend_width) // 2
        legend_bar_h = 18
        
        # Draw gradient bar with border
        num_colors = len(spec['colors'])
        block_w = legend_width // num_colors
        
        for i, color_hex in enumerate(spec['colors']):
            color_rgb = tuple(int(color_hex.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
            x0 = legend_x_start + i * block_w
            y0 = legend_y
            x1 = x0 + block_w
            y1 = y0 + legend_bar_h
            draw.rectangle([x0, y0, x1, y1], fill=color_rgb, outline=(0, 0, 0), width=1)
        
        # Add value labels below legend
        label_y = legend_y + legend_bar_h + 5
        labels = spec['labels']
        
        for i, label in enumerate(labels):
            if len(labels) > 1:
                label_x = legend_x_start + (i * legend_width // (len(labels) - 1))
            else:
                label_x = legend_x_start + legend_width // 2
            
            # Center the label text
            if hasattr(label_font, 'getbbox'):
                bbox = draw.textbbox((0, 0), label, font=label_font)
                label_text_width = bbox[2] - bbox[0]
            else:
                label_text_width = len(label) * 6
                
            label_x -= label_text_width // 2
            draw.text((label_x, label_y), str(label), fill=(0, 0, 0), font=label_font)
        
        logger.info(f"✅ Added professional colormap legend to {layer_type} mosaic")
        return new_img
        
    except Exception as e:
        logger.warning(f"Could not add colormap legend: {e}")
        return mosaic_pil


def make_draw_header(location_text: str | None, aoi_label: str, left_title: str = "krishizest"):
    """Return a header drawing function capturing the location text."""

    def draw_header(canvas, doc) -> None:
        canvas.saveState()

        canvas.setFont("Helvetica-Bold", 12)
        canvas.setFillColor(colors.HexColor("#0b486b"))
        canvas.drawString(22 * mm, doc.pagesize[1] - 18 * mm, left_title)

        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.black)
        canvas.drawRightString(
            doc.pagesize[0] - 22 * mm,
            doc.pagesize[1] - 18 * mm,
            f"{aoi_label}: {location_text or 'Not provided'}",
        )

        canvas.setLineWidth(0.5)
        canvas.setStrokeColor(colors.lightgrey)
        canvas.line(20 * mm, doc.pagesize[1] - 20 * mm, doc.pagesize[0] - 20 * mm, doc.pagesize[1] - 20 * mm)

        canvas.restoreState()

    return draw_header


def download_image(url: str) -> Optional[io.BytesIO]:
    if not url: return None
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return io.BytesIO(response.content)
    except Exception as e:
        logger.error(f"Error downloading image {url}: {e}")
        return None


def _px_to_points(px: int, dpi: float) -> float:
    if not dpi or dpi <= 0: dpi = 96.0
    return px * 72.0 / dpi


def make_reportlab_image(buf: io.BytesIO, max_width_pt: float, max_height_pt: float) -> Image:
    """Create a ReportLab Image Flowable that fits into max_width_pt x max_height_pt."""
    try:
        buf.seek(0)
        pil = PILImage.open(buf)
        px_w, px_h = pil.size
        dpi = pil.info.get('dpi', (96, 96))[0] if isinstance(pil.info.get('dpi', None), tuple) else (pil.info.get('dpi', 96) or 96)
        img_w_pt = _px_to_points(px_w, dpi)
        img_h_pt = _px_to_points(px_h, dpi)

        scale = min(1.0, max_width_pt / img_w_pt if img_w_pt > 0 else 1.0, max_height_pt / img_h_pt if img_h_pt > 0 else 1.0)
        draw_w = img_w_pt * scale
        draw_h = img_h_pt * scale

        out_buf = io.BytesIO()
        pil.save(out_buf, format='PNG')
        out_buf.seek(0)
        rl_img = Image(out_buf, width=draw_w, height=draw_h)
        return rl_img
    except Exception:
        try:
            buf.seek(0)
            rl_img = Image(buf)
            rl_img._restrictSize(max_width_pt, max_height_pt)
            return rl_img
        except Exception:
            placeholder = PILImage.new('RGB', (800, 480), color=(240, 240, 240))
            pbuf = io.BytesIO()
            placeholder.save(pbuf, format='PNG')
            pbuf.seek(0)
            return Image(pbuf, width=max_width_pt, height=max_height_pt)


def get_logo_path(provided_path: str | None = None) -> str | None:
    """Resolve the logo path using several fallbacks."""
    if provided_path and os.path.exists(provided_path):
        return provided_path

    candidate_paths = [
        provided_path,
        "static/krishizest_logo.png",
        "krishizest_logo.png",
        "../static/krishizest_logo.png",
        "./static/krishizest_logo.png",
        os.path.join(os.path.dirname(__file__), "static", "krishizest_logo.png"),
        os.path.join(os.path.dirname(__file__), "krishizest_logo.png"),
    ]

    for path in candidate_paths:
        if path and os.path.exists(path):
            return path

    return None


def _render_ndvi_chart(series, outbuf: io.BytesIO, title: str = "NDVI time series", na_text: str = "NDVI not available") -> None:
    dates, vals = [], []
    for item in series:
        date_val = item.get("date") or item.get("day") or item.get("year_month")
        ndvi_val = item.get("ndvi") or item.get("NDVI") or item.get("ndvi_mean")
        try:
            if date_val and ndvi_val is not None:
                dates.append(pd.to_datetime(date_val))
                vals.append(float(ndvi_val))
        except Exception:
            continue

    if not vals:
        fig = plt.figure(figsize=(8, 2.5))
        plt.text(0.5, 0.5, na_text, ha="center", va="center")
        plt.axis("off")
        fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        outbuf.seek(0)
        return

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(dates, vals, marker="o", linewidth=1.4)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("NDVI")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    outbuf.seek(0)


def _render_pie_chart(landcover_dict, outbuf: io.BytesIO, title: str = "Vegetation vs Bare Land", no_data_label: str = "No data") -> None:
    labels = list(landcover_dict.keys()) if landcover_dict else [no_data_label]
    sizes = [float(v) for v in landcover_dict.values()] if landcover_dict else [1]
    total = sum(sizes)
    if total == 0:
        labels, sizes = [no_data_label], [1]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", textprops={"fontsize": 8})
    ax.set_title(title, fontsize=10)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    outbuf.seek(0)


def _render_aoi_map(aoi_coords, outbuf: io.BytesIO, title: str = "AOI Map", na_text: str = "AOI not available") -> None:
    try:
        polygon = aoi_coords[0] if isinstance(aoi_coords[0][0], (list, tuple)) else aoi_coords
        lons = [p[0] for p in polygon]
        lats = [p[1] for p in polygon]

        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.fill(lons, lats, facecolor="#98FB98", edgecolor="darkgreen", alpha=0.6)
        ax.plot(lons + [lons[0]], lats + [lats[0]], color="darkgreen")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)
        ax.set_xlim(min(lons) - 0.001, max(lons) + 0.001)
        ax.set_ylim(min(lats) - 0.001, max(lats) + 0.001)
        fig.tight_layout()
        fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        outbuf.seek(0)
    except Exception:
        img = PILImage.new("RGB", (800, 480), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((200, 220), na_text, fill=(0, 0, 0))
        img.save(outbuf, format="PNG")
        outbuf.seek(0)


def _render_irrigation_timeline(calendar_data, outbuf: io.BytesIO, title: str = "Irrigation Timeline") -> None:
    """Render irrigation timeline chart with amounts and priorities."""
    try:
        if not calendar_data:
            fig = plt.figure(figsize=(8, 3))
            plt.text(0.5, 0.5, "No irrigation data available", ha="center", va="center")
            plt.axis("off")
            fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            outbuf.seek(0)
            return

        dates, amounts, priorities, rains = [], [], [], []
        for entry in calendar_data:
            try:
                date_val = entry.get("date")
                if date_val:
                    dates.append(pd.to_datetime(date_val))
                    amounts.append(float(entry.get("final_irrigation_mm", 0)))
                    priorities.append(entry.get("priority", "LOW"))
                    rains.append(float(entry.get("rain_mm", 0)))
            except Exception:
                continue

        if not dates:
            fig = plt.figure(figsize=(8, 3))
            plt.text(0.0, 0.5, "No valid irrigation dates", ha="center", va="center")
            plt.axis("off")
            fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            outbuf.seek(0)
            return

        color_map = {"URGENT": "#d32f2f", "HIGH": "#f57c00", "MEDIUM": "#fbc02d", "LOW": "#388e3c"}
        colors_list = [color_map.get(p, "#cccccc") for p in priorities]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.bar(dates, amounts, color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel("Irrigation (mm)")
        ax1.set_title(title, fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        legend_elements = [mpatches.Patch(facecolor=color_map[p], label=p, edgecolor='black') 
                          for p in ["URGENT", "HIGH", "MEDIUM", "LOW"]]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

        ax2.bar(dates, rains, color='#1976d2', alpha=0.6, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel("Rain (mm)", fontsize=9)
        ax2.set_xlabel("Date")
        ax2.grid(axis='y', alpha=0.3)
        
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        outbuf.seek(0)

    except Exception as e:
        logger.error(f"Error rendering irrigation timeline: {e}")
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "Error rendering chart", ha="center", va="center")
        plt.axis("off")
        fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        outbuf.seek(0)


def _render_priority_distribution(calendar_data, outbuf: io.BytesIO, title: str = "Irrigation Priority Distribution") -> None:
    """Render pie chart showing distribution of irrigation priorities."""
    try:
        if not calendar_data:
            fig = plt.figure(figsize=(5, 4))
            plt.text(0.5, 0.5, "No irrigation data", ha="center", va="center")
            plt.axis("off")
            fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            outbuf.seek(0)
            return

        priority_counts = {"URGENT": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for entry in calendar_data:
            if entry.get("should_irrigate"):
                priority = entry.get("priority", "LOW")
                if priority in priority_counts:
                    priority_counts[priority] += 1

        labels = [k for k, v in priority_counts.items() if v > 0]
        sizes = [v for v in priority_counts.values() if v > 0]

        if not sizes:
            fig = plt.figure(figsize=(5, 4))
            plt.text(0.5, 0.5, "No irrigation events", ha="center", va="center")
            plt.axis("off")
            fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            outbuf.seek(0)
            return

        color_map = {"URGENT": "#d32f2f", "HIGH": "#f57c00", "MEDIUM": "#fbc02d", "LOW": "#388e3c"}
        colors_list = [color_map.get(label, "#cccccc") for label in labels]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors_list, 
               textprops={"fontsize": 9, "weight": "bold"}, startangle=90)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis("equal")
        fig.tight_layout()
        fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        outbuf.seek(0)

    except Exception as e:
        logger.error(f"Error rendering priority distribution: {e}")
        fig = plt.figure(figsize=(5, 4))
        plt.text(0.5, 0.5, "Error rendering chart", ha="center", va="center")
        plt.axis("off")
        fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        outbuf.seek(0)


def _render_water_comparison(calendar_data, outbuf: io.BytesIO, title: str = "Water Need vs Rainfall") -> None:
    """Render comparison chart of irrigation need vs rainfall."""
    try:
        if not calendar_data:
            fig = plt.figure(figsize=(8, 3))
            plt.text(0.5, 0.5, "No data available", ha="center", va="center")
            plt.axis("off")
            fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            outbuf.seek(0)
            return

        dates, irrigation, rainfall = [], [], []
        for entry in calendar_data:
            try:
                date_val = entry.get("date")
                if date_val:
                    dates.append(pd.to_datetime(date_val))
                    irrigation.append(float(entry.get("final_irrigation_mm", 0)))
                    rainfall.append(float(entry.get("rain_mm", 0)))
            except Exception:
                continue

        if not dates:
            fig = plt.figure(figsize=(8, 3))
            plt.text(0.5, 0.5, "No valid data", ha="center", va="center")
            plt.axis("off")
            fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            outbuf.seek(0)
            return

        fig, ax = plt.subplots(figsize=(8, 3.5))
        
        width = 0.35
        x = range(len(dates))
        
        ax.bar([i - width/2 for i in x], irrigation, width, label='Irrigation Need', 
               color='#ff6b6b', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar([i + width/2 for i in x], rainfall, width, label='Rainfall', 
               color='#4ecdc4', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel("Water (mm)")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime('%m/%d') for d in dates], rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        outbuf.seek(0)

    except Exception as e:
        logger.error(f"Error rendering water comparison: {e}")
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "Error rendering chart", ha="center", va="center")
        plt.axis("off")
        fig.savefig(outbuf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        outbuf.seek(0)


def _extract_series_data(series, title_key: str):
    """Extracts series data based on a normalized key (e.g., 'ndvi', 'cwr')."""
    key_map = {
        "ndvi": ["ndvi", "NDVI", "ndvi_mean"],
        "cwr": ["cwr_mm", "cwr"],
        "kc": ["kc"],
        "lst": ["lst_c", "lst"],
        "etc": ["etc_mm", "etc"],
        "deltas": ["deltas_mm", "deltas"],
        "soilmoisture": ["soilmoisture_mm", "soilmoisture", "soil_moisture"],
    }

    keys_to_try = key_map.get(title_key, [])

    values = []
    for item in series:
        value = None
        for key in keys_to_try:
            if item.get(key) is not None:
                value = item.get(key)
                break
        try:
            if value is not None:
                values.append(float(value))
        except Exception:
            continue
    return values


def get_translations(lang: str = "en") -> dict:
    """
    Returns a dictionary of translated strings for the report.
    """
    translations = {
        "en": {
            "report_title": "Crop Health Report",
            "logo_alt": "Krishizest Agricultural Intelligence",
            "crop": "Crop",
            "period": "Period",
            "location": "Location",
            "not_specified": "Not specified",
            "aoi_title": "Area of Interest (AOI)",
            "aoi_label": "AOI",
            "aoi_map_title": "AOI (user-drawn)",
            "aoi_na": "AOI not provided",
            "aoi_area": "AOI Area (ha)",
            "time_period": "Time Period",
            "cloud_cover": "Cloud Cover",
            "report_generated": "Report Generated",
            "ndvi_title": "Crop Health",
            "ndvi_chart_title_suffix": "Crop Health over time",
            "ndvi_na": "NDVI time series not available",
            "cwr_title": "Crop Water Requirement",
            "kc_title": "crop Coefficient",
            "lst_title": "Field Temperature",
            "etc_title": "Total Water Loss",
            "sm_title": "Soil Moisture Change",
            "soilmoisture_title": "Soil Moisture (%)",
            "soilmoisture_avg": "<b>Average Soil Moisture: {mean:.2f}%</b>",
            "soilmoisture_trend": "<b>Soil Moisture Trend:</b> {trend}",
            "analytics_title": "Overall Crop Analytics (summary)",
            "analytics_health_na": "Crop Health data not available",
            "analytics_cwr_na": "Water Requirement not available",
            "analytics_health": "Current canopy health (NDVI mean): <b>{mean:.2f}</b>",
            "analytics_cwr": "Approx. water need (mean CWR): <b>{mean:.2f} mm/day</b>",
            "analytics_temp": "Mean field temperature: <b>{mean:.2f} °C</b>",
            "analytics_sm": "Avg. Soil Moisture Change: <b>{mean:.2f} mm</b>",
            "analytics_rec": "Recommendation: Walk the field to confirm these findings and take action.",
            "pie_title": "Field Cover: Vegetation vs Bare Land",
            "pie_interp": "Interpretation: Higher vegetation percentage indicates good canopy cover or dense weeds. Lower percentage indicates exposed soil, sparse vegetation, or very early crop stages.",
            "pie_no_data": "No data",
            "landcover_veg": "Vegetation",
            "landcover_bare": "Bare Land",
            "thanks": "Thank you",
            "contact": "For any queries, contact: contact@terraquauav.com",
            "analysis_title": "<b>{title} — Detailed Analysis:</b>",
            "analysis_no_data": "Data not available to compute a robust summary for this parameter.",
            "analysis_trend_stable": "The readings have been <b>stable</b>, starting at {start:.2f} and ending near {end:.2f}.",
            "analysis_trend_positive": "A <b>positive trend</b> was observed, starting at {start:.2f} and rising to {end:.2f}.",
            "analysis_trend_downward": "A <b>downward trend</b> was observed, starting at {start:.2f} and falling to {end:.2f}.",
            "analysis_recent_stable": "The most recent readings show <b>stability</b>.",
            "analysis_recent_improve": "The most recent readings show a <b>recent improvement</b>.",
            "analysis_recent_decline": "The most recent readings show a <b>recent decline</b>.",
            "analysis_note": "<b>Note:</b> These recommendations should be adapted based on local conditions, soil type, and specific crop requirements.",
            "ndvi_avg": "<b>Average Crop Health Index: {mean:.3f}</b>",
            "ndvi_vlow": "• <b>Very Low Health:</b> Indicates bare soil, stressed vegetation, or crop failure",
            "ndvi_low": "• <b>Low to Moderate Health:</b> Shows early stress signs or sparse vegetation",
            "ndvi_mod": "• <b>Moderate Health:</b> Average crop condition with room for improvement",
            "ndvi_good": "• <b>Good Health:</b> Healthy vegetation with strong photosynthetic activity",
            "ndvi_exc": "• <b>Excellent Health:</b> Optimal crop condition with dense, vigorous vegetation",
            "ndvi_trend": "<b>Historical Trend:</b> {trend}",
            "ndvi_recent": "<b>Recent Pattern:</b> {trend}",
            "cwr_avg": "<b>Average Daily Water Need: {mean:.2f} mm/day</b>",
            "cwr_trend": "<b>Water Need Trend:</b> {trend}",
            "kc_avg": "<b>Average Crop Coefficient (Kc): {mean:.2f}</b>",
            "kc_trend": "<b>Growth Stage Trend:</b> {trend}",
            "lst_avg": "<b>Average Field Temperature: {mean:.1f}°C</b>",
            "lst_trend": "<b>Temperature Trend:</b> {trend}",
            "etc_avg": "<b>Average Total Water Loss: {mean:.2f} mm/day</b>",
            "etc_trend": "<b>Water Loss Trend:</b> {trend}",
            "sm_avg": "<b>Average Daily Soil Moisture Change: {mean:.2f} mm</b>",
            "sm_trend": "<b>Soil Moisture Trend:</b> {trend}",
            "default_avg": "<b>Average value: {mean:.2f}.</b>",
            "default_trend": "<b>Trend Analysis:</b> {trend}",
            "irrigation_title": "Smart Irrigation Schedule",
            "irrigation_summary_title": "Irrigation Summary",
            "irrigation_timeline_title": "Irrigation Timeline",
            "irrigation_priority_title": "Priority Distribution",
            "irrigation_comparison_title": "Water Need vs Rainfall Comparison",
            "irrigation_calendar_title": "Irrigation Calendar",
            "irrigation_total_events": "Total Irrigation Events",
            "irrigation_total_water": "Total Water Required",
            "irrigation_water_saved": "Water Saved (Smart Scheduling)",
            "irrigation_savings_percent": "Water Savings",
            "irrigation_urgent": "Urgent Events",
            "irrigation_high": "High Priority Events",
            "irrigation_medium": "Medium Priority Events",
            "irrigation_date": "Date",
            "irrigation_amount": "Amount (mm)",
            "irrigation_priority": "Priority",
            "irrigation_advice": "Advice",
            "irrigation_no_data": "No irrigation data available",
        },
    }
    return translations.get(lang, translations["en"])


def generate_plain_language_explanation(
    title_key: str,
    title_text: str,
    metric: dict,
    series: list,
    lang_strings: dict,
    max_chars: int = 4500,
) -> str:
    """Generates the rule-based explanation in the selected language."""

    parts: list[str] = []
    mean = None
    try:
        if metric and isinstance(metric, dict):
            mean = metric.get("mean")
    except Exception:
        mean = None

    series_vals = _extract_series_data(series, title_key)
    trend_desc = ""
    recent_trend = ""
    if len(series_vals) >= 2:
        change = series_vals[-1] - series_vals[0]
        if abs(change) < 0.02:
            trend_desc = lang_strings["analysis_trend_stable"].format(start=series_vals[0], end=series_vals[-1])
        elif change > 0:
            trend_desc = lang_strings["analysis_trend_positive"].format(start=series_vals[0], end=series_vals[-1])
        else:
            trend_desc = lang_strings["analysis_trend_downward"].format(start=series_vals[0], end=series_vals[-1])

        if len(series_vals) >= 3:
            recent_change = series_vals[-1] - series_vals[-3]
            if abs(recent_change) < 0.01:
                recent_trend = lang_strings["analysis_recent_stable"]
            elif recent_change > 0:
                recent_trend = lang_strings["analysis_recent_improve"]
            else:
                recent_trend = lang_strings["analysis_recent_decline"]

    parts.append(lang_strings["analysis_title"].format(title=title_text))

    if mean is None:
        parts.append(lang_strings["analysis_no_data"])
    else:
        if title_key == "ndvi":
            parts.append(lang_strings["ndvi_avg"].format(mean=mean))
            if mean < 0.2:
                parts.append(lang_strings["ndvi_vlow"])
            elif mean < 0.4:
                parts.append(lang_strings["ndvi_low"])
            elif mean < 0.6:
                parts.append(lang_strings["ndvi_mod"])
            elif mean < 0.8:
                parts.append(lang_strings["ndvi_good"])
            else:
                parts.append(lang_strings["ndvi_exc"])
            if trend_desc:
                parts.append(lang_strings["ndvi_trend"].format(trend=trend_desc))
            if recent_trend:
                parts.append(lang_strings["ndvi_recent"].format(trend=recent_trend))
        elif title_key == "cwr":
            parts.append(lang_strings["cwr_avg"].format(mean=mean))
            if trend_desc:
                parts.append(lang_strings["cwr_trend"].format(trend=trend_desc))
        elif title_key == "kc":
            parts.append(lang_strings["kc_avg"].format(mean=mean))
            if trend_desc:
                parts.append(lang_strings["kc_trend"].format(trend=trend_desc))
        elif title_key == "lst":
            parts.append(lang_strings["lst_avg"].format(mean=mean))
            if trend_desc:
                parts.append(lang_strings["lst_trend"].format(trend=trend_desc))
        elif title_key == "etc":
            parts.append(lang_strings["etc_avg"].format(mean=mean))
            if trend_desc:
                parts.append(lang_strings["etc_trend"].format(trend=trend_desc))
        elif title_key == "deltas":
            parts.append(lang_strings["sm_avg"].format(mean=mean))
            if trend_desc:
                parts.append(lang_strings["sm_trend"].format(trend=trend_desc))
        elif title_key == "soilmoisture":
            parts.append(lang_strings["soilmoisture_avg"].format(mean=mean))
            if trend_desc:
                parts.append(lang_strings["soilmoisture_trend"].format(trend=trend_desc))        
        else:
            parts.append(lang_strings["default_avg"].format(mean=mean))
            if trend_desc:
                parts.append(lang_strings["default_trend"].format(trend=trend_desc))

    parts.append(lang_strings["analysis_note"])

    result = "\n\n".join(parts)
    if len(result) > max_chars:
        result = result[: max_chars - 200] + "\n\n[Explanation trimmed due to length]"
    return result


def generate_ai_explanation(
    title_text: str,
    metric: dict,
    series: list,
    language: str,
    crop_name: str,
) -> str:
    """
    Generates an explanation using an AI model (e.g., Gemini).
    """
    logger.info(f"Generating AI explanation for '{title_text}' in '{language}'...")

    data_summary_parts = []
    if metric and metric.get("mean") is not None:
        data_summary_parts.append(f"Average value (mean): {metric.get('mean'):.3f}")
    if series:
        data_summary_parts.append(f"Time series data points (date, value):")
        for item in series[:10]:
            date_val = item.get("date") or item.get("day") or item.get("year_month")
            
            val = (item.get("ndvi") or item.get("NDVI") or item.get("ndvi_mean")
                   or item.get("cwr_mm") or item.get("cwr")
                   or item.get("kc")
                   or item.get("lst_c") or item.get("lst")
                   or item.get("etc_mm") or item.get("etc")
                   or item.get("deltas_mm") or item.get("deltas")
                   or item.get("soilmoisture_mm") or item.get("soilmoisture") or item.get("soil_moisture"))
            
            if date_val and val is not None:
                data_summary_parts.append(f"- {date_val}: {val}")
        if len(series) > 10:
             data_summary_parts.append("...and more data points.")

    data_summary = "\n".join(data_summary_parts)
    if not data_summary:
        data_summary = "No data available."

    lang_name = "English"

    prompt = f"""
    You are an expert agronomist analyzing a report for a farmer.
    Your task is to provide a brief, insightful analysis of the provided data in {lang_name}.

    The analysis is for the "{title_text}" metric for a {crop_name} crop.

    Here is the data:
    ---
    {data_summary}
    ---

    Please provide your analysis in simple HTML format (using <b>, <br>, <ul>, <li>).
    Your response should include:
    1. A single bolded title (e.g., "<b>Crop Health Analysis</b>").
    2. A 3-4 sentence summary of the key finding from the data.
    3. A bulleted list (<ul>) with 5-6 key observations (e.g., "The trend is positive," "The average is low," etc.).
    4. A short, actionable recommendation for the farmer.

    Keep the entire response concise and easy to understand for a non-expert.
    Respond ONLY in {lang_name}.
    """

    try:
        api_key = "AIzaSyCJDuNiytOFccf0V3MXCEgMKnByye8HiKY"
        
        if not api_key or api_key == "YOUR_NEW_GEMINI_API_KEY_HERE":
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                logger.info("Using API key from GEMINI_API_KEY environment variable.")
            else:
                raise ValueError("GEMINI_API_KEY not set. Please update the hardcoded key in the code or set GEMINI_API_KEY environment variable.")
        else:
            logger.info("Using hardcoded API key from code.")

        logger.info(f"Configuring Gemini API with key: {api_key[:10]}...")
        genai.configure(api_key=api_key)
        
        available_models = []
        try:
            models_list = list(genai.list_models())
            available_models = [m.name for m in models_list if 'generateContent' in m.supported_generation_methods]
            if available_models:
                logger.info(f"Found {len(available_models)} available models with generateContent support")
                for m in available_models[:3]:
                    logger.info(f"  - {m}")
        except Exception as list_error:
            logger.warning(f"Could not list models (this is okay): {list_error}")
        
        model_names_to_try = []
        
        if available_models:
            flash_models = [m for m in available_models if 'flash' in m.lower()]
            pro_models = [m for m in available_models if 'pro' in m.lower() and m not in flash_models]
            other_models = [m for m in available_models if m not in flash_models and m not in pro_models]
            model_names_to_try = flash_models + pro_models + other_models
        
        if not model_names_to_try:
            model_names_to_try = [
                'gemini-1.5-flash-latest',
                'gemini-1.5-pro-latest', 
                'gemini-pro',
                'models/gemini-pro',
                'gemini-1.5-flash',
                'gemini-1.5-pro'
            ]
        
        response = None
        last_error = None
        
        for model_name in model_names_to_try:
            try:
                logger.info(f"Trying model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, safety_settings=None)
                logger.info(f"Successfully used model: {model_name}")
                break
            except Exception as model_error:
                logger.debug(f"Model {model_name} failed: {model_error}")
                last_error = model_error
                continue
        
        if response is None:
            error_msg = f"No working Gemini model found. Tried: {', '.join(model_names_to_try[:3])}. "
            if available_models:
                error_msg += f"Your API key has access to: {', '.join(available_models[:3])}. "
            error_msg += f"Last error: {last_error}. "
            error_msg += "Please ensure your API key has the Generative Language API enabled in Google Cloud Console."
            raise ValueError(error_msg)
        
        if hasattr(response, 'text'):
            ai_response_raw = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            ai_response_raw = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else ""
        else:
            ai_response_raw = str(response)
        
        if not ai_response_raw:
            raise ValueError("Empty response from Gemini API")

        ai_response = ai_response_raw.replace("```html", "").replace("```", "").strip()

        if not ai_response:
            raise ValueError("No meaningful content in AI response")

        logger.info("Successfully received AI-generated analysis.")
        return ai_response

    except Exception as e:
        logger.error(f"AI generation failed for '{title_text}': {e}", exc_info=True)
        logger.info(f"Falling back to rule-based explanation for '{title_text}'")
        return None


def generate_ai_irrigation_analysis(
    calendar_data: list,
    summary: dict,
    crop_name: str,
) -> str:
    """
    Generates AI-powered irrigation analysis using Gemini.
    """
    logger.info("Generating AI-powered irrigation analysis...")

    data_summary_parts = []
    
    if summary:
        data_summary_parts.append("Irrigation Summary:")
        if summary.get("irrigation"):
            irr = summary["irrigation"]
            data_summary_parts.append(f"- Total Events: {irr.get('total_events', 0)}")
            data_summary_parts.append(f"- Total Water: {irr.get('total_water_mm', 0):.2f} mm")
            data_summary_parts.append(f"- Water Saved: {irr.get('water_saved_mm', 0):.2f} mm ({irr.get('savings_percent', 0):.1f}%)")
        
        if summary.get("urgency"):
            urg = summary["urgency"]
            data_summary_parts.append(f"\nUrgency Breakdown:")
            data_summary_parts.append(f"- URGENT: {urg.get('urgent', 0)} events")
            data_summary_parts.append(f"- HIGH: {urg.get('high', 0)} events")
            data_summary_parts.append(f"- MEDIUM: {urg.get('medium', 0)} events")
    
    if calendar_data:
        data_summary_parts.append("\nSample Irrigation Events:")
        for entry in calendar_data[:5]:
            if entry.get("should_irrigate"):
                data_summary_parts.append(
                    f"- {entry.get('date')}: {entry.get('final_irrigation_mm', 0):.1f}mm "
                    f"({entry.get('priority', 'LOW')}) - {entry.get('advice', 'No advice')}"
                )
    
    data_summary = "\n".join(data_summary_parts)
    if not data_summary:
        data_summary = "No irrigation data available."

    prompt = f"""
    You are an expert irrigation agronomist analyzing a smart irrigation schedule for a farmer.
    Your task is to provide actionable insights about the irrigation plan in English.

    The analysis is for a {crop_name} crop.

    Here is the irrigation data:
    ---
    {data_summary}
    ---

    Please provide your analysis in simple HTML format (using <b>, <br>, <ul>, <li>).
    Your response should include:
    1. A bolded title: "<b>Smart Irrigation Analysis</b>"
    2. A 3-4 sentence overview of the irrigation schedule and water savings
    3. A bulleted list with 5-7 key insights, such as:
       - Critical irrigation dates and their urgency
       - Water savings compared to traditional methods
       - Recommendations for monitoring soil moisture
       - Suggestions for adjusting irrigation based on weather
       - Best practices for efficient water use
    4. A short, actionable recommendation for the farmer

    Keep the response concise and farmer-friendly.
    Respond ONLY in English.
    """

    try:
        api_key = "AIzaSyCJDuNiytOFccf0V3MXCEgMKnByye8HiKY"
        
        if not api_key or api_key == "YOUR_NEW_GEMINI_API_KEY_HERE":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        
        model_names_to_try = [
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro-latest', 
            'gemini-pro',
        ]
        
        response = None
        for model_name in model_names_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, safety_settings=None)
                break
            except Exception:
                continue
        
        if response is None:
            return None
        
        if hasattr(response, 'text'):
            ai_response_raw = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            ai_response_raw = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else ""
        else:
            ai_response_raw = str(response)
        
        if not ai_response_raw:
            return None

        ai_response = ai_response_raw.replace("```html", "").replace("```", "").strip()
        
        if not ai_response:
            return None

        logger.info("Successfully generated AI irrigation analysis.")
        return ai_response

    except Exception as e:
        logger.error(f"AI irrigation analysis failed: {e}", exc_info=True)
        return None


def sanitize_html_for_reportlab(html: str) -> str:
    """
    ReportLab Paragraph supports a limited HTML subset.
    """
    if not html:
        return ""
    text = html
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    import re
    text = re.sub(r"<\s*br\s*>", "<br/>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*br\s*/\s*>", "<br/>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*ul[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*ul\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*li[^>]*>(.*?)</\s*li\s*>", r"• \1<br/>", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</\s*p\s*>", "<br/>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*p[^>]*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def get_location_name(lat: float, lon: float) -> str | None:
    try:
        geolocator = Nominatim(user_agent="krishizest_report_app_v1")
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=5, language="en")
        if location and location.raw.get("address"):
            address = location.raw["address"]
            city = address.get("city") or address.get("town") or address.get("village")
            state = address.get("state")
            country = address.get("country")
            parts = [p for p in (city, state) if p]
            if not parts and country:
                parts.append(country)
            return ", ".join(parts) if parts else None
    except (GeocoderTimedOut, GeocoderUnavailable):
        logger.warning("Reverse geocoding service timed out or is unavailable.")
    except Exception as exc:
        logger.warning("Reverse geocoding error: %s", exc)
    return None


def make_reportlab_image_high_quality(buf: io.BytesIO, max_width_pt: float, max_height_pt: float) -> Image:
    """
    Create a HIGH-QUALITY ReportLab Image with proper DPI settings.
    """
    try:
        buf.seek(0)
        pil_img = PILImage.open(buf)
        
        # Get original dimensions
        px_w, px_h = pil_img.size
        
        # Set high DPI for print quality
        target_dpi = 300
        
        # Calculate dimensions in points (1 inch = 72 points)
        img_w_pt = (px_w / target_dpi) * 72
        img_h_pt = (px_h / target_dpi) * 72
        
        # Scale to fit within max dimensions
        scale = min(1.0, max_width_pt / img_w_pt, max_height_pt / img_h_pt)
        draw_w = img_w_pt * scale
        draw_h = img_h_pt * scale
        
        # Save with high quality
        out_buf = io.BytesIO()
        pil_img.save(out_buf, format='PNG', optimize=False, compress_level=1, dpi=(target_dpi, target_dpi))
        out_buf.seek(0)
        
        # Create ReportLab image
        rl_img = Image(out_buf, width=draw_w, height=draw_h)
        
        return rl_img
        
    except Exception as e:
        logger.error(f"Error creating high-quality image: {e}")
        # Fallback
        buf.seek(0)
        return Image(buf, width=max_width_pt, height=max_height_pt)


def generate_report_response(payload: dict):
    """Build and return the PDF report as a Flask response."""

    if not payload:
        raise ValueError("No JSON payload received")

    language = payload.get("language", "en").lower()
    if language not in ["en", "hi"]:
        language = "en"
    
    lang_strings = get_translations(language)

    crop_name = payload.get("crop_name", "Crop")
    aoi = payload.get("aoi")
    aoi_coords = None
    center_lat = None
    center_lon = None
    try:
        aoi_coords = aoi["geometry"]["coordinates"]
    except Exception:
        aoi_coords = None

    location_text = payload.get("location")
    if not location_text and aoi_coords:
        try:
            polygon = aoi_coords[0] if isinstance(aoi_coords[0][0], (list, tuple)) else aoi_coords
            avg_lon = sum(point[0] for point in polygon) / len(polygon)
            avg_lat = sum(point[1] for point in polygon) / len(polygon)
            center_lon = avg_lon
            center_lat = avg_lat
            location_text = get_location_name(avg_lat, avg_lon) or f"{round(avg_lat, 5)}, {round(avg_lon, 5)}"
        except Exception:
            location_text = None

    cloud_cover = payload.get("cloud_cover", "Not provided")
    area_ha = payload.get("area_ha")
    start_date = payload.get("start_date", "")
    end_date = payload.get("end_date", "")
    metrics = payload.get("metrics") or {}
    series = payload.get("series") or []
    landcover = payload.get("irrigation_calendar") or payload.get("land_cover") or {}
    
    irrigation_calendar = payload.get("irrigation_calendar") or []
    irrigation_summary = payload.get("irrigation_summary") or {}
    backend_url = payload.get("backend_url") or "http://localhost:5000"

    # Ensure landcover is always a dict
    if isinstance(landcover, list):
        landcover = {}

    if not landcover:
        landcover = {lang_strings["landcover_veg"]: 70, lang_strings["landcover_bare"]: 30}
    else:
        translated_landcover = {}
        for k, v in landcover.items():
            if "veg" in k.lower():
                translated_landcover[lang_strings["landcover_veg"]] = v
            elif "bare" in k.lower():
                translated_landcover[lang_strings["landcover_bare"]] = v
            else:
                translated_landcover[k] = v
        landcover = translated_landcover

    logo_path = get_logo_path(payload.get("logo_path"))
    if not logo_path:
        logger.warning("Logo not found in any expected locations")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=30 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    normal.spaceAfter = 6

    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontSize=18, alignment=1, spaceAfter=8)

    story = [Spacer(1, 8), Paragraph(lang_strings["report_title"], heading), Spacer(1, 12)]

    if logo_path and os.path.exists(logo_path):
        try:
            pil_img = PILImage.open(logo_path)
            pil_img.thumbnail((380, 150), PILImage.LANCZOS)
            logo_buffer = io.BytesIO()
            pil_img.save(logo_buffer, format="PNG")
            logo_buffer.seek(0)
            story.append(Image(logo_buffer, width=pil_img.width * 0.75, height=pil_img.height * 0.75, hAlign="CENTER"))
            story.append(Spacer(1, 20))
        except Exception as exc:
            logger.warning("Logo loading error: %s", exc)
            story.append(Spacer(1, 24))
            story.append(Paragraph(lang_strings["logo_alt"], ParagraphStyle("LogoPlaceholder", fontSize=14, alignment=1, textColor=colors.HexColor("#0b486b"))))
    else:
        story.append(Spacer(1, 24))
        story.append(Paragraph(lang_strings["logo_alt"], ParagraphStyle("LogoPlaceholder", fontSize=14, alignment=1, textColor=colors.HexColor("#0b486b"))))

    story.append(Spacer(1, 40))
    for info in (
        f"<b>{lang_strings['crop']}:</b> {crop_name}",
        f"<b>{lang_strings['period']}:</b> {start_date} to {end_date}",
        f"<b>{lang_strings['location']}:</b> {location_text or lang_strings['not_specified']}",
    ):
        story.append(Paragraph(info, ParagraphStyle("CoverInfo", fontSize=12, alignment=1, spaceAfter=8)))

    story.append(PageBreak())

    # --- AOI Map + Metadata Section ---
    story.append(Paragraph(lang_strings["aoi_title"], ParagraphStyle("AOIHeader", fontSize=16, spaceAfter=10)))
    aoi_buffer = io.BytesIO()
    if aoi_coords:
        _render_aoi_map(aoi_coords, aoi_buffer, title=lang_strings["aoi_map_title"], na_text=lang_strings["aoi_na"])
    else:
        _render_aoi_map(None, aoi_buffer, na_text=lang_strings["aoi_na"])
        
    story.append(Image(aoi_buffer, width=170 * mm, height=95 * mm))
    story.append(Spacer(1, 8))

    # Build metadata rows; include center coordinates if available
    meta_rows = [
        [lang_strings["aoi_area"], str(area_ha if area_ha is not None else "—")],
        [lang_strings["location"], str(location_text or "—")],
    ]
    if center_lat is not None and center_lon is not None:
        meta_rows.append(["Center Coordinates", f"Lat: {center_lat:.5f}, Lon: {center_lon:.5f}"])
    meta_rows.extend([
        [lang_strings["time_period"], f"{start_date} to {end_date}"],
        [lang_strings["cloud_cover"], str(cloud_cover)],
        [lang_strings["report_generated"], datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
    ])

    meta_table = Table(meta_rows, colWidths=[55 * mm, 105 * mm])
    meta_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f6f8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(meta_table)
    story.append(PageBreak())

    # --- NDVI / Crop Health Section ---
    story.append(Paragraph(lang_strings["ndvi_title"], ParagraphStyle("SecHeading", fontSize=16, spaceAfter=8)))
    
    # NDVI Graph
    ndvi_buffer = io.BytesIO()
    _render_ndvi_chart(
        series,
        ndvi_buffer,
        title=f"{crop_name} {lang_strings['ndvi_chart_title_suffix']}",
        na_text=lang_strings["ndvi_na"],
    )
    story.append(Image(ndvi_buffer, width=170 * mm, height=55 * mm))
    story.append(Spacer(1, 8))

    # NDVI Large Tile Map
    try:
        logger.info("Fetching enhanced NDVI tile mosaic...")
        ndvi_tile_buf, ndvi_stats = create_high_quality_tile_mosaic(
            aoi, start_date, end_date, 'ndvi', 
            backend_url=backend_url,
            zoom=15,
            grid_size=3,
            final_size=2400
        )
        
        if ndvi_tile_buf:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Crop Health Map Visualization", 
                                  ParagraphStyle("MapTitle", fontSize=12, alignment=1, spaceAfter=6)))
            
            try:
                # Add professional colormap legend
                ndvi_pil = PILImage.open(ndvi_tile_buf)
                styled_ndvi = add_colormap_legend(ndvi_pil, 'ndvi', 'Crop Health Map (NDVI)')

                # Save styled image
                styled_buf = BytesIO()
                styled_ndvi.save(styled_buf, format='PNG', optimize=True)
                styled_buf.seek(0)

                # Use larger size for better visibility and put inside a boxed table
                rl_img = make_reportlab_image_high_quality(styled_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_img]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
                story.append(Spacer(1, 8))
                
                # Add map description
                if ndvi_stats:
                    ndvi_mean = ndvi_stats.get('mean', 0)
                    health_level = "Excellent" if ndvi_mean > 0.7 else "Good" if ndvi_mean > 0.5 else "Moderate" if ndvi_mean > 0.3 else "Poor"
                    map_desc = f"<i>Map shows NDVI values across your field. Current average: <b>{ndvi_mean:.3f}</b> ({health_level} health). Green areas indicate healthy vegetation.</i>"
                    story.append(Paragraph(map_desc, ParagraphStyle("MapDesc", fontSize=9, textColor=colors.grey)))
                
            except Exception as e:
                logger.warning(f"Failed to style NDVI tile image: {e}")
                # Fallback: use original tile wrapped in box
                rl_img = make_reportlab_image_high_quality(ndvi_tile_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_img]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
    except Exception as e:
        logger.debug(f"NDVI tile mosaic not available: {e}")

    # NDVI Analysis
    story.append(Spacer(1, 12))
    ndvi_expl = generate_ai_explanation(
        lang_strings["ndvi_title"], metrics.get("ndvi") or {}, series, language, crop_name
    )
    
    if ndvi_expl is None:
        ndvi_expl = generate_plain_language_explanation(
            "ndvi", lang_strings["ndvi_title"], metrics.get("ndvi") or {}, series, lang_strings, max_chars=3500
        )
    
    for block in ndvi_expl.split("\n\n"):
        story.append(Paragraph(sanitize_html_for_reportlab(block), normal))
    
    story.append(PageBreak())

    # --- CWR Section ---
    story.append(Paragraph(lang_strings["cwr_title"], ParagraphStyle("ParamHeading", fontSize=14, spaceAfter=6)))
    
    # CWR Graph
    cwr_series_vals = []
    for entry in series:
        if entry.get("cwr_mm") is not None:
            cwr_series_vals.append({"date": entry.get("date"), "val": entry.get("cwr_mm")})
        elif entry.get("cwr") is not None:
            cwr_series_vals.append({"date": entry.get("date"), "val": entry.get("cwr")})

    if cwr_series_vals:
        dates, vals = [], []
        for item in cwr_series_vals:
            try:
                dates.append(pd.to_datetime(item["date"]))
                vals.append(float(item["val"]))
            except Exception:
                continue
        if vals:
            fig, ax = plt.subplots(figsize=(7.4, 2.4))
            ax.plot(dates, vals, marker="o", color='blue')
            ax.set_title("Crop Water Requirement Over Time", fontsize=9)
            ax.set_ylabel("CWR (mm)")
            ax.grid(alpha=0.25)
            fig.autofmt_xdate(rotation=30)
            cwr_buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(cwr_buf, format="png", bbox_inches="tight", dpi=140)
            plt.close(fig)
            cwr_buf.seek(0)
            story.append(Image(cwr_buf, width=170 * mm, height=45 * mm))
            story.append(Spacer(1, 8))

    # CWR Large Tile Map
    try:
        logger.info("Fetching CWR tile mosaic...")
        cwr_tile_buf, cwr_stats = create_high_quality_tile_mosaic(
            aoi, start_date, end_date, 'cwr', 
            backend_url=backend_url,
            zoom=15,
            grid_size=3,
            final_size=2400
        )
        
        if cwr_tile_buf:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Crop Water Requirement Map", 
                                  ParagraphStyle("MapTitle", fontSize=12, alignment=1, spaceAfter=6)))
            
            try:
                cwr_pil = PILImage.open(cwr_tile_buf)
                styled_cwr = add_colormap_legend(cwr_pil, 'cwr', 'Crop Water Requirement Map')
                
                styled_cwr_buf = BytesIO()
                styled_cwr.save(styled_cwr_buf, format='PNG', optimize=True)
                styled_cwr_buf.seek(0)

                rl_cwr = make_reportlab_image_high_quality(styled_cwr_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_cwr]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
                story.append(Spacer(1, 8))

                if cwr_stats:
                    cwr_mean = cwr_stats.get('mean', 0)
                    cwr_level = "High" if cwr_mean > 7 else "Moderate" if cwr_mean > 4 else "Low"
                    cwr_desc = f"<i>Map shows daily water requirement across your field. Current average: <b>{cwr_mean:.2f} mm/day</b> ({cwr_level} requirement). Blue areas indicate higher water needs.</i>"
                    story.append(Paragraph(cwr_desc, ParagraphStyle("MapDesc", fontSize=9, textColor=colors.grey)))

            except Exception as e:
                logger.warning(f"Failed to style CWR tile image: {e}")
                rl_cwr = make_reportlab_image_high_quality(cwr_tile_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_cwr]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
    except Exception as e:
        logger.debug(f"CWR tile mosaic not available: {e}")

    # CWR Analysis
    story.append(Spacer(1, 12))
    cwr_expl = generate_ai_explanation(
        lang_strings["cwr_title"], metrics.get("cwr_mm") or metrics.get("cwr") or {}, series, language, crop_name
    )
    
    if cwr_expl is None:
        cwr_expl = generate_plain_language_explanation(
            "cwr", lang_strings["cwr_title"], metrics.get("cwr_mm") or metrics.get("cwr") or {}, series, lang_strings, max_chars=2500
        )
    
    for paragraph in cwr_expl.split("\n\n"):
        story.append(Paragraph(sanitize_html_for_reportlab(paragraph), normal))
    story.append(Spacer(1, 4))
    story.append(PageBreak())

    # --- LST Section ---
    story.append(Paragraph(lang_strings["lst_title"], ParagraphStyle("ParamHeading", fontSize=14, spaceAfter=6)))
    
    # LST Graph
    lst_series_vals = []
    for entry in series:
        if entry.get("lst_c") is not None:
            lst_series_vals.append({"date": entry.get("date"), "val": entry.get("lst_c")})
        elif entry.get("lst") is not None:
            lst_series_vals.append({"date": entry.get("date"), "val": entry.get("lst")})

    if lst_series_vals:
        dates, vals = [], []
        for item in lst_series_vals:
            try:
                dates.append(pd.to_datetime(item["date"]))
                vals.append(float(item["val"]))
            except Exception:
                continue
        if vals:
            fig, ax = plt.subplots(figsize=(7.4, 2.4))
            ax.plot(dates, vals, marker="o", color='red')
            ax.set_title("Land Surface Temperature Over Time", fontsize=9)
            ax.set_ylabel("Temperature (°C)")
            ax.grid(alpha=0.25)
            fig.autofmt_xdate(rotation=30)
            lst_buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(lst_buf, format="png", bbox_inches="tight", dpi=140)
            plt.close(fig)
            lst_buf.seek(0)
            story.append(Image(lst_buf, width=170 * mm, height=45 * mm))
            story.append(Spacer(1, 8))

    # LST Large Tile Map
    try:
        logger.info("Fetching LST tile mosaic...")
        lst_tile_buf, lst_stats = create_high_quality_tile_mosaic(
            aoi, start_date, end_date, 'lst', 
            backend_url=backend_url,
            zoom=15,
            grid_size=3,
            final_size=2400
        )
        
        if lst_tile_buf:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Land Surface Temperature Map", 
                                  ParagraphStyle("MapTitle", fontSize=12, alignment=1, spaceAfter=6)))
            
            try:
                lst_pil = PILImage.open(lst_tile_buf)
                styled_lst = add_colormap_legend(lst_pil, 'lst', 'Land Surface Temperature Map')
                
                styled_lst_buf = BytesIO()
                styled_lst.save(styled_lst_buf, format='PNG', optimize=True)
                styled_lst_buf.seek(0)

                rl_lst = make_reportlab_image_high_quality(styled_lst_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_lst]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
                story.append(Spacer(1, 8))

                if lst_stats:
                    lst_mean = lst_stats.get('mean', 0)
                    temp_level = "Hot" if lst_mean > 35 else "Warm" if lst_mean > 25 else "Cool"
                    lst_desc = f"<i>Map shows surface temperature across your field. Current average: <b>{lst_mean:.1f}°C</b> ({temp_level}). Red areas indicate higher temperatures.</i>"
                    story.append(Paragraph(lst_desc, ParagraphStyle("MapDesc", fontSize=9, textColor=colors.grey)))

            except Exception as e:
                logger.warning(f"Failed to style LST tile image: {e}")
                rl_lst = make_reportlab_image_high_quality(lst_tile_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_lst]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
    except Exception as e:
        logger.debug(f"LST tile mosaic not available: {e}")

    # LST Analysis
    story.append(Spacer(1, 12))
    lst_expl = generate_ai_explanation(
        lang_strings["lst_title"], metrics.get("lst_c") or metrics.get("lst") or {}, series, language, crop_name
    )
    
    if lst_expl is None:
        lst_expl = generate_plain_language_explanation(
            "lst", lang_strings["lst_title"], metrics.get("lst_c") or metrics.get("lst") or {}, series, lang_strings, max_chars=2500
        )
    
    for paragraph in lst_expl.split("\n\n"):
        story.append(Paragraph(sanitize_html_for_reportlab(paragraph), normal))
    story.append(Spacer(1, 4))
    story.append(PageBreak())

    # --- ETC Section ---
    story.append(Paragraph(lang_strings["etc_title"], ParagraphStyle("ParamHeading", fontSize=14, spaceAfter=6)))
    
    # ETC Graph
    etc_series_vals = []
    for entry in series:
        if entry.get("etc_mm") is not None:
            etc_series_vals.append({"date": entry.get("date"), "val": entry.get("etc_mm")})
        elif entry.get("etc") is not None:
            etc_series_vals.append({"date": entry.get("date"), "val": entry.get("etc")})

    if etc_series_vals:
        dates, vals = [], []
        for item in etc_series_vals:
            try:
                dates.append(pd.to_datetime(item["date"]))
                vals.append(float(item["val"]))
            except Exception:
                continue
        if vals:
            fig, ax = plt.subplots(figsize=(7.4, 2.4))
            ax.plot(dates, vals, marker="o", color='green')
            ax.set_title("Evapotranspiration Over Time", fontsize=9)
            ax.set_ylabel("ETc (mm)")
            ax.grid(alpha=0.25)
            fig.autofmt_xdate(rotation=30)
            etc_buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(etc_buf, format="png", bbox_inches="tight", dpi=140)
            plt.close(fig)
            etc_buf.seek(0)
            story.append(Image(etc_buf, width=170 * mm, height=45 * mm))
            story.append(Spacer(1, 8))

    # ETC Large Tile Map
    try:
        logger.info("Fetching ETC tile mosaic...")
        etc_tile_buf, etc_stats = create_high_quality_tile_mosaic(
            aoi, start_date, end_date, 'etc', 
            backend_url=backend_url,
            zoom=15,
            grid_size=3,
            final_size=2400
        )
        
        if etc_tile_buf:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Evapotranspiration Map", 
                                  ParagraphStyle("MapTitle", fontSize=12, alignment=1, spaceAfter=6)))
            
            try:
                etc_pil = PILImage.open(etc_tile_buf)
                styled_etc = add_colormap_legend(etc_pil, 'etc', 'Evapotranspiration Map')
                
                styled_etc_buf = BytesIO()
                styled_etc.save(styled_etc_buf, format='PNG', optimize=True)
                styled_etc_buf.seek(0)

                rl_etc = make_reportlab_image_high_quality(styled_etc_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_etc]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
                story.append(Spacer(1, 8))

                if etc_stats:
                    etc_mean = etc_stats.get('mean', 0)
                    etc_level = "High" if etc_mean > 7 else "Moderate" if etc_mean > 4 else "Low"
                    etc_desc = f"<i>Map shows evapotranspiration rates across your field. Current average: <b>{etc_mean:.2f} mm/day</b> ({etc_level}). Dark blue areas indicate higher water loss.</i>"
                    story.append(Paragraph(etc_desc, ParagraphStyle("MapDesc", fontSize=9, textColor=colors.grey)))

            except Exception as e:
                logger.warning(f"Failed to style ETC tile image: {e}")
                rl_etc = make_reportlab_image_high_quality(etc_tile_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_etc]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
    except Exception as e:
        logger.debug(f"ETC tile mosaic not available: {e}")

    # ETC Analysis
    story.append(Spacer(1, 12))
    etc_expl = generate_ai_explanation(
        lang_strings["etc_title"], metrics.get("etc_mm") or metrics.get("etc") or {}, series, language, crop_name
    )
    
    if etc_expl is None:
        etc_expl = generate_plain_language_explanation(
            "etc", lang_strings["etc_title"], metrics.get("etc_mm") or metrics.get("etc") or {}, series, lang_strings, max_chars=2500
        )
    
    for paragraph in etc_expl.split("\n\n"):
        story.append(Paragraph(sanitize_html_for_reportlab(paragraph), normal))
    story.append(Spacer(1, 4))
    story.append(PageBreak())

    # --- Kc Section (No Tile Available) ---
    story.append(Paragraph(lang_strings["kc_title"], ParagraphStyle("ParamHeading", fontSize=14, spaceAfter=6)))
    
    # Kc Graph
    kc_series_vals = []
    for entry in series:
        if entry.get("kc") is not None:
            kc_series_vals.append({"date": entry.get("date"), "val": entry.get("kc")})

    if kc_series_vals:
        dates, vals = [], []
        for item in kc_series_vals:
            try:
                dates.append(pd.to_datetime(item["date"]))
                vals.append(float(item["val"]))
            except Exception:
                continue
        if vals:
            fig, ax = plt.subplots(figsize=(7.4, 2.4))
            ax.plot(dates, vals, marker="o", color='purple')
            ax.set_title("Crop Coefficient (Kc) Over Time", fontsize=9)
            ax.set_ylabel("Kc")
            ax.grid(alpha=0.25)
            fig.autofmt_xdate(rotation=30)
            kc_buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(kc_buf, format="png", bbox_inches="tight", dpi=140)
            plt.close(fig)
            kc_buf.seek(0)
            story.append(Image(kc_buf, width=170 * mm, height=45 * mm))
            story.append(Spacer(1, 8))

    # Kc Analysis (No tile available)
    story.append(Spacer(1, 12))
    kc_expl = generate_ai_explanation(
        lang_strings["kc_title"], metrics.get("kc") or {}, series, language, crop_name
    )
    
    if kc_expl is None:
        kc_expl = generate_plain_language_explanation(
            "kc", lang_strings["kc_title"], metrics.get("kc") or {}, series, lang_strings, max_chars=2500
        )
    
    for paragraph in kc_expl.split("\n\n"):
        story.append(Paragraph(sanitize_html_for_reportlab(paragraph), normal))
    story.append(Spacer(1, 4))
    story.append(PageBreak())

    # --- Soil Moisture Section ---
    story.append(Paragraph(lang_strings["soilmoisture_title"], ParagraphStyle("ParamHeading", fontSize=14, spaceAfter=6)))
    
    # Soil Moisture Graph
    sm_series_vals = []
    for entry in series:
        if entry.get("soilmoisture_mm") is not None:
            sm_series_vals.append({"date": entry.get("date"), "val": entry.get("soilmoisture_mm")})
        elif entry.get("soilmoisture") is not None:
            sm_series_vals.append({"date": entry.get("date"), "val": entry.get("soilmoisture")})
        elif entry.get("soil_moisture") is not None:
            sm_series_vals.append({"date": entry.get("date"), "val": entry.get("soil_moisture")})

    if sm_series_vals:
        dates, vals = [], []
        for item in sm_series_vals:
            try:
                dates.append(pd.to_datetime(item["date"]))
                vals.append(float(item["val"]))
            except Exception:
                continue
        if vals:
            fig, ax = plt.subplots(figsize=(7.4, 2.4))
            ax.plot(dates, vals, marker="o", color='brown')
            ax.set_title("Soil Moisture Over Time", fontsize=9)
            ax.set_ylabel("Soil Moisture")
            ax.grid(alpha=0.25)
            fig.autofmt_xdate(rotation=30)
            sm_buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(sm_buf, format="png", bbox_inches="tight", dpi=140)
            plt.close(fig)
            sm_buf.seek(0)
            story.append(Image(sm_buf, width=170 * mm, height=45 * mm))
            story.append(Spacer(1, 8))

    # Soil Moisture Large Tile Map
    try:
        logger.info("Fetching soil moisture tile mosaic...")
        sm_tile_buf, sm_stats = create_high_quality_tile_mosaic(
            aoi, start_date, end_date, 'soilmoisture', 
            backend_url=backend_url,
            zoom=15,
            grid_size=3,
            final_size=2400
        )
        
        if sm_tile_buf:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Soil Moisture Map", 
                                  ParagraphStyle("MapTitle", fontSize=12, alignment=1, spaceAfter=6)))
            
            try:
                sm_pil = PILImage.open(sm_tile_buf)
                styled_sm = add_colormap_legend(sm_pil, 'soilmoisture', 'Soil Moisture Map')
                
                styled_sm_buf = BytesIO()
                styled_sm.save(styled_sm_buf, format='PNG', optimize=True)
                styled_sm_buf.seek(0)

                rl_sm = make_reportlab_image_high_quality(styled_sm_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_sm]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
                story.append(Spacer(1, 8))

                if sm_stats:
                    sm_mean = sm_stats.get('mean', 0)
                    moisture_level = "High" if sm_mean > 60 else "Adequate" if sm_mean > 40 else "Low" if sm_mean > 20 else "Very Low"
                    sm_desc = f"<i>Map shows soil moisture distribution. Current average: <b>{sm_mean:.1f}%</b> ({moisture_level} moisture). Darker green areas indicate higher moisture content.</i>"
                    story.append(Paragraph(sm_desc, ParagraphStyle("MapDesc", fontSize=9, textColor=colors.grey)))

            except Exception as e:
                logger.warning(f"Failed to style soil moisture tile image: {e}")
                rl_sm = make_reportlab_image_high_quality(sm_tile_buf, 170 * mm, 120 * mm)
                tile_table = Table([[rl_sm]], colWidths=[170 * mm], rowHeights=[120 * mm])
                tile_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(tile_table)
    except Exception as e:
        logger.debug(f"Soil moisture tile mosaic not available: {e}")

    # Soil Moisture Analysis
    story.append(Spacer(1, 12))
    sm_expl = generate_ai_explanation(
        lang_strings["soilmoisture_title"], metrics.get("soilmoisture_mm") or metrics.get("soilmoisture") or {}, series, language, crop_name
    )
    
    if sm_expl is None:
        sm_expl = generate_plain_language_explanation(
            "soilmoisture", lang_strings["soilmoisture_title"], metrics.get("soilmoisture_mm") or metrics.get("soilmoisture") or {}, series, lang_strings, max_chars=2500
        )
    
    for paragraph in sm_expl.split("\n\n"):
        story.append(Paragraph(sanitize_html_for_reportlab(paragraph), normal))
    story.append(Spacer(1, 4))
    story.append(PageBreak())

    # ==================== IRRIGATION SECTION ====================
    if irrigation_calendar and irrigation_summary:
        story.append(Paragraph(lang_strings["irrigation_title"], 
                              ParagraphStyle("IrrigationHeading", fontSize=16, spaceAfter=10, 
                                           textColor=colors.HexColor("#0b486b"))))
        
        story.append(Paragraph(lang_strings["irrigation_summary_title"], 
                              ParagraphStyle("IrrigationSubHeading", fontSize=14, spaceAfter=8)))
        
        irr_data = irrigation_summary.get("irrigation", {})
        urg_data = irrigation_summary.get("urgency", {})
        
        summary_table_data = [
            [lang_strings["irrigation_total_events"], str(irr_data.get("total_events", 0))],
            [lang_strings["irrigation_total_water"], f"{irr_data.get('total_water_mm', 0):.2f} mm"],
            [lang_strings["irrigation_water_saved"], f"{irr_data.get('water_saved_mm', 0):.2f} mm"],
            [lang_strings["irrigation_savings_percent"], f"{irr_data.get('savings_percent', 0):.1f}%"],
            [lang_strings["irrigation_urgent"], str(urg_data.get("urgent", 0))],
            [lang_strings["irrigation_high"], str(urg_data.get("high", 0))],
            [lang_strings["irrigation_medium"], str(urg_data.get("medium", 0))],
        ]
        
        summary_table = Table(summary_table_data, colWidths=[80 * mm, 80 * mm])
        summary_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e3f2fd")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ])
        )
        story.append(summary_table)
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(lang_strings["irrigation_timeline_title"], 
                              ParagraphStyle("ChartTitle", fontSize=12, spaceAfter=6)))
        timeline_buffer = io.BytesIO()
        _render_irrigation_timeline(irrigation_calendar, timeline_buffer, 
                                   title=lang_strings["irrigation_timeline_title"])
        story.append(Image(timeline_buffer, width=170 * mm, height=90 * mm))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(lang_strings["irrigation_priority_title"], 
                              ParagraphStyle("ChartTitle", fontSize=12, spaceAfter=6)))
        priority_buffer = io.BytesIO()
        _render_priority_distribution(irrigation_calendar, priority_buffer,
                                     title=lang_strings["irrigation_priority_title"])
        story.append(Image(priority_buffer, width=120 * mm, height=90 * mm, hAlign="CENTER"))
        story.append(Spacer(1, 12))
        story.append(PageBreak())
        
        story.append(Paragraph(lang_strings["irrigation_comparison_title"], 
                              ParagraphStyle("ChartTitle", fontSize=12, spaceAfter=6)))
        comparison_buffer = io.BytesIO()
        _render_water_comparison(irrigation_calendar, comparison_buffer,
                                title=lang_strings["irrigation_comparison_title"])
        story.append(Image(comparison_buffer, width=170 * mm, height=80 * mm))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(lang_strings["irrigation_calendar_title"], 
                              ParagraphStyle("CalendarTitle", fontSize=12, spaceAfter=6)))
        
        irrigation_events = [e for e in irrigation_calendar if e.get("should_irrigate")][:10]
        
        if irrigation_events:
            calendar_table_data = [[
                lang_strings["irrigation_date"],
                lang_strings["irrigation_amount"],
                lang_strings["irrigation_priority"],
                lang_strings["irrigation_advice"]
            ]]
            
            for event in irrigation_events:
                date_str = event.get("date", "—")
                amount = f"{event.get('final_irrigation_mm', 0):.1f}"
                priority = event.get("priority", "LOW")
                advice = event.get("advice", "No advice")
                
                if len(advice) > 60:
                    advice = advice[:57] + "..."
                
                calendar_table_data.append([date_str, amount, priority, advice])
            
            calendar_table = Table(calendar_table_data, 
                                  colWidths=[25*mm, 25*mm, 25*mm, 85*mm])
            
            table_style = [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b486b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
            
            for i, event in enumerate(irrigation_events, start=1):
                priority = event.get("priority", "LOW")
                if priority == "URGENT":
                    table_style.append(("BACKGROUND", (2, i), (2, i), colors.HexColor("#ffcdd2")))
                elif priority == "HIGH":
                    table_style.append(("BACKGROUND", (2, i), (2, i), colors.HexColor("#ffe0b2")))
                elif priority == "MEDIUM":
                    table_style.append(("BACKGROUND", (2, i), (2, i), colors.HexColor("#fff9c4")))
                else:
                    table_style.append(("BACKGROUND", (2, i), (2, i), colors.HexColor("#c8e6c9")))
            
            calendar_table.setStyle(TableStyle(table_style))
            story.append(calendar_table)
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(lang_strings["irrigation_no_data"], normal))
        
        story.append(PageBreak())
        
        ai_irrigation_analysis = generate_ai_irrigation_analysis(
            irrigation_calendar, 
            irrigation_summary, 
            crop_name
        )
        
        if ai_irrigation_analysis:
            story.append(Paragraph("AI-Powered Irrigation Recommendations", 
                                  ParagraphStyle("AIHeading", fontSize=14, spaceAfter=8, 
                                               textColor=colors.HexColor("#0b486b"))))
            
            for block in ai_irrigation_analysis.split("\n\n"):
                story.append(Paragraph(sanitize_html_for_reportlab(block), normal))
            story.append(Spacer(1, 4))
            story.append(PageBreak())
        else:
            story.append(Paragraph("Irrigation Recommendations", 
                                  ParagraphStyle("AIHeading", fontSize=14, spaceAfter=8, 
                                               textColor=colors.HexColor("#0b486b"))))
            
            recommendations = []
            recommendations.append("<b>Smart Irrigation Insights:</b>")
            
            total_events = irr_data.get("total_events", 0)
            total_water = irr_data.get("total_water_mm", 0)
            water_saved = irr_data.get("water_saved_mm", 0)
            savings_percent = irr_data.get("savings_percent", 0)
            
            if total_events > 0:
                recommendations.append(
                    f"• Your field requires irrigation on <b>{total_events} dates</b> during this period, "
                    f"with a total water requirement of <b>{total_water:.1f}mm</b>."
                )
            
            if water_saved > 0:
                recommendations.append(
                    f"• By following this smart schedule, you can save approximately <b>{water_saved:.1f}mm</b> "
                    f"of water (<b>{savings_percent:.1f}%</b> savings) compared to traditional irrigation methods."
                )
            
            urgent_count = urg_data.get("urgent", 0)
            high_count = urg_data.get("high", 0)
            
            if urgent_count > 0:
                recommendations.append(
                    f"• <b>Critical Alert:</b> {urgent_count} date(s) require <b>URGENT irrigation</b>. "
                    "Immediate action needed to prevent crop stress."
                )
            
            if high_count > 0:
                recommendations.append(
                    f"• <b>High Priority:</b> {high_count} date(s) need irrigation soon. "
                    "Plan ahead to ensure water availability."
                )
            
            recommendations.append(
                "• <b>Best Practices:</b><br/>"
                "&nbsp;&nbsp;- Monitor soil moisture regularly using probes or manual inspection<br/>"
                "&nbsp;&nbsp;- Irrigate early morning or late evening to reduce evaporation losses<br/>"
                "&nbsp;&nbsp;- Check weather forecasts before irrigating to avoid wasting water<br/>"
                "&nbsp;&nbsp;- Ensure uniform water distribution across the field<br/>"
                "&nbsp;&nbsp;- Adjust irrigation based on actual field conditions"
            )
            
            recommendations.append(
                "<b>Note:</b> This schedule is based on satellite data and weather patterns. "
                "Always verify with on-ground observations and adjust as needed."
            )
            
            for rec in recommendations:
                story.append(Paragraph(sanitize_html_for_reportlab(rec), normal))
                story.append(Spacer(1, 4))
            
            story.append(PageBreak())
    
    # ==================== END IRRIGATION SECTION ====================

    # --- Analytics Summary Section ---
    story.append(Paragraph(lang_strings["analytics_title"], ParagraphStyle("AnalyticsHeading", fontSize=16, spaceAfter=8)))
    analytics = []
    nd_mean = (metrics.get("ndvi") or {}).get("mean")
    cwr_mean = (metrics.get("cwr_mm") or metrics.get("cwr") or {}).get("mean")
    sm_mean = (metrics.get("deltas_mm") or metrics.get("deltas") or {}).get("mean")
    lst_mean = (metrics.get("lst_c") or metrics.get("lst") or {}).get("mean")

    if nd_mean is not None:
        analytics.append(lang_strings["analytics_health"].format(mean=nd_mean))
    else:
        analytics.append(f"- {lang_strings['analytics_health_na']}")
    if cwr_mean is not None:
        analytics.append(lang_strings["analytics_cwr"].format(mean=cwr_mean))
    else:
        analytics.append(f"- {lang_strings['analytics_cwr_na']}")
    if lst_mean is not None:
        analytics.append(lang_strings["analytics_temp"].format(mean=lst_mean))
    if sm_mean is not None:
        analytics.append(lang_strings["analytics_sm"].format(mean=sm_mean))
    analytics.append(f"- {lang_strings['analytics_rec']}")

    for line in analytics:
        story.append(Paragraph(line, normal))
    story.append(PageBreak())

    # Pie chart removed per user request (Field Cover: Vegetation vs Bare Land)

    # --- Final Page ---
    story.append(Spacer(1, 80))
    story.append(Paragraph(lang_strings["thanks"], ParagraphStyle("Thanks", fontSize=30, leading=36, alignment=1)))
    story.append(Spacer(1, 40))
    story.append(Paragraph(lang_strings["contact"], ParagraphStyle("Contact", fontSize=10, alignment=1)))
    story.append(PageBreak())

    header_fn = make_draw_header(location_text or "Unknown", lang_strings["aoi_label"])
    doc.build(story, onFirstPage=lambda c, d: (header_fn(c, d), draw_footer(c, d)), onLaterPages=lambda c, d: (header_fn(c, d), draw_footer(c, d)))

    buffer.seek(0)
    filename = f"{crop_name}_CropHealthReport_{start_date}_to_{end_date}.pdf"
    return send_file(buffer, download_name=filename, as_attachment=True, mimetype="application/pdf")