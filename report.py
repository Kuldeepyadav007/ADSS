from datetime import datetime
import io
import logging
import os
import math
import traceback
from typing import Optional
from io import BytesIO

from flask import send_file
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim
import matplotlib
import requests
import numpy as np

# Use a non-interactive backend for servers without display support
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from PIL import Image as PILImage, ImageDraw, ImageFont, ImageChops
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    SimpleDocTemplate,
)

import google.generativeai as genai


logger = logging.getLogger(__name__)

FOOTER_TEXT = "krishizest powered by TerrAqua UAV"


# ==========================================
#      IMAGE PROCESSING HELPER FUNCTIONS
# ==========================================

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


# ==========================================
#           REPORTLAB & MAP FUNCTIONS
# ==========================================

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


def create_tile_mosaic(aoi, start_date, end_date, layer_type, backend_url="http://localhost:5000", 
                       zoom=14, grid_radius=2, out_size_px=1200, max_tiles=25):
    """
    Create a high-quality tile mosaic by fetching multiple tiles, stitching, 
    CROPPING to content, and adding a WHITE background.
    """
    try:
        logger.info(f"Creating {layer_type} tile mosaic with zoom {zoom}...")
        
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
            logger.warning(f"Failed to get {layer_type} tile template: {response.status_code}")
            return None, None
        
        data = response.json()
        tile_url_template = data.get('urlFormat')
        stats = data.get('stats')
        
        if not tile_url_template:
            return None, None
        
        # Calculate center of AOI
        coords = aoi['geometry']['coordinates'][0] if isinstance(aoi['geometry']['coordinates'][0][0], list) else aoi['geometry']['coordinates']
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        
        center_x, center_y = lonlat_to_tile_xy(center_lon, center_lat, zoom)
        
        tile_images = []
        successful_tiles = 0
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                if successful_tiles >= max_tiles:
                    break
                    
                tile_x = int(center_x) + dx
                tile_y = int(center_y) + dy
                
                try:
                    tile_url = tile_url_template.replace('{x}', str(tile_x)).replace('{y}', str(tile_y)).replace('{z}', str(zoom))
                    if '{s}' in tile_url:
                        tile_url = tile_url.replace('{s}', 'a')
                    
                    tile_response = requests.get(tile_url, timeout=15)
                    
                    if tile_response.status_code == 200:
                        # OPEN AS RGBA TO PRESERVE TRANSPARENCY
                        tile_img = PILImage.open(BytesIO(tile_response.content)).convert('RGBA')
                        tile_images.append((dx, dy, tile_img))
                        successful_tiles += 1
                    
                except Exception as e:
                    continue
        
        if not tile_images:
            return None, None
        
        # Calculate mosaic dimensions
        tile_size = tile_images[0][2].size[0]
        grid_size = 2 * grid_radius + 1
        mosaic_size = (grid_size * tile_size, grid_size * tile_size)
        
        # Create TRANSPARENT canvas
        mosaic = PILImage.new('RGBA', mosaic_size, color=(255, 255, 255, 0))
        
        # Place tiles
        for dx, dy, tile_img in tile_images:
            x_pos = (dx + grid_radius) * tile_size
            y_pos = (dy + grid_radius) * tile_size
            mosaic.paste(tile_img, (x_pos, y_pos))
        
        # Convert to buffer
        mosaic_buffer = BytesIO()
        mosaic.save(mosaic_buffer, format='PNG')
        
        # PROCESS: Crop + White Background
        final_buffer = process_map_image(mosaic_buffer, target_width=out_size_px)
        
        return final_buffer, stats
        
    except Exception as e:
        logger.error(f"Error creating {layer_type} tile mosaic: {e}")
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
    Add a professional colormap legend and title.
    Now handles inputs with white backgrounds correctly.
    """
    try:
        colormap_specs = {
            'ndvi': {
                'title': title or 'NDVI (Normalized Difference Vegetation Index)',
                'subtitle': 'Normalized Difference Vegetation Index (NDVI)',
                'colors': ['#0000ff', '#1e90ff', '#00ffff', '#ffff00', '#ff7f00', '#ff0000'],
                'labels': ['-1.0', '-0.3', '0.3', '1.0']
            },
            'soilmoisture': {
                'title': title or 'Soil Moisture',
                'subtitle': 'Soil Moisture (%)',
                'colors': ['#8b4513', '#d2691e', '#daa520', '#90ee90', '#006400'],
                'labels': ['0', '25', '50', '75', '100']
            },
            'cwr': {
                'title': title or 'Crop Water Requirement',
                'subtitle': 'Crop Water Requirement (mm)',
                'colors': ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
                'labels': ['0', '2.5', '5', '7.5', '10']
            }
        }
        
        spec = colormap_specs.get(layer_type, colormap_specs['ndvi'])
        
        mosaic_w, mosaic_h = mosaic_pil.size
        title_h = 40
        subtitle_h = 20
        legend_h = 60
        padding = 20
        
        new_w = mosaic_w + padding * 2
        new_h = mosaic_h + title_h + subtitle_h + legend_h + padding * 2
        
        # Create white background
        new_img = PILImage.new('RGB', (new_w, new_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(new_img)
        
        # Paste original mosaic centered
        paste_x = padding
        paste_y = title_h + subtitle_h + padding
        new_img.paste(mosaic_pil, (paste_x, paste_y))
        
        # Fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            subtitle_font = ImageFont.truetype("arial.ttf", 16)
            label_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
        
        # Title
        title_text = spec['title']
        draw.text((new_w//2, 15), title_text, fill=(0, 0, 0), font=title_font, anchor="mt")
        
        # Subtitle
        subtitle_text = spec['subtitle']
        draw.text((new_w//2, 15 + 30), subtitle_text, fill=(80, 80, 80), font=subtitle_font, anchor="mt")
        
        # Legend Bar
        legend_y = paste_y + mosaic_h + 15
        legend_width = min(600, mosaic_w - 40)
        legend_x_start = (new_w - legend_width) // 2
        legend_bar_h = 25
        
        num_colors = len(spec['colors'])
        block_w = legend_width / num_colors
        
        for i, color_hex in enumerate(spec['colors']):
            color_rgb = tuple(int(color_hex.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
            x0 = legend_x_start + i * block_w
            y0 = legend_y
            x1 = x0 + block_w
            y1 = y0 + legend_bar_h
            draw.rectangle([x0, y0, x1, y1], fill=color_rgb, outline=(0,0,0), width=1)
        
        # Labels
        label_y = legend_y + legend_bar_h + 8
        labels = spec['labels']
        for i, label in enumerate(labels):
            if len(labels) > 1:
                label_x = legend_x_start + (i * legend_width // (len(labels) - 1))
            else:
                label_x = legend_x_start + legend_width // 2
            draw.text((label_x, label_y), str(label), fill=(0,0,0), font=label_font, anchor="mt")
            
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
    """Create a ReportLab Image Flowable."""
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
        return Image(out_buf, width=draw_w, height=draw_h)
    except Exception:
        placeholder = PILImage.new('RGB', (800, 480), color=(240, 240, 240))
        pbuf = io.BytesIO()
        placeholder.save(pbuf, format='PNG')
        pbuf.seek(0)
        return Image(pbuf, width=max_width_pt, height=max_height_pt)


def get_logo_path(provided_path: str | None = None) -> str | None:
    if provided_path and os.path.exists(provided_path): return provided_path
    candidate_paths = [
        provided_path, "static/krishizest_logo.png", "krishizest_logo.png",
        "../static/krishizest_logo.png", "./static/krishizest_logo.png",
        os.path.join(os.path.dirname(__file__), "static", "krishizest_logo.png"),
        os.path.join(os.path.dirname(__file__), "krishizest_logo.png"),
    ]
    for path in candidate_paths:
        if path and os.path.exists(path): return path
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
    if total == 0: labels, sizes = [no_data_label], [1]

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
            "kc_title": "Water Use Factor",
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
            "pie_interp": "Interpretation: Higher vegetation percentage indicates good canopy cover or dense weeds.",
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
            "analysis_note": "<b>Note:</b> These recommendations should be adapted based on local conditions.",
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
            "irrigation_calendar_title": "Irrigation Calendar (Next 10 Events)",
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
            if mean < 0.2: parts.append(lang_strings["ndvi_vlow"])
            elif mean < 0.4: parts.append(lang_strings["ndvi_low"])
            elif mean < 0.6: parts.append(lang_strings["ndvi_mod"])
            elif mean < 0.8: parts.append(lang_strings["ndvi_good"])
            else: parts.append(lang_strings["ndvi_exc"])
            if trend_desc: parts.append(lang_strings["ndvi_trend"].format(trend=trend_desc))
            if recent_trend: parts.append(lang_strings["ndvi_recent"].format(trend=recent_trend))
        elif title_key == "cwr":
            parts.append(lang_strings["cwr_avg"].format(mean=mean))
            if trend_desc: parts.append(lang_strings["cwr_trend"].format(trend=trend_desc))
        elif title_key == "kc":
            parts.append(lang_strings["kc_avg"].format(mean=mean))
            if trend_desc: parts.append(lang_strings["kc_trend"].format(trend=trend_desc))
        elif title_key == "lst":
            parts.append(lang_strings["lst_avg"].format(mean=mean))
            if trend_desc: parts.append(lang_strings["lst_trend"].format(trend=trend_desc))
        elif title_key == "etc":
            parts.append(lang_strings["etc_avg"].format(mean=mean))
            if trend_desc: parts.append(lang_strings["etc_trend"].format(trend=trend_desc))
        elif title_key == "deltas":
            parts.append(lang_strings["sm_avg"].format(mean=mean))
            if trend_desc: parts.append(lang_strings["sm_trend"].format(trend=trend_desc))
        elif title_key == "soilmoisture":
            parts.append(lang_strings["soilmoisture_avg"].format(mean=mean))
            if trend_desc: parts.append(lang_strings["soilmoisture_trend"].format(trend=trend_desc))        
        else:
            parts.append(lang_strings["default_avg"].format(mean=mean))
            if trend_desc: parts.append(lang_strings["default_trend"].format(trend=trend_desc))

    parts.append(lang_strings["analysis_note"])
    result = "\n\n".join(parts)
    if len(result) > max_chars:
        result = result[: max_chars - 200] + "\n\n[Explanation trimmed due to length]"
    return result


def generate_ai_explanation(title_text: str, metric: dict, series: list, language: str, crop_name: str) -> str:
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
                   or item.get("soilmoisture_mm") or item.get("soilmoisture"))
            if date_val and val is not None:
                data_summary_parts.append(f"- {date_val}: {val}")
        if len(series) > 10: data_summary_parts.append("...and more data points.")

    data_summary = "\n".join(data_summary_parts)
    if not data_summary: data_summary = "No data available."
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
    3. A bulleted list (<ul>) with 5-6 key observations.
    4. A short, actionable recommendation for the farmer.
    Respond ONLY in {lang_name}.
    """

    try:
        api_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyCJDuNiytOFccf0V3MXCEgMKnByye8HiKY"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.replace("```html", "").replace("```", "").strip()
    except Exception as e:
        logger.error(f"AI generation failed: {e}")
        return None


def generate_ai_irrigation_analysis(calendar_data: list, summary: dict, crop_name: str) -> str:
    logger.info("Generating AI-powered irrigation analysis...")
    data_summary_parts = []
    
    if summary:
        data_summary_parts.append("Irrigation Summary:")
        if summary.get("irrigation"):
            irr = summary["irrigation"]
            data_summary_parts.append(f"- Total Events: {irr.get('total_events', 0)}")
            data_summary_parts.append(f"- Total Water: {irr.get('total_water_mm', 0):.2f} mm")
            data_summary_parts.append(f"- Water Saved: {irr.get('water_saved_mm', 0):.2f} mm")
    
    data_summary = "\n".join(data_summary_parts)
    prompt = f"""
    You are an expert irrigation agronomist analyzing a smart irrigation schedule for a farmer.
    Your task is to provide actionable insights in English.
    The analysis is for a {crop_name} crop.
    Here is the data:
    ---
    {data_summary}
    ---
    Provide a concise analysis in HTML format (<b>, <br>, <ul>, <li>).
    1. Bold Title: "<b>Smart Irrigation Analysis</b>"
    2. 3-4 sentence overview.
    3. Bulleted list of 5 key insights.
    4. Actionable recommendation.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyCJDuNiytOFccf0V3MXCEgMKnByye8HiKY"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.replace("```html", "").replace("```", "").strip()
    except Exception:
        return None


def sanitize_html_for_reportlab(html: str) -> str:
    if not html: return ""
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
            if not parts and country: parts.append(country)
            return ", ".join(parts) if parts else None
    except Exception:
        return None


def fetch_centered_tile(aoi, start_date, end_date, layer_type, backend_url="http://localhost:5000", zoom=14, tile_size=1024):
    """
    Fetch a single high-res tile centered on AOI centroid, cropped to content with white bg.
    """
    try:
        coords = aoi['geometry']['coordinates'][0] if isinstance(aoi['geometry']['coordinates'][0][0], list) else aoi['geometry']['coordinates']
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        avg_lon = sum(lons) / len(lons)
        avg_lat = sum(lats) / len(lats)
        
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
        if response.status_code != 200: return None, None
        data = response.json()
        tile_url_template = data.get('urlFormat')
        stats = data.get('stats')
        
        if not tile_url_template: return None, None
        
        n = 2.0 ** zoom
        x = int((avg_lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(avg_lat)
        y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        
        url = tile_url_template.replace("{x}", str(x)).replace("{y}", str(y)).replace("{z}", str(zoom))
        if "{s}" in url: url = url.replace("{s}", "a")
        
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200: return None, None
        
        # Process with the new cropping logic
        processed_buf = process_map_image(BytesIO(resp.content), target_width=tile_size)
        return processed_buf, stats
        
    except Exception as e:
        logger.error(f"Error fetching centered tile: {e}")
        return None, None


def generate_report_response(payload: dict):
    if not payload: raise ValueError("No JSON payload received")

    language = payload.get("language", "en").lower()
    if language not in ["en", "hi"]: language = "en"
    
    lang_strings = get_translations(language)
    crop_name = payload.get("crop_name", "Crop")
    aoi = payload.get("aoi")
    aoi_coords = None
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

    if isinstance(landcover, list): landcover = {}
    if not landcover:
        landcover = {lang_strings["landcover_veg"]: 70, lang_strings["landcover_bare"]: 30}
    
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

    logo_path = get_logo_path(payload.get("logo_path"))
    if logo_path:
        try:
            pil_img = PILImage.open(logo_path)
            pil_img.thumbnail((380, 150), PILImage.LANCZOS)
            logo_buffer = io.BytesIO()
            pil_img.save(logo_buffer, format="PNG")
            logo_buffer.seek(0)
            story.append(Image(logo_buffer, width=pil_img.width * 0.75, height=pil_img.height * 0.75, hAlign="CENTER"))
            story.append(Spacer(1, 20))
        except Exception:
            pass

    story.append(Spacer(1, 40))
    for info in (
        f"<b>{lang_strings['crop']}:</b> {crop_name}",
        f"<b>{lang_strings['period']}:</b> {start_date} to {end_date}",
        f"<b>{lang_strings['location']}:</b> {location_text or lang_strings['not_specified']}",
    ):
        story.append(Paragraph(info, ParagraphStyle("CoverInfo", fontSize=12, alignment=1, spaceAfter=8)))

    story.append(PageBreak())

    story.append(Paragraph(lang_strings["aoi_title"], ParagraphStyle("AOIHeader", fontSize=16, spaceAfter=10)))
    aoi_buffer = io.BytesIO()
    _render_aoi_map(aoi_coords if aoi_coords else None, aoi_buffer, title=lang_strings["aoi_map_title"], na_text=lang_strings["aoi_na"])
    story.append(Image(aoi_buffer, width=170 * mm, height=95 * mm))
    story.append(Spacer(1, 8))

    meta_table = Table([
        [lang_strings["aoi_area"], str(area_ha if area_ha is not None else "—")],
        [lang_strings["location"], str(location_text or "—")],
        [lang_strings["time_period"], f"{start_date} to {end_date}"],
        [lang_strings["cloud_cover"], str(cloud_cover)],
        [lang_strings["report_generated"], datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
    ], colWidths=[55 * mm, 105 * mm])
    meta_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f6f8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # --- NDVI Section ---
    story.append(Paragraph(lang_strings["ndvi_title"], ParagraphStyle("SecHeading", fontSize=16, spaceAfter=8)))
    ndvi_buffer = io.BytesIO()
    _render_ndvi_chart(series, ndvi_buffer, title=f"{crop_name} {lang_strings['ndvi_chart_title_suffix']}", na_text=lang_strings["ndvi_na"])
    story.append(Image(ndvi_buffer, width=170 * mm, height=55 * mm))
    story.append(Spacer(1, 8))

    # Fetch NDVI Map with Zoom and Crop
    try:
        logger.info("Fetching enhanced NDVI tile mosaic...")
        # Using create_tile_mosaic to get a stitched, cropped, white-bg image
        ndvi_tile_buf, ndvi_stats = create_tile_mosaic(
            aoi, start_date, end_date, 'ndvi', 
            backend_url=backend_url,
            zoom=15,  # Increased zoom
            out_size_px=1600
        )
        
        if ndvi_tile_buf:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Crop Health Map Visualization", ParagraphStyle("MapTitle", fontSize=12, alignment=1, spaceAfter=6)))
            
            # Add Legend
            ndvi_pil = PILImage.open(ndvi_tile_buf)
            styled_ndvi = add_colormap_legend(ndvi_pil, 'ndvi', 'Crop Health Map (NDVI)')
            styled_buf = BytesIO()
            styled_ndvi.save(styled_buf, format='PNG', optimize=True)
            styled_buf.seek(0)
            
            # Large image
            rl_img = make_reportlab_image(styled_buf, 170 * mm, 140 * mm)
            img_table = Table([[rl_img]], colWidths=[175 * mm])
            img_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
            story.append(img_table)
            
            if ndvi_stats:
                ndvi_mean = ndvi_stats.get('mean', 0)
                map_desc = f"<i>Current average: <b>{ndvi_mean:.3f}</b>. Green areas indicate healthy vegetation.</i>"
                story.append(Paragraph(map_desc, ParagraphStyle("MapDesc", fontSize=9, textColor=colors.grey)))
                
    except Exception as e:
        logger.debug(f"NDVI tile mosaic not available: {e}")

    ndvi_expl = generate_ai_explanation(lang_strings["ndvi_title"], metrics.get("ndvi") or {}, series, language, crop_name) or \
                generate_plain_language_explanation("ndvi", lang_strings["ndvi_title"], metrics.get("ndvi") or {}, series, lang_strings)
    
    for block in ndvi_expl.split("\n\n"):
        story.append(Paragraph(sanitize_html_for_reportlab(block), normal))
    story.append(PageBreak())

    # --- Other Metrics Section ---
    for title_key, key, alt_key in (
        ("cwr", "cwr_mm", "cwr"),
        ("kc", "kc", "kc"),
        ("lst", "lst_c", "lst"),
        ("etc", "etc_mm", "etc"),
        ("soilmoisture", "soilmoisture_mm", "soilmoisture"),
    ):
        title_text = lang_strings.get(f"{title_key}_title", title_key.upper())
        story.append(Paragraph(title_text, ParagraphStyle("ParamHeading", fontSize=14, spaceAfter=6)))

        series_vals = []
        for entry in series:
            val = entry.get(key) if entry.get(key) is not None else entry.get(alt_key)
            if val is not None:
                series_vals.append({"date": entry.get("date"), "val": val})

        if series_vals:
            dates, vals = [], []
            for item in series_vals:
                try:
                    dates.append(pd.to_datetime(item["date"]))
                    vals.append(float(item["val"]))
                except: continue
            
            if vals:
                fig, ax = plt.subplots(figsize=(7.4, 2.4))
                ax.plot(dates, vals, marker="o")
                ax.set_title(title_text, fontsize=9)
                ax.grid(alpha=0.25)
                fig.autofmt_xdate(rotation=30)
                line_buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(line_buf, format="png", bbox_inches="tight", dpi=140)
                plt.close(fig)
                line_buf.seek(0)
                story.append(Image(line_buf, width=170 * mm, height=45 * mm))

                # Soil Moisture Map
                if title_key == 'soilmoisture':
                    try:
                        sm_tile_buf, sm_stats = create_tile_mosaic(
                            aoi, start_date, end_date, 'soilmoisture', 
                            backend_url=backend_url, zoom=15, out_size_px=1600
                        )
                        if sm_tile_buf:
                            story.append(Spacer(1, 10))
                            sm_pil = PILImage.open(sm_tile_buf)
                            styled_sm = add_colormap_legend(sm_pil, 'soilmoisture', 'Soil Moisture Map')
                            styled_sm_buf = BytesIO()
                            styled_sm.save(styled_sm_buf, format='PNG', optimize=True)
                            styled_sm_buf.seek(0)
                            
                            rl_sm = make_reportlab_image(styled_sm_buf, 170 * mm, 140 * mm)
                            sm_table = Table([[rl_sm]], colWidths=[175 * mm])
                            sm_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
                            story.append(sm_table)
                    except Exception: pass

        story.append(Spacer(1, 8))
        explanation = generate_ai_explanation(title_text, metrics.get(key) or metrics.get(alt_key) or {}, series, language, crop_name) or \
                      generate_plain_language_explanation(title_key, title_text, metrics.get(key) or metrics.get(alt_key) or {}, series, lang_strings)
        
        for paragraph in explanation.split("\n\n"):
            story.append(Paragraph(sanitize_html_for_reportlab(paragraph), normal))
        story.append(PageBreak())

    # --- Irrigation Section ---
    if irrigation_calendar:
        story.append(Paragraph(lang_strings["irrigation_title"], ParagraphStyle("IrrigationHeading", fontSize=16, spaceAfter=10, textColor=colors.HexColor("#0b486b"))))
        
        irr_data = irrigation_summary.get("irrigation", {})
        urg_data = irrigation_summary.get("urgency", {})
        
        summary_table = Table([
            [lang_strings["irrigation_total_events"], str(irr_data.get("total_events", 0))],
            [lang_strings["irrigation_total_water"], f"{irr_data.get('total_water_mm', 0):.2f} mm"],
            [lang_strings["irrigation_water_saved"], f"{irr_data.get('water_saved_mm', 0):.2f} mm"],
            [lang_strings["irrigation_savings_percent"], f"{irr_data.get('savings_percent', 0):.1f}%"],
        ], colWidths=[80 * mm, 80 * mm])
        summary_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e3f2fd")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 12))
        
        timeline_buffer = io.BytesIO()
        _render_irrigation_timeline(irrigation_calendar, timeline_buffer)
        story.append(Image(timeline_buffer, width=170 * mm, height=90 * mm))
        story.append(Spacer(1, 12))

        priority_buffer = io.BytesIO()
        _render_priority_distribution(irrigation_calendar, priority_buffer)
        story.append(Image(priority_buffer, width=120 * mm, height=90 * mm, hAlign="CENTER"))
        story.append(PageBreak())
        
        ai_irr_analysis = generate_ai_irrigation_analysis(irrigation_calendar, irrigation_summary, crop_name)
        if ai_irr_analysis:
            story.append(Paragraph("AI-Powered Irrigation Recommendations", ParagraphStyle("AIHeading", fontSize=14, textColor=colors.HexColor("#0b486b"))))
            for block in ai_irr_analysis.split("\n\n"):
                story.append(Paragraph(sanitize_html_for_reportlab(block), normal))
            story.append(PageBreak())

    # --- Footer Pages ---
    story.append(Spacer(1, 80))
    story.append(Paragraph(lang_strings["thanks"], ParagraphStyle("Thanks", fontSize=30, leading=36, alignment=1)))
    story.append(Spacer(1, 40))
    story.append(Paragraph(lang_strings["contact"], ParagraphStyle("Contact", fontSize=10, alignment=1)))

    header_fn = make_draw_header(location_text or "Unknown", lang_strings["aoi_label"])
    doc.build(story, onFirstPage=lambda c, d: (header_fn(c, d), draw_footer(c, d)), onLaterPages=lambda c, d: (header_fn(c, d), draw_footer(c, d)))

    buffer.seek(0)
    filename = f"{crop_name}_CropHealthReport_{start_date}_to_{end_date}.pdf"
    return send_file(buffer, download_name=filename, as_attachment=True, mimetype="application/pdf")