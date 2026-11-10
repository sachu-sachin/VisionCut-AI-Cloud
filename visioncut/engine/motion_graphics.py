"""
Professional Motion Graphics Generator for VisionCut AI Cloud.
Generates standalone, scalable vector graphics (SVG) and MLT keyframe data
for rectangles, arrows, circles, cursor effects, callout badges, and camera zooms.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from visioncut.core.models import BoundingBox, AnnotationItem
from visioncut.core.config import MotionGraphicsSettings


class MotionGraphicsEngine:
    """
    Renders resolution-independent SVG assets and camera transformation keyframes.
    All graphics are saved as individual editable files in the project folder.
    """

    def __init__(self, settings: Optional[MotionGraphicsSettings] = None):
        self.settings = settings or MotionGraphicsSettings()

    def generate_rectangle_svg(
        self,
        output_path: str,
        bbox: BoundingBox,
        width: int = 1920,
        height: int = 1080,
        color: str = "#FFCC00",
        border_width: int = 5,
        corner_radius: int = 12,
        animated: bool = True
    ) -> str:
        """Generates an animated glowing rounded rectangle overlay SVG."""
        rect_x, rect_y, rect_w, rect_h = bbox.to_pixel_rect(width, height)

        # SVG markup with subtle pulse animation and glow filter
        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>
  <rect x="{rect_x}" y="{rect_y}" width="{rect_w}" height="{rect_h}"
        rx="{corner_radius}" ry="{corner_radius}"
        fill="none" stroke="{color}" stroke-width="{border_width}"
        filter="url(#glow)">
    {f'<animate attributeName="stroke-opacity" values="0.4;1.0;0.4" dur="1.8s" repeatCount="indefinite" />' if animated else ''}
  </rect>
</svg>'''
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(svg_content, encoding="utf-8")
        return str(out_p)

    def generate_circle_svg(
        self,
        output_path: str,
        bbox: BoundingBox,
        width: int = 1920,
        height: int = 1080,
        color: str = "#00E5FF",
        border_width: int = 5,
        animated: bool = True
    ) -> str:
        """Generates an animated circular/elliptical highlight ring SVG."""
        rect_x, rect_y, rect_w, rect_h = bbox.to_pixel_rect(width, height)
        cx = rect_x + rect_w / 2.0
        cy = rect_y + rect_h / 2.0
        rx = max(20.0, rect_w / 1.7)
        ry = max(20.0, rect_h / 1.7)

        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <filter id="glow-circle" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="5" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>
  <ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"
           fill="none" stroke="{color}" stroke-width="{border_width}"
           filter="url(#glow-circle)">
    {f'<animate attributeName="rx" values="{rx:.1f};{rx*1.08:.1f};{rx:.1f}" dur="1.5s" repeatCount="indefinite" />' if animated else ''}
    {f'<animate attributeName="ry" values="{ry:.1f};{ry*1.08:.1f};{ry:.1f}" dur="1.5s" repeatCount="indefinite" />' if animated else ''}
  </ellipse>
</svg>'''
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(svg_content, encoding="utf-8")
        return str(out_p)

    def generate_arrow_svg(
        self,
        output_path: str,
        start_pt: Tuple[float, float],  # normalized (x, y)
        end_pt: Tuple[float, float],    # normalized (x, y)
        width: int = 1920,
        height: int = 1080,
        color: str = "#FF3366",
        border_width: int = 6,
        curved: bool = True
    ) -> str:
        """Generates a smooth directional pointer arrow pointing from start to target."""
        x1 = start_pt[0] * width
        y1 = start_pt[1] * height
        x2 = end_pt[0] * width
        y2 = end_pt[1] * height

        # If curved, calculate quadratic bezier control point
        if curved:
            dx = x2 - x1
            dy = y2 - y1
            cx = (x1 + x2) / 2.0 - dy * 0.25
            cy = (y1 + y2) / 2.0 + dx * 0.25
            path_d = f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"
        else:
            path_d = f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}"

        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <polygon points="0 0, 10 5, 0 10" fill="{color}" />
    </marker>
  </defs>
  <path d="{path_d}" fill="none" stroke="{color}" stroke-width="{border_width}"
        marker-end="url(#arrowhead)" stroke-linecap="round" />
</svg>'''
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(svg_content, encoding="utf-8")
        return str(out_p)

    def generate_cursor_ripple_svg(
        self,
        output_path: str,
        center_pt: Tuple[float, float],  # normalized (x, y)
        width: int = 1920,
        height: int = 1080,
        color: str = "#00E5FF"
    ) -> str:
        """Generates an animated expanding click ripple circle."""
        cx = center_pt[0] * width
        cy = center_pt[1] * height

        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="15" fill="none" stroke="{color}" stroke-width="4">
    <animate attributeName="r" values="12;60" dur="0.6s" repeatCount="indefinite" />
    <animate attributeName="opacity" values="1;0" dur="0.6s" repeatCount="indefinite" />
  </circle>
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}" />
</svg>'''
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(svg_content, encoding="utf-8")
        return str(out_p)

    def generate_callout_svg(
        self,
        output_path: str,
        text: str,
        bbox: Optional[BoundingBox] = None,
        width: int = 1920,
        height: int = 1080,
        callout_type: str = "label",  # "label", "shortcut", "step", "warning", "success"
        step_number: Optional[int] = None
    ) -> str:
        """Generates a text callout badge with modern UI styling."""
        if bbox:
            bx, by, bw, bh = bbox.to_pixel_rect(width, height)
            pos_x = max(40, bx)
            pos_y = max(50, by - 55)  # Position right above the bounding box
        else:
            pos_x, pos_y = 120, 100

        badge_colors = {
            "label": ("#1E222D", "#00E5FF", "#FFFFFF"),
            "shortcut": ("#282C34", "#98C379", "#61AFEF"),
            "step": ("#007ACC", "#FFFFFF", "#FFFFFF"),
            "warning": ("#D32F2F", "#FF5252", "#FFFFFF"),
            "success": ("#2E7D32", "#4CAF50", "#FFFFFF")
        }
        bg_col, border_col, txt_col = badge_colors.get(callout_type, ("#1E222D", "#FFCC00", "#FFFFFF"))

        display_text = f"STEP {step_number}: {text}" if step_number else text
        approx_w = max(180, len(display_text) * 14 + 40)
        approx_h = 48

        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <g transform="translate({pos_x}, {pos_y})">
    <rect width="{approx_w}" height="{approx_h}" rx="8" ry="8"
          fill="{bg_col}" fill-opacity="0.92"
          stroke="{border_col}" stroke-width="2.5" />
    <text x="20" y="30" font-family="Segoe UI, Helvetica, Arial, sans-serif"
          font-size="18" font-weight="600" fill="{txt_col}">
      {display_text}
    </text>
  </g>
</svg>'''
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(svg_content, encoding="utf-8")
        return str(out_p)

    def compute_zoom_geometry(
        self,
        target_bbox: BoundingBox,
        video_w: int = 1920,
        video_h: int = 1080,
        zoom_factor: float = 1.35
    ) -> Dict[str, Any]:
        """
        Calculates keyframe coordinates for Kdenlive / MLT affine transformation
        to smoothly center and zoom onto target bounding box.
        """
        cx = target_bbox.center_x * video_w
        cy = target_bbox.center_y * video_h

        crop_w = int(video_w / zoom_factor)
        crop_h = int(video_h / zoom_factor)

        crop_x = int(max(0, min(video_w - crop_w, cx - crop_w / 2)))
        crop_y = int(max(0, min(video_h - crop_h, cy - crop_h / 2)))

        return {
            "zoom_factor": zoom_factor,
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "original_w": video_w,
            "original_h": video_h,
            # MLT geometry string format: "X Y W H:opacity"
            "mlt_start_geom": f"0 0 {video_w} {video_h}:1.0",
            "mlt_zoomed_geom": f"{-crop_x * zoom_factor:.0f} {-crop_y * zoom_factor:.0f} {video_w * zoom_factor:.0f} {video_h * zoom_factor:.0f}:1.0",
            "mlt_end_geom": f"0 0 {video_w} {video_h}:1.0"
        }

