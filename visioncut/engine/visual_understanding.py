"""
Visual Understanding Engine for VisionCut AI Cloud.
Parses, filters, validates, and enriches visual events detected by Cloud VLM workers
or local preprocessing heuristics into structured, actionable editing events.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from visioncut.core.models import (
    UIEvent, UIEventType, EditingActionType, ConfidenceLevel, BoundingBox
)


class VisualUnderstandingEngine:
    """
    Transforms raw visual detections into refined tutorial editing events.
    Enforces conservative editing rules: never invent UI elements without visual support,
    never remove content solely based on static-screen detection.
    """

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold

    def parse_raw_events(self, raw_events: List[Dict[str, Any]]) -> List[UIEvent]:
        """Validates and parses raw dictionary events into UIEvent models."""
        parsed_events: List[UIEvent] = []

        for item in raw_events:
            try:
                # 1. Parse Bounding Box and ensure strictly normalized [0, 1]
                raw_bbox = item.get("bbox", {})
                ymin = max(0.0, min(1.0, float(raw_bbox.get("ymin", 0.0))))
                xmin = max(0.0, min(1.0, float(raw_bbox.get("xmin", 0.0))))
                ymax = max(0.0, min(1.0, float(raw_bbox.get("ymax", 1.0))))
                xmax = max(0.0, min(1.0, float(raw_bbox.get("xmax", 1.0))))

                if ymax < ymin:
                    ymin, ymax = ymax, ymin
                if xmax < xmin:
                    xmin, xmax = xmax, xmin

                bbox = BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)

                # 2. Parse confidence
                conf = float(item.get("confidence", 0.85))
                conf = max(0.0, min(1.0, conf))

                # Determine Confidence Level
                if conf >= 0.85:
                    conf_level = ConfidenceLevel.HIGH
                elif conf >= 0.50:
                    conf_level = ConfidenceLevel.MEDIUM
                else:
                    conf_level = ConfidenceLevel.LOW

                # Filter out low-confidence hallucinated detections
                if conf < self.confidence_threshold:
                    continue

                # 3. Determine Event Type
                raw_type = item.get("event_type", "ui_element_focus")
                try:
                    event_type = UIEventType(raw_type)
                except ValueError:
                    event_type = UIEventType.UI_ELEMENT_FOCUS

                # 4. Determine Recommended Action
                raw_action = item.get("recommended_action")
                if raw_action:
                    try:
                        rec_action = EditingActionType(raw_action)
                    except ValueError:
                        rec_action = self._infer_action(event_type)
                else:
                    rec_action = self._infer_action(event_type)

                # 5. Build reason if empty
                reason = item.get("reason") or self._generate_reason(event_type, item.get("ui_target", "element"))

                event = UIEvent(
                    event_id=item.get("event_id", f"evt_{len(parsed_events):04d}"),
                    timestamp=round(float(item.get("timestamp", 0.0)), 2),
                    duration=round(float(item.get("duration", 1.5)), 2),
                    event_type=event_type,
                    description=item.get("description", f"Visual event at {item.get('ui_target', 'UI')}"),
                    ui_target=item.get("ui_target", "Interface Element"),
                    bbox=bbox,
                    confidence=conf,
                    confidence_level=conf_level,
                    recommended_action=rec_action,
                    reason=reason,
                    approved=True
                )
                parsed_events.append(event)

            except Exception as e:
                print(f"[ENGINE WARNING] Skipping invalid event: {e}")

        # Sort chronologically
        return sorted(parsed_events, key=lambda e: e.timestamp)

    def _infer_action(self, event_type: UIEventType) -> EditingActionType:
        if event_type == UIEventType.BUTTON_CLICK:
            return EditingActionType.HIGHLIGHT_BOX
        elif event_type == UIEventType.TERMINAL_EXEC:
            return EditingActionType.ZOOM_PAN
        elif event_type == UIEventType.CODE_EDIT:
            return EditingActionType.PRESERVE_ORIGINAL
        elif event_type == UIEventType.ERROR_ALERT:
            return EditingActionType.TEXT_CALLOUT
        elif event_type == UIEventType.RESULT_DISPLAY:
            return EditingActionType.HIGHLIGHT_BOX
        elif event_type == UIEventType.IDLE_WAIT:
            return EditingActionType.SHORTEN_WAIT
        elif event_type == UIEventType.SETTINGS_CHANGE:
            return EditingActionType.DRAW_ARROW
        return EditingActionType.HIGHLIGHT_BOX

    def _generate_reason(self, event_type: UIEventType, target: str) -> str:
        reasons = {
            UIEventType.BUTTON_CLICK: f"Direct viewer focus to clicked '{target}' control",
            UIEventType.TERMINAL_EXEC: f"Emphasize terminal command execution and results",
            UIEventType.CODE_EDIT: f"Preserve code changes clearly without visual obstruction",
            UIEventType.ERROR_ALERT: f"Highlight critical error message and diagnostic steps",
            UIEventType.RESULT_DISPLAY: f"Showcase final successful tutorial outcome",
            UIEventType.IDLE_WAIT: f"Shorten non-productive wait pause while preserving context",
            UIEventType.SETTINGS_CHANGE: f"Indicate configuration modification on '{target}'",
            UIEventType.BROWSER_NAV: f"Follow browser navigation transition"
        }
        return reasons.get(event_type, f"Visual emphasis for {target}")

