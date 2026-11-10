"""
Interactive Editing Decisions Table Widget for VisionCut AI Cloud.
Allows users to review, inspect, approve, or reject AI-proposed cuts and annotations.
"""

from __future__ import annotations
from typing import List, Optional
from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, QWidget,
    QHBoxLayout, QCheckBox, QLabel
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from visioncut.core.models import EditingDecision, ConfidenceLevel


class DecisionTableWidget(QTableWidget):
    """Interactive table showing AI editing suggestions with approve/reject toggles."""

    decisionToggled = Signal(str, bool)  # decision_id, approved

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.decisions: List[EditingDecision] = []
        self._setup_table()

    def _setup_table(self):
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels([
            "Approve", "Time", "Duration", "Action", "Target", "Confidence", "Reason"
        ])
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def load_decisions(self, decisions: List[EditingDecision]):
        self.decisions = decisions
        self.setRowCount(len(decisions))

        for row, dec in enumerate(decisions):
            # Checkbox widget
            chk_widget = QWidget()
            layout = QHBoxLayout(chk_widget)
            layout.setContentsMargins(8, 0, 8, 0)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk = QCheckBox()
            chk.setChecked(dec.approved)
            dec_id = dec.decision_id

            def make_handler(did):
                return lambda state: self._on_checkbox_changed(did, state == Qt.CheckState.Checked.value)

            chk.stateChanged.connect(make_handler(dec_id))
            layout.addWidget(chk)
            self.setCellWidget(row, 0, chk_widget)

            # Time
            t_str = f"{dec.start_time:.1f}s - {dec.end_time:.1f}s"
            item_time = QTableWidgetItem(t_str)
            item_time.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, 1, item_time)

            # Duration
            dur_str = f"{(dec.end_time - dec.start_time):.1f}s"
            item_dur = QTableWidgetItem(dur_str)
            item_dur.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, 2, item_dur)

            # Action
            action_label = dec.action_type.value.replace("_", " ").title()
            item_action = QTableWidgetItem(action_label)
            item_action.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, 3, item_action)

            # Target
            target_str = dec.text_content or (dec.sfx_type.value if dec.sfx_type else "Timeline")
            item_target = QTableWidgetItem(target_str)
            self.setItem(row, 4, item_target)

            # Confidence Badge
            pct = int(dec.confidence * 100)
            item_conf = QTableWidgetItem(f"{pct}% ({dec.confidence_level.value.upper()})")
            item_conf.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if dec.confidence_level == ConfidenceLevel.HIGH:
                item_conf.setForeground(QColor("#00E676"))
            elif dec.confidence_level == ConfidenceLevel.MEDIUM:
                item_conf.setForeground(QColor("#FFCC00"))
            else:
                item_conf.setForeground(QColor("#9E9E9E"))
            self.setItem(row, 5, item_conf)

            # Reason
            item_reason = QTableWidgetItem(dec.reason)
            self.setItem(row, 6, item_reason)

    def _on_checkbox_changed(self, decision_id: str, approved: bool):
        for dec in self.decisions:
            if dec.decision_id == decision_id:
                dec.approved = approved
                break
        self.decisionToggled.emit(decision_id, approved)

    def select_all(self, approve: bool = True):
        for r in range(self.rowCount()):
            cell = self.cellWidget(r, 0)
            if cell:
                chk = cell.findChild(QCheckBox)
                if chk:
                    chk.setChecked(approve)

