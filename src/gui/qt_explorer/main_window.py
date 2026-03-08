from pathlib import Path
from typing import Optional, List

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget,
    QLineEdit, QCheckBox, QListWidget, QListWidgetItem, QTextEdit,
    QSplitter, QFormLayout, QMessageBox, QFrame, QApplication
)

# Imports internes
from src.config import DATASETS_DIR
from .data_loader import DatasetLoader
from .styles import PLOT_COLORS, apply_global_styles


class DatasetExplorerWindow(QMainWindow):
    """
    Main Application Window using PyQt6 and PyQtGraph.
    """

    def __init__(self, start_dir: Path = DATASETS_DIR):
        super().__init__()
        apply_global_styles()

        self.loader = DatasetLoader()
        self.start_dir = start_dir

        # State
        self.current_split: Optional[str] = None
        self.visual_to_real_idx: List[int] = []

        self.setWindowTitle("AI ALPHA 2 | Dataset Explorer")
        self.resize(1400, 900)
        self._init_ui()

    def _init_ui(self):
        # --- Main Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 1. Toolbar
        self._setup_toolbar(main_layout)

        # 2. Splitter (Left Controls / Right Visu)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        main_layout.addWidget(splitter)

        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        self._setup_controls(left_layout)
        splitter.addWidget(left_panel)

        # Right Panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._setup_visualization(right_layout)
        splitter.addWidget(right_panel)

        splitter.setSizes([450, 950])

    def _setup_toolbar(self, parent_layout):
        toolbar_frame = QFrame()
        toolbar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        tb_layout = QHBoxLayout(toolbar_frame)
        tb_layout.setContentsMargins(5, 5, 5, 5)

        btn_load = QPushButton("Load Dataset (.npz)")
        btn_load.clicked.connect(self.load_dataset_dialog)

        self.lbl_status = QLabel(f"Default Dir: {self.start_dir.name}/")
        self.lbl_status.setStyleSheet("color: #555; font-weight: bold;")

        tb_layout.addWidget(btn_load)
        tb_layout.addWidget(self.lbl_status)
        tb_layout.addStretch()
        parent_layout.addWidget(toolbar_frame)

    def _setup_controls(self, layout):
        # Dataset Split
        layout.addWidget(QLabel("<b>Dataset Split:</b>"))
        self.cb_split = QComboBox()
        self.cb_split.currentIndexChanged.connect(self._on_split_changed)
        layout.addWidget(self.cb_split)

        # Filters Notebook
        layout.addWidget(QLabel("<b>Filters:</b>"))
        self.tabs_filters = QTabWidget()

        # Tab Target
        tab_target = QWidget()
        f_target = QFormLayout(tab_target)
        self.input_min_y = QLineEdit()
        self.input_min_y.setPlaceholderText("-inf")
        self.input_max_y = QLineEdit()
        self.input_max_y.setPlaceholderText("+inf")
        f_target.addRow("Min y:", self.input_min_y)
        f_target.addRow("Max y:", self.input_max_y)
        self.tabs_filters.addTab(tab_target, "Target")

        # Tab Advanced
        tab_adv = QWidget()
        v_adv = QVBoxLayout(tab_adv)
        f_adv = QFormLayout()
        self.cb_adv_feature = QComboBox()
        self.cb_adv_feature.currentTextChanged.connect(self._on_adv_feat_changed)
        self.cb_adv_metric = QComboBox()
        self.cb_adv_metric.addItems(["Max", "Min", "Std"])
        f_adv.addRow("Feature:", self.cb_adv_feature)
        f_adv.addRow("Metric:", self.cb_adv_metric)
        v_adv.addLayout(f_adv)

        h_perc = QHBoxLayout()
        self.cb_adv_direction = QComboBox()
        self.cb_adv_direction.addItems(["Top", "Bottom"])
        self.input_percentile = QLineEdit("5")
        self.input_percentile.setFixedWidth(40)
        h_perc.addWidget(self.cb_adv_direction)
        h_perc.addWidget(self.input_percentile)
        h_perc.addWidget(QLabel("%"))
        v_adv.addLayout(h_perc)

        self.chk_adv_enable = QCheckBox("Enable Stat Filter")
        v_adv.addWidget(self.chk_adv_enable)
        self.tabs_filters.addTab(tab_adv, "Stats")
        layout.addWidget(self.tabs_filters)

        # Buttons
        h_btns = QHBoxLayout()
        btn_apply = QPushButton("APPLY FILTERS")
        btn_apply.setStyleSheet("background-color: #0d6efd; color: white; font-weight: bold;")
        btn_apply.clicked.connect(self.apply_filters)
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_filters)
        h_btns.addWidget(btn_apply)
        h_btns.addWidget(btn_reset)
        layout.addLayout(h_btns)

        # Samples List
        layout.addWidget(QLabel("<b>Samples (Flights):</b>"))
        self.list_samples = QListWidget()
        self.list_samples.setAlternatingRowColors(True)
        self.list_samples.itemSelectionChanged.connect(self._on_sample_selected)
        layout.addWidget(self.list_samples)

        # Features List
        layout.addWidget(QLabel("<b>Features to Plot:</b>"))
        self.list_features = QListWidget()
        self.list_features.itemChanged.connect(self.refresh_plot)
        layout.addWidget(self.list_features)

    def _setup_visualization(self, layout):
        self.text_details = QTextEdit()
        self.text_details.setMaximumHeight(80)
        self.text_details.setReadOnly(True)
        self.text_details.setStyleSheet("font-family: Consolas; background: #f8f9fa;")
        layout.addWidget(self.text_details)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(10, 10))
        layout.addWidget(self.plot_widget)

    # --- LOGIC ---

    def load_dataset_dialog(self):
        f_path, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset", str(self.start_dir), "Numpy Zip (*.npz)"
        )
        if not f_path: return

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.loader.load_file(f_path)
            self.lbl_status.setText(f"Loaded: <b>{self.loader.filename}</b>")

            self._populate_splits()
            self._populate_features()
            self._update_adv_options()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _populate_splits(self):
        self.cb_split.blockSignals(True)
        self.cb_split.clear()
        for s in self.loader.get_splits():
            count = self.loader.data[f"X_{s}"].shape[0]
            self.cb_split.addItem(f"{s.upper()} ({count})", s)
        self.cb_split.blockSignals(False)
        if self.cb_split.count() > 0: self._on_split_changed(0)

    def _populate_features(self):
        self.list_features.clear()
        for i, feat in enumerate(self.loader.feature_names):
            item = QListWidgetItem(feat)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if i < 3 else Qt.CheckState.Unchecked)
            self.list_features.addItem(item)

    def _update_adv_options(self):
        self.cb_adv_feature.clear()
        self.cb_adv_feature.addItem("Mask Length")
        self.cb_adv_feature.addItems(self.loader.feature_names)

    def _on_split_changed(self, idx):
        self.current_split = self.cb_split.currentData()
        self.input_min_y.clear()
        self.input_max_y.clear()
        self.chk_adv_enable.setChecked(False)
        self._update_sample_list(None)

    def _on_adv_feat_changed(self, text):
        self.cb_adv_metric.setEnabled(text != "Mask Length")

    def _update_sample_list(self, indices):
        self.list_samples.clear()
        if not self.current_split: return

        if indices is None:
            n = self.loader.data[f"X_{self.current_split}"].shape[0]
            indices = np.arange(n)

        self.visual_to_real_idx = indices
        y_data = self.loader.data[f"y_{self.current_split}"]

        # Add to list
        # Optimization: for very large lists, use QAbstractListModel, but OK for <5k
        items = []
        for idx in indices:
            fid = self.loader.get_flight_id(self.current_split, idx)
            val = y_data[idx]
            items.append(f"[{fid}] y={val:.4f}")
        self.list_samples.addItems(items)

        self.lbl_status.setText(f"File: {self.loader.filename} | Showing {len(indices)} samples")

    def apply_filters(self):
        if not self.current_split: return
        try:
            # 1. Target
            s_min, s_max = self.input_min_y.text(), self.input_max_y.text()
            v_min = float(s_min) if s_min else float('-inf')
            v_max = float(s_max) if s_max else float('inf')

            y = self.loader.data[f"y_{self.current_split}"]
            indices = np.where((y >= v_min) & (y <= v_max))[0]

            # 2. Advanced
            if self.chk_adv_enable.isChecked():
                adv_idx = self._compute_adv_indices()
                if adv_idx is not None:
                    indices = np.intersect1d(indices, adv_idx)

            self._update_sample_list(indices)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid numeric values.")

    def _compute_adv_indices(self):
        feat = self.cb_adv_feature.currentText()
        metric = self.cb_adv_metric.currentText()
        direct = self.cb_adv_direction.currentText()
        try:
            perc = float(self.input_percentile.text())
        except ValueError:
            return None

        X = self.loader.data[f"X_{self.current_split}"]
        mask = self.loader.data[f"mask_{self.current_split}"]
        stats = None

        if feat == "Mask Length":
            stats = np.sum(mask, axis=1)
        elif feat in self.loader.feature_names:
            f_idx = self.loader.feature_names.index(feat)
            raw = X[:, :, f_idx].astype(np.float32).copy()
            raw[mask == 0] = np.nan
            with np.errstate(all='ignore'):
                if metric == "Max":
                    stats = np.nanmax(raw, axis=1)
                elif metric == "Min":
                    stats = np.nanmin(raw, axis=1)
                elif metric == "Std":
                    stats = np.nanstd(raw, axis=1)

        if stats is None: return None

        valid = stats[~np.isnan(stats)]
        if len(valid) == 0: return np.array([])

        cutoff = np.percentile(valid, 100 - perc) if direct == "Top" else np.percentile(valid, perc)
        return np.where(stats >= cutoff)[0] if direct == "Top" else np.where(stats <= cutoff)[0]

    def reset_filters(self):
        self.input_min_y.clear()
        self.input_max_y.clear()
        self.chk_adv_enable.setChecked(False)
        self._update_sample_list(None)

    def _on_sample_selected(self):
        sel = self.list_samples.selectedIndexes()
        if not sel: return
        r_idx = sel[0].row()
        if r_idx < len(self.visual_to_real_idx):
            self._render(self.visual_to_real_idx[r_idx])

    def refresh_plot(self):
        self._on_sample_selected()

    def _render(self, idx):
        split = self.current_split
        X = self.loader.data[f"X_{split}"][idx]
        mask = self.loader.data[f"mask_{split}"][idx]
        fid = self.loader.get_flight_id(split, idx)

        valid = int(np.sum(mask))

        # Details
        self.text_details.setText(
            f"ID: {fid}\nSPLIT: {split} | IDX: {idx}\n"
            f"TARGET: {self.loader.data[f'y_{split}'][idx]:.6f}\n"
            f"LENGTH: {valid} ticks"
        )

        # Plot
        self.plot_widget.clear()

        active = []
        for i in range(self.list_features.count()):
            it = self.list_features.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                active.append(it.text())

        if not active: return

        # Draw Padding Region
        if valid < len(mask):
            r = pg.LinearRegionItem([valid, len(mask)], movable=False, brush=pg.mkBrush(220, 220, 220, 100))
            self.plot_widget.addItem(r)

        # Draw Lines
        for i, f_name in enumerate(active):
            if f_name in self.loader.feature_names:
                fi = self.loader.feature_names.index(f_name)
                col = PLOT_COLORS[i % len(PLOT_COLORS)]

                # --- NEW LOGIC: Use NaN to break lines at padding ---
                y_plot = X[:, fi].astype(float).copy()
                y_plot[mask == 0] = np.nan
                # ----------------------------------------------------

                self.plot_widget.plot(y_plot, pen=pg.mkPen(col, width=2), name=f_name)