"""
DAVID ORACLE v2.0 — Desktop Launcher (PyQt5)
=============================================
Run with: python desktop_launcher.py

Full-blown dashboard with:
- 3 action buttons (Fetch Spot, Sync Data, Train Models)
- Prophet Dashboard (verdict, regime, whipsaw, S/R, sentiment)
- Forecast & Ranges (7-day / 30-day probability cones)
- Strategy Lab (Iron Condor, Bounce Calculator)
- Data Inspector (CSV + Model status)
- Real-time log console
"""

import sys
import os
from datetime import datetime

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QTextEdit, QTableWidget,
    QTableWidgetItem, QSplitter, QFrame, QGroupBox, QGridLayout,
    QSpinBox, QSlider, QHeaderView, QSizePolicy, QSystemTrayIcon,
    QMenu, QAction, QStatusBar, QProgressBar, QShortcut, QScrollArea,
    QComboBox, QDateEdit, QLineEdit, QDoubleSpinBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSize, QDate
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QIcon, QKeySequence, QPixmap
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

import david_desktop as backend

# ═══════════════════════════════════════════════════════════════════════════════
# STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DARK_BG = "#0B0E14"
DARK_CARD = "#151A22"
DARK_BORDER = "#1F2937"
ACCENT_GREEN = "#00FF7F"
ACCENT_RED = "#FF4B4B"
ACCENT_GOLD = "#FFD700"
ACCENT_CYAN = "#00CED1"
TEXT_PRIMARY = "#FFFFFF"
TEXT_DIM = "#8B8D97"
TEXT_SEC = "#A0AEC0"

GLOBAL_STYLE = f"""
    QMainWindow, QWidget {{
        background-color: {DARK_BG};
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }}
    QTabWidget::pane {{
        border: 1px solid {DARK_BORDER};
        background-color: {DARK_BG};
    }}
    QTabBar::tab {{
        background-color: {DARK_CARD};
        color: {TEXT_DIM};
        padding: 10px 20px;
        border: 1px solid {DARK_BORDER};
        border-bottom: none;
        font-size: 13px;
        font-weight: bold;
    }}
    QTabBar::tab:selected {{
        background-color: {DARK_BG};
        color: {ACCENT_CYAN};
        border-bottom: 2px solid {ACCENT_CYAN};
    }}
    QPushButton {{
        background-color: {DARK_CARD};
        color: {TEXT_PRIMARY};
        border: 1px solid {DARK_BORDER};
        border-radius: 6px;
        padding: 12px 20px;
        font-size: 13px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: #252836;
        border-color: {ACCENT_CYAN};
    }}
    QPushButton:pressed {{
        background-color: #1E2130;
    }}
    QPushButton:disabled {{
        color: #555;
        background-color: #151820;
    }}
    QTextEdit {{
        background-color: #0A0D12;
        color: #B0B0B0;
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 11px;
        padding: 4px;
    }}
    QTableWidget {{
        background-color: {DARK_CARD};
        color: {TEXT_PRIMARY};
        border: 1px solid {DARK_BORDER};
        gridline-color: {DARK_BORDER};
        font-size: 12px;
    }}
    QTableWidget::item {{
        padding: 6px;
    }}
    QHeaderView::section {{
        background-color: #12151C;
        color: {ACCENT_CYAN};
        border: 1px solid {DARK_BORDER};
        padding: 6px;
        font-weight: bold;
        font-size: 12px;
    }}
    QGroupBox {{
        border: none;
        margin-top: 15px;
        padding-top: 15px;
        font-size: 15px;
        font-weight: bold;
        color: {TEXT_PRIMARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 0px;
        padding: 0px;
        color: {ACCENT_CYAN};
    }}
    QLabel {{
        color: {TEXT_PRIMARY};
    }}
    QStatusBar {{
        background-color: #0A0D12;
        color: {TEXT_DIM};
        border-top: 1px solid {DARK_BORDER};
    }}
    QProgressBar {{
        background-color: {DARK_CARD};
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
        text-align: center;
        color: {TEXT_PRIMARY};
        font-size: 11px;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT_CYAN};
        border-radius: 3px;
    }}
    QSpinBox {{
        background-color: {DARK_CARD};
        color: {TEXT_PRIMARY};
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
        padding: 6px;
        font-size: 13px;
    }}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER THREADS
# ═══════════════════════════════════════════════════════════════════════════════

class WorkerThread(QThread):
    """Generic worker thread for background operations."""
    finished = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        # Redirect logs to signal
        def log_to_gui(msg):
            self.log_signal.emit(msg)
        
        backend.set_log_callback(log_to_gui)
        result = self.func(*self.args, **self.kwargs)
        self.finished.emit(result)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC CARD WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class MetricCard(QFrame):
    """A styled card showing a metric value with title."""
    
    def __init__(self, title="", value="—", color=ACCENT_CYAN, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(21, 26, 34, 0.7);
                border: 1px solid rgba(45, 49, 57, 0.8);
                border-top: 2px solid rgba(100, 110, 130, 0.3);
                border-radius: 12px;
            }}
        """)
        self.setMinimumHeight(100)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignCenter)
        
        layout.addStretch()
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addStretch()
    
    def set_value(self, value, color=None):
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class ChartCanvas(FigureCanvas):
    """Matplotlib canvas embedded in PyQt5 with dark theme."""
    
    def __init__(self, width=5, height=3, parent=None):
        self.fig = Figure(figsize=(width, height), facecolor=DARK_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(DARK_BG)
        self.ax.tick_params(colors=TEXT_DIM, labelsize=9)
        self.ax.spines['bottom'].set_color(DARK_BORDER)
        self.ax.spines['top'].set_color(DARK_BORDER)
        self.ax.spines['left'].set_color(DARK_BORDER)
        self.ax.spines['right'].set_color(DARK_BORDER)
        super().__init__(self.fig)
        self.fig.tight_layout(pad=1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class DavidOracleWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦅 David Oracle v2.1 — Desktop")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # State
        self.prediction = None
        self.worker = None
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self.on_fetch_spot)
        
        # Build UI
        self._build_ui()
        self._setup_shortcuts()
        self._setup_status_bar()
        
        # Initial data status check
        QTimer.singleShot(500, self.refresh_data_inspector)
    
    # ─────────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # LEFT PANEL (buttons + mini inspector)
        left_panel = self._build_left_panel()
        left_panel.setFixedWidth(260)
        
        # RIGHT AREA (tabs + log console)
        right_splitter = QSplitter(Qt.Vertical)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tab_dashboard = self._build_dashboard_tab()
        self.tab_forecast = self._build_forecast_tab()
        self.tab_strategy = self._build_strategy_tab()
        self.tab_intraday = self._build_intraday_tab()
        self.tab_inspector = self._build_inspector_tab()
        
        self.tabs.addTab(self.tab_dashboard, "🦅 Dashboard")
        self.tabs.addTab(self._build_command_center_tab(), "🚦 Command Center")
        self.tabs.addTab(self.tab_forecast, "📈 Forecast")
        self.tabs.addTab(self.tab_intraday, "⚡ Intraday ML")
        self.tabs.addTab(self.tab_strategy, "🧪 Strategy Lab")
        self.tabs.addTab(self._build_position_manager_tab(), "🛡️ Position Manager")
        self.tabs.addTab(self._build_trade_recommendations_tab(), "💡 Trade Recommendations")
        self.tabs.addTab(self._build_price_action_tab(), "📉 Price Action")
        self.tabs.addTab(self._build_war_room_tab(), "⚔️ War Room")
        self.tabs.addTab(self._build_weekly_report_tab(), "📊 Weekly Report")
        self.tabs.addTab(self.tab_inspector, "📋 Data Inspector")
        self.tabs.addTab(self._build_codex_tab(), "📖 David Codex")
        
        # Log console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(180)
        self.log_console.setPlaceholderText("Log output will appear here...")
        
        right_splitter.addWidget(self.tabs)
        right_splitter.addWidget(self.log_console)
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_splitter, stretch=1)
    
    def _build_left_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_CARD};
                border: 1px solid {DARK_BORDER};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Logo / Title
        title = QLabel("🦅 DAVID ORACLE")
        title.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 18px; font-weight: bold; border: none;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        version = QLabel("v2.1 — Desktop Edition")
        version.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; border: none;")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)
        
        # Spot info
        self.spot_label = QLabel("NIFTY: —")
        self.spot_label.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 16px; font-weight: bold; border: none; padding: 6px;")
        self.spot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.spot_label)
        
        self.vix_label = QLabel("VIX: —")
        self.vix_label.setStyleSheet(f"color: {ACCENT_GOLD}; font-size: 14px; font-weight: bold; border: none; padding: 2px;")
        self.vix_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.vix_label)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"background-color: {DARK_BORDER}; border: none; max-height: 1px;")
        layout.addWidget(separator)
        
        # BUTTON 1: Fetch Spot
        self.btn_fetch = QPushButton("🔴  Fetch Spot + Predict")
        self.btn_fetch.setToolTip("F5 — Fetch live NIFTY + VIX and run instant prediction")
        self.btn_fetch.clicked.connect(self.on_fetch_spot)
        self.btn_fetch.setStyleSheet(self.btn_fetch.styleSheet() + f"""
            QPushButton {{
                border-left: 3px solid {ACCENT_RED};
            }}
        """)
        layout.addWidget(self.btn_fetch)
        
        # BUTTON 2: Sync Data
        self.btn_sync = QPushButton("📊  Sync Full Data")
        self.btn_sync.setToolTip("F6 — Download all market data CSVs")
        self.btn_sync.clicked.connect(self.on_sync_data)
        self.btn_sync.setStyleSheet(self.btn_sync.styleSheet() + f"""
            QPushButton {{
                border-left: 3px solid {ACCENT_CYAN};
            }}
        """)
        layout.addWidget(self.btn_sync)
        
        # BUTTON 3: Train Models
        self.btn_train = QPushButton("🧠  Train Models")
        self.btn_train.setToolTip("F7 — Train all ML models from CSVs")
        self.btn_train.clicked.connect(self.on_train_models)
        self.btn_train.setStyleSheet(self.btn_train.styleSheet() + f"""
            QPushButton {{
                border-left: 3px solid {ACCENT_GOLD};
            }}
        """)
        layout.addWidget(self.btn_train)
        
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setStyleSheet(f"background-color: {DARK_BORDER}; border: none; max-height: 1px;")
        layout.addWidget(separator2)
        
        # BUTTON 4: Sync 15m Data
        self.btn_sync_15m = QPushButton("📉  Sync 15m Data")
        self.btn_sync_15m.setToolTip("F8 — Fetch latest 15-minute NIFTY + VIX candles")
        self.btn_sync_15m.clicked.connect(self.on_sync_15m)
        self.btn_sync_15m.setStyleSheet(self.btn_sync_15m.styleSheet() + f"""
            QPushButton {{
                border-left: 3px solid {ACCENT_GREEN};
            }}
        """)
        layout.addWidget(self.btn_sync_15m)
        
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setStyleSheet(f"background-color: {DARK_BORDER}; border: none; max-height: 1px;")
        layout.addWidget(separator3)
        
        # Mini data inspector
        inspector_title = QLabel("📁 Data Status")
        inspector_title.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 12px; font-weight: bold; border: none;")
        layout.addWidget(inspector_title)
        
        self.mini_inspector = QTableWidget(5, 2)
        self.mini_inspector.setHorizontalHeaderLabels(["Data", "Latest"])
        self.mini_inspector.horizontalHeader().setStretchLastSection(True)
        self.mini_inspector.verticalHeader().setVisible(False)
        self.mini_inspector.setMaximumHeight(170)
        self.mini_inspector.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mini_inspector.setStyleSheet(f"""
            QTableWidget {{ border: none; background-color: transparent; }}
            QTableWidget::item {{ padding: 3px; font-size: 11px; }}
        """)
        layout.addWidget(self.mini_inspector)
        
        # Model status indicator
        self.model_status_label = QLabel("⚪ Models: Checking...")
        self.model_status_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; border: none;")
        layout.addWidget(self.model_status_label)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        self.progress.setMaximumHeight(18)
        layout.addWidget(self.progress)
        
        layout.addStretch()
        
        # Timestamp
        self.last_update_label = QLabel("Last update: Never")
        self.last_update_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9px; border: none;")
        self.last_update_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.last_update_label)
        
        return panel
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    
    def _create_sr_tile(self):
        f = QFrame()
        f.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(21, 26, 34, 0.7);
                border: 1px solid rgba(45, 49, 57, 0.8);
                border-top: 2px solid rgba(100, 110, 130, 0.5);
                border-radius: 8px;
            }}
        """)
        lay = QVBoxLayout(f)
        lay.setContentsMargins(10, 15, 10, 15)
        lay.setSpacing(8)
        
        lbl_level = QLabel("—")
        lbl_level.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: bold; border: none; background: transparent;")
        lbl_level.setAlignment(Qt.AlignCenter)
        
        lbl_stats = QLabel("Dist: —  |  Str: —")
        lbl_stats.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; border: none; background: transparent;")
        lbl_stats.setAlignment(Qt.AlignCenter)
        
        lay.addWidget(lbl_level)
        lay.addWidget(lbl_stats)
        
        f.lbl_level = lbl_level
        f.lbl_stats = lbl_stats
        return f

    def _build_dashboard_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {DARK_BG}; }}")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(25)
        
        # Title
        title = QLabel("🦅 Prophet Dashboard")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # Row 1: Verdict | Regime | Whipsaw
        row1 = QHBoxLayout()
        
        self.card_verdict = MetricCard("VERDICT", "—", ACCENT_GREEN)
        self.card_regime = MetricCard("REGIME", "—", ACCENT_CYAN)
        self.card_whipsaw = MetricCard("WHIPSAW", "—", ACCENT_GOLD)
        self.card_confidence = MetricCard("CONFIDENCE", "—", TEXT_PRIMARY)
        
        row1.addWidget(self.card_verdict)
        row1.addWidget(self.card_regime)
        row1.addWidget(self.card_whipsaw)
        row1.addWidget(self.card_confidence)
        layout.addLayout(row1)
        
        # Row 2: Ensemble | Intraday prediction 
        row2 = QHBoxLayout()
        
        self.card_tree = MetricCard("📊 DAILY ENSEMBLE", "—", TEXT_PRIMARY)
        self.card_intraday_dash = MetricCard("⚡ 15-MIN INTRADAY", "—", TEXT_PRIMARY)
        
        row2.addWidget(self.card_tree)
        row2.addWidget(self.card_intraday_dash)
        layout.addLayout(row2)
        
        # Row 3: Market Sentiment
        sentiment_title = QLabel("📊 Market Sentiment")
        sentiment_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 16px; font-weight: bold;")
        layout.addWidget(sentiment_title)
        
        row3 = QHBoxLayout()
        self.card_pcr = MetricCard("PUT-CALL RATIO", "—", ACCENT_CYAN)
        self.card_fii = MetricCard("FII NET (Cr)", "—", ACCENT_GREEN)
        self.card_dii = MetricCard("DII NET (Cr)", "—", ACCENT_GOLD)
        
        row3.addWidget(self.card_pcr)
        row3.addWidget(self.card_fii)
        row3.addWidget(self.card_dii)
        layout.addLayout(row3)
        
        # Row 4: Support & Resistance
        sr_title = QLabel("📍 Support & Resistance")
        sr_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 16px; font-weight: bold;")
        layout.addWidget(sr_title)
        
        row4 = QHBoxLayout()
        
        self.res_tiles = []
        res_group = QGroupBox("Resistance (Above - Wait for Rejections)")
        res_layout = QHBoxLayout(res_group)
        for _ in range(3):
            tile = self._create_sr_tile()
            self.res_tiles.append(tile)
            res_layout.addWidget(tile)
        
        self.support_tiles = []
        sup_group = QGroupBox("Support (Below - Wait for Bounces)")
        sup_layout = QHBoxLayout(sup_group)
        for _ in range(3):
            tile = self._create_sr_tile()
            self.support_tiles.append(tile)
            sup_layout.addWidget(tile)
        
        row4.addWidget(res_group)
        row4.addWidget(sup_group)
        layout.addLayout(row4)
        
        # Placeholder message
        self.dashboard_placeholder = QLabel("Press 🔴 Fetch Spot + Predict (F5) to see the full dashboard")
        self.dashboard_placeholder.setStyleSheet(f"color: {TEXT_DIM}; font-size: 16px; margin-top: 20px;")
        self.dashboard_placeholder.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.dashboard_placeholder)
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        
        return tab
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: FORECAST & RANGES
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_forecast_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        
        title = QLabel("📈 Price Forecast & Ranges")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # 7-day and 30-day range cards
        row = QHBoxLayout()
        self.card_7d_low = MetricCard("7-Day Low (10%)", "—", ACCENT_RED)
        self.card_7d_mid = MetricCard("7-Day Median", "—", ACCENT_CYAN)
        self.card_7d_high = MetricCard("7-Day High (90%)", "—", ACCENT_GREEN)
        row.addWidget(self.card_7d_low)
        row.addWidget(self.card_7d_mid)
        row.addWidget(self.card_7d_high)
        layout.addLayout(row)
        
        row2 = QHBoxLayout()
        self.card_30d_low = MetricCard("30-Day Low (10%)", "—", ACCENT_RED)
        self.card_30d_mid = MetricCard("30-Day Median", "—", ACCENT_CYAN)
        self.card_30d_high = MetricCard("30-Day High (90%)", "—", ACCENT_GREEN)
        row2.addWidget(self.card_30d_low)
        row2.addWidget(self.card_30d_mid)
        row2.addWidget(self.card_30d_high)
        layout.addLayout(row2)
        
        # Probability cone chart
        self.forecast_chart = ChartCanvas(width=8, height=4)
        layout.addWidget(self.forecast_chart)
        
        layout.addStretch()
        return tab
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: STRATEGY LAB
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_strategy_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        
        title = QLabel("🧪 Strategy Lab")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # Iron Condor section
        condor_group = QGroupBox("🛡️ Iron Condor Analyzer")
        condor_layout = QGridLayout(condor_group)
        
        condor_layout.addWidget(QLabel("Strike Price:"), 0, 0)
        self.strike_input = QSpinBox()
        self.strike_input.setRange(10000, 50000)
        self.strike_input.setSingleStep(100)
        self.strike_input.setValue(25000)
        condor_layout.addWidget(self.strike_input, 0, 1)
        
        condor_layout.addWidget(QLabel("Days:"), 0, 2)
        self.days_input = QSpinBox()
        self.days_input.setRange(1, 30)
        self.days_input.setValue(5)
        condor_layout.addWidget(self.days_input, 0, 3)
        
        self.btn_analyze = QPushButton("Analyze Strike")
        self.btn_analyze.clicked.connect(self.on_analyze_strike)
        condor_layout.addWidget(self.btn_analyze, 0, 4)
        
        # Results
        condor_results = QHBoxLayout()
        self.card_touch_prob = MetricCard("Touch Prob", "—", ACCENT_RED)
        self.card_recovery = MetricCard("Recovery", "—", ACCENT_GREEN)
        self.card_firefight = MetricCard("Firefight Level", "—", ACCENT_GOLD)
        condor_results.addWidget(self.card_touch_prob)
        condor_results.addWidget(self.card_recovery)
        condor_results.addWidget(self.card_firefight)
        condor_layout.addLayout(condor_results, 1, 0, 1, 5)
        
        self.condor_verdict = QLabel("")
        self.condor_verdict.setAlignment(Qt.AlignCenter)
        self.condor_verdict.setStyleSheet(f"font-size: 14px; font-weight: bold;")
        condor_layout.addWidget(self.condor_verdict, 2, 0, 1, 5)
        
        layout.addWidget(condor_group)
        
        # Bounce Calculator
        bounce_group = QGroupBox("🔄 Bounce-Back Calculator")
        bounce_layout = QGridLayout(bounce_group)
        
        bounce_layout.addWidget(QLabel("Target Price:"), 0, 0)
        self.target_input = QSpinBox()
        self.target_input.setRange(10000, 50000)
        self.target_input.setSingleStep(100)
        self.target_input.setValue(23000)
        bounce_layout.addWidget(self.target_input, 0, 1)
        
        self.btn_bounce = QPushButton("Check Bounce")
        self.btn_bounce.clicked.connect(self.on_check_bounce)
        bounce_layout.addWidget(self.btn_bounce, 0, 2)
        
        self.bounce_table = QTableWidget(0, 3)
        self.bounce_table.setHorizontalHeaderLabels(["Days", "Recovery %", "Avg Recovery Days"])
        self.bounce_table.horizontalHeader().setStretchLastSection(True)
        self.bounce_table.verticalHeader().setVisible(False)
        self.bounce_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.bounce_table.setMaximumHeight(150)
        bounce_layout.addWidget(self.bounce_table, 1, 0, 1, 3)
        
        layout.addWidget(bounce_group)
        layout.addStretch()
        return tab
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: DATA INSPECTOR
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_inspector_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        
        title = QLabel("📋 Data Inspector")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # CSV Status Table
        csv_group = QGroupBox("📊 CSV Data Files")
        csv_layout = QVBoxLayout(csv_group)
        
        self.csv_table = QTableWidget(5, 5)
        self.csv_table.setHorizontalHeaderLabels(["Name", "Rows", "Latest Date", "Size", "Status"])
        self.csv_table.horizontalHeader().setStretchLastSection(True)
        self.csv_table.verticalHeader().setVisible(False)
        self.csv_table.setEditTriggers(QTableWidget.NoEditTriggers)
        csv_layout.addWidget(self.csv_table)
        layout.addWidget(csv_group)
        
        # Model Status Table
        model_group = QGroupBox("🧠 ML Model Files (.pkl)")
        model_layout = QVBoxLayout(model_group)
        
        self.model_table = QTableWidget(5, 4)
        self.model_table.setHorizontalHeaderLabels(["Model", "Size", "Last Modified", "Status"])
        self.model_table.horizontalHeader().setStretchLastSection(True)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setEditTriggers(QTableWidget.NoEditTriggers)
        model_layout.addWidget(self.model_table)
        layout.addWidget(model_group)
        
        # Refresh button
        btn_refresh = QPushButton("🔄 Refresh Inspector")
        btn_refresh.clicked.connect(self.refresh_data_inspector)
        layout.addWidget(btn_refresh)
        
        layout.addStretch()
        return tab
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB: POSITION MANAGER — The Backbone
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_position_manager_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {DARK_BG}; }}")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 16, 12)
        layout.setSpacing(12)
        
        title = QLabel("🛡️ Position Manager — Your Trading Backbone")
        title.setStyleSheet(f"color: {ACCENT_GOLD}; font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # ── SECTION 1: Log Position ──
        pos_group = QGroupBox("📌 Open Position")
        pos_layout = QGridLayout(pos_group)
        
        pos_layout.addWidget(QLabel("Direction:"), 0, 0)
        self.pos_direction = QComboBox()
        self.pos_direction.addItems([
            "UP (Bull Call Spread - Debit)", 
            "UP (Bull Put Spread - Credit)", 
            "DOWN (Bear Put Spread - Debit)", 
            "DOWN (Bear Call Spread - Credit)", 
            "SIDEWAYS (Iron Condor)"
        ])
        pos_layout.addWidget(self.pos_direction, 0, 1)
        
        pos_layout.addWidget(QLabel("Entry Price (NIFTY):"), 0, 2)
        self.pos_entry_price = QSpinBox()
        self.pos_entry_price.setRange(10000, 50000)
        self.pos_entry_price.setSingleStep(50)
        self.pos_entry_price.setValue(24450)
        pos_layout.addWidget(self.pos_entry_price, 0, 3)
        
        pos_layout.addWidget(QLabel("Entry Date:"), 1, 0)
        self.pos_entry_date = QDateEdit()
        self.pos_entry_date.setCalendarPopup(True)
        self.pos_entry_date.setDate(QDate.currentDate())
        self.pos_entry_date.setDisplayFormat("yyyy-MM-dd")
        pos_layout.addWidget(self.pos_entry_date, 1, 1)
        
        pos_layout.addWidget(QLabel("Expiry:"), 1, 2)
        self.pos_expiry_date = QDateEdit()
        self.pos_expiry_date.setCalendarPopup(True)
        self.pos_expiry_date.setDate(QDate.currentDate().addDays(14))
        self.pos_expiry_date.setDisplayFormat("yyyy-MM-dd")
        pos_layout.addWidget(self.pos_expiry_date, 1, 3)
        
        self.btn_log_position = QPushButton("📌  Log Position")
        self.btn_log_position.clicked.connect(self.on_log_position)
        pos_layout.addWidget(self.btn_log_position, 2, 0, 1, 2)
        
        self.btn_close_position = QPushButton("❌  Close Position")
        self.btn_close_position.clicked.connect(self.on_close_position)
        pos_layout.addWidget(self.btn_close_position, 2, 2, 1, 2)
        
        layout.addWidget(pos_group)
        
        # ── SECTION 2: Hold/Exit Signal ──
        signal_group = QGroupBox("🦅 David's Verdict on Your Position")
        signal_layout = QVBoxLayout(signal_group)
        
        self.signal_verdict = QLabel("Press 📌 Log Position, then 🔴 Fetch Spot to get a HOLD / EXIT signal")
        self.signal_verdict.setWordWrap(True)
        self.signal_verdict.setStyleSheet(f"color: {TEXT_DIM}; font-size: 14px; padding: 12px;")
        signal_layout.addWidget(self.signal_verdict)
        
        signal_cards = QHBoxLayout()
        self.card_hold_exit = MetricCard("SIGNAL", "—", ACCENT_CYAN)
        self.card_days_held = MetricCard("DAYS HELD", "—", TEXT_PRIMARY)
        self.card_current_pnl = MetricCard("P&L", "—", TEXT_PRIMARY)
        signal_cards.addWidget(self.card_hold_exit)
        signal_cards.addWidget(self.card_days_held)
        signal_cards.addWidget(self.card_current_pnl)
        signal_layout.addLayout(signal_cards)
        
        layout.addWidget(signal_group)
        
        # ── SECTION 3: Recovery Probability ──
        recovery_group = QGroupBox("📊 Recovery Probability (Historical)")
        recovery_layout = QVBoxLayout(recovery_group)
        
        self.recovery_message = QLabel("Will update when a position is logged and Fetch Spot is clicked.")
        self.recovery_message.setWordWrap(True)
        self.recovery_message.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px; padding: 8px;")
        recovery_layout.addWidget(self.recovery_message)
        
        recovery_cards = QHBoxLayout()
        self.card_recovery_pct = MetricCard("RECOVERY %", "—", ACCENT_GREEN)
        self.card_recovery_days = MetricCard("AVG DAYS", "—", ACCENT_CYAN)
        self.card_scenarios = MetricCard("SIMILAR DIPS", "—", TEXT_PRIMARY)
        recovery_cards.addWidget(self.card_recovery_pct)
        recovery_cards.addWidget(self.card_recovery_days)
        recovery_cards.addWidget(self.card_scenarios)
        recovery_layout.addLayout(recovery_cards)
        
        layout.addWidget(recovery_group)
        
        # ── SECTION 4: Expiry Advisor ──
        expiry_group = QGroupBox("📅 Optimal Expiry Advisor")
        expiry_layout = QVBoxLayout(expiry_group)
        
        self.expiry_recommendation = QLabel("Expiry recommendation will appear after Fetch Spot.")
        self.expiry_recommendation.setWordWrap(True)
        self.expiry_recommendation.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 14px; font-weight: bold; padding: 8px;")
        expiry_layout.addWidget(self.expiry_recommendation)
        
        self.expiry_reasoning = QLabel("")
        self.expiry_reasoning.setWordWrap(True)
        self.expiry_reasoning.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px; padding: 4px 8px;")
        expiry_layout.addWidget(self.expiry_reasoning)
        
        layout.addWidget(expiry_group)
        
        # ── SECTION 5: Drawdown Alert ──
        dd_group = QGroupBox("⚠️ Drawdown Alert System")
        dd_layout = QVBoxLayout(dd_group)
        
        dd_cards = QHBoxLayout()
        self.card_dd_severity = MetricCard("SEVERITY", "🟢 SAFE", ACCENT_GREEN)
        self.card_dd_drawdown = MetricCard("DRAWDOWN", "0.0%", TEXT_PRIMARY)
        self.card_dd_streak = MetricCard("LOSS STREAK", "0", TEXT_PRIMARY)
        dd_cards.addWidget(self.card_dd_severity)
        dd_cards.addWidget(self.card_dd_drawdown)
        dd_cards.addWidget(self.card_dd_streak)
        dd_layout.addLayout(dd_cards)
        
        self.dd_message = QLabel("")
        self.dd_message.setWordWrap(True)
        self.dd_message.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px; padding: 4px 8px;")
        dd_layout.addWidget(self.dd_message)
        
        layout.addWidget(dd_group)
        
        # ── SECTION 6: 15-Min Intraday Regime ──
        regime15_group = QGroupBox("📊 15-Min Intraday Regime")
        regime15_layout = QVBoxLayout(regime15_group)
        
        regime15_cards = QHBoxLayout()
        self.card_intraday_regime = MetricCard("REGIME", "—", ACCENT_CYAN)
        self.card_entry_quality = MetricCard("ENTRY QUALITY", "—", TEXT_PRIMARY)
        self.card_rsi_15m = MetricCard("RSI (15m)", "—", TEXT_PRIMARY)
        self.card_vix_15m = MetricCard("VIX TREND", "—", TEXT_PRIMARY)
        regime15_cards.addWidget(self.card_intraday_regime)
        regime15_cards.addWidget(self.card_entry_quality)
        regime15_cards.addWidget(self.card_rsi_15m)
        regime15_cards.addWidget(self.card_vix_15m)
        regime15_layout.addLayout(regime15_cards)
        
        layout.addWidget(regime15_group)
        
        # ── Refresh Button ──
        self.btn_refresh_pm = QPushButton("🔄  Refresh Position Manager")
        self.btn_refresh_pm.clicked.connect(self.on_refresh_position_manager)
        layout.addWidget(self.btn_refresh_pm)
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5: INTRADAY ANALYSIS (15-Min ML)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_intraday_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(25)
        
        # ── Header ──
        header = QLabel("⚡ 15-Minute ML Intraday Classifier")
        header.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 24px; font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(header)
        
        desc = QLabel("Trained on 67,000+ rows of 15-min candle data using XGBoost. Predicts next 1-4 hour direction based on RSI, VWAP setup, MACD, and VIX structure. Use this for <b>entry timing</b>.")
        desc.setStyleSheet(f"color: {TEXT_SEC}; font-size: 14px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # ── 1. Verdict Box ──
        verdict_group = QGroupBox("🤖 ML Intraday Verdict")
        verdict_group.setStyleSheet(f"""
            QGroupBox {{ border: 1px solid {DARK_BORDER}; border-radius: 8px; margin-top: 15px; font-weight: bold; color: {TEXT_PRIMARY}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 15px; padding: 0 5px; }}
        """)
        v_layout = QVBoxLayout(verdict_group)
        v_layout.setContentsMargins(20, 30, 20, 20)
        
        self.lbl_intraday_dir = QLabel("—")
        self.lbl_intraday_dir.setAlignment(Qt.AlignCenter)
        self.lbl_intraday_dir.setStyleSheet(f"color: {TEXT_DIM}; font-size: 36px; font-weight: bold;")
        v_layout.addWidget(self.lbl_intraday_dir)
        
        self.lbl_intraday_timing = QLabel("Click Fetch Spot + Predict to analyze 15m data.")
        self.lbl_intraday_timing.setAlignment(Qt.AlignCenter)
        self.lbl_intraday_timing.setStyleSheet(f"color: {TEXT_SEC}; font-size: 16px; margin-top: 10px;")
        v_layout.addWidget(self.lbl_intraday_timing)
        
        layout.addWidget(verdict_group)
        
        # ── 2. Probabilities ──
        prob_group = QGroupBox("📊 15-Min Class Probabilities")
        prob_group.setStyleSheet(f"""
            QGroupBox {{ border: 1px solid {DARK_BORDER}; border-radius: 8px; margin-top: 15px; font-weight: bold; color: {TEXT_PRIMARY}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 15px; padding: 0 5px; }}
        """)
        p_layout = QHBoxLayout(prob_group)
        p_layout.setContentsMargins(20, 30, 20, 20)
        p_layout.setSpacing(15)
        
        self.card_intra_up = MetricCard("UP PROB", "—", ACCENT_GREEN)
        self.card_intra_chop = MetricCard("SIDEWAYS", "—", ACCENT_GOLD)
        self.card_intra_down = MetricCard("DOWN PROB", "—", ACCENT_RED)
        
        p_layout.addWidget(self.card_intra_up, 1)
        p_layout.addWidget(self.card_intra_chop, 1)
        p_layout.addWidget(self.card_intra_down, 1)
        layout.addWidget(prob_group)
        
        # ── 3. Engineered Features ──
        feat_group = QGroupBox("🧬 Current 15m Technical State")
        feat_group.setStyleSheet(f"""
            QGroupBox {{ border: 1px solid {DARK_BORDER}; border-radius: 8px; margin-top: 15px; font-weight: bold; color: {TEXT_PRIMARY}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 15px; padding: 0 5px; }}
        """)
        f_layout = QGridLayout(feat_group)
        f_layout.setContentsMargins(20, 30, 20, 20)
        f_layout.setVerticalSpacing(15)
        f_layout.setHorizontalSpacing(15)
        
        self.card_feat_rsi = MetricCard("RSI-14 (15m)", "—")
        self.card_feat_vwap = MetricCard("VWAP Dev", "—")
        self.card_feat_volz = MetricCard("Vol Z-Score", "—")
        self.card_feat_atr = MetricCard("ATR %", "—")
        
        f_layout.addWidget(self.card_feat_rsi, 0, 0)
        f_layout.addWidget(self.card_feat_vwap, 0, 1)
        f_layout.addWidget(self.card_feat_volz, 1, 0)
        f_layout.addWidget(self.card_feat_atr, 1, 1)
        layout.addWidget(feat_group)
        
        # ── Refresh Button ──
        self.btn_refresh_intraday = QPushButton("🔄  Refresh Intraday Analysis")
        self.btn_refresh_intraday.clicked.connect(self._fetch_intraday_only)
        layout.addWidget(self.btn_refresh_intraday)
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab

    def _fetch_intraday_only(self):
        self.append_log("\n" + "═" * 50)
        self._start_worker(
            backend.predict_intraday_now,
            self._on_intraday_done,
            "⚡ Running Intraday ML Model..."
        )

    def _on_intraday_done(self, result):
        if result.get("error"):
            self.statusBar().showMessage(f"❌ Intraday Error: {result['error']}")
            return
            
        direction = result.get("direction", "UNKNOWN")
        conf = result.get("confidence", 0) * 100
        
        colors = {"UP": ACCENT_GREEN, "DOWN": ACCENT_RED, "SIDEWAYS": ACCENT_GOLD}
        color = colors.get(direction, TEXT_DIM)
        
        self.lbl_intraday_dir.setText(f"{direction} ({conf:.1f}%)")
        self.lbl_intraday_dir.setStyleSheet(f"color: {color}; font-size: 36px; font-weight: bold;")
        
        if hasattr(self, 'card_intraday_dash'):
            self.card_intraday_dash.set_value(f"{direction} ({conf:.0f}%)", color)
            
        
        timing = result.get("entry_timing", "")
        self.lbl_intraday_timing.setText(timing)
        if "WAIT" in timing or "CAUTION" in timing:
            self.lbl_intraday_timing.setStyleSheet(f"color: {ACCENT_RED}; font-size: 16px; margin-top: 10px; font-weight: bold;")
        elif "GOOD" in timing:
            self.lbl_intraday_timing.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 16px; margin-top: 10px; font-weight: bold;")
        else:
            self.lbl_intraday_timing.setStyleSheet(f"color: {TEXT_SEC}; font-size: 16px; margin-top: 10px;")
            
        probs = result.get("probabilities", {})
        self.card_intra_up.set_value(f"{probs.get('UP', 0):.1f}%")
        self.card_intra_chop.set_value(f"{probs.get('SIDEWAYS', 0):.1f}%")
        self.card_intra_down.set_value(f"{probs.get('DOWN', 0):.1f}%")
        
        feats = result.get("raw_features", {})
        
        rsi = feats.get("rsi_14", 50)
        rsi_col = ACCENT_RED if rsi > 70 or rsi < 30 else ACCENT_GREEN
        self.card_feat_rsi.set_value(f"{rsi:.1f}", rsi_col)
        
        vwap = feats.get("vwap_dev", 0)
        self.card_feat_vwap.set_value(f"{vwap:+.2f}%", ACCENT_GREEN if vwap > 0 else ACCENT_RED)
        
        volz = feats.get("volume_zscore", 0)
        self.card_feat_volz.set_value(f"{volz:+.1f}", ACCENT_CYAN if volz > 2 else TEXT_PRIMARY)
        
        self.card_feat_atr.set_value(f"{feats.get('atr_pct', 0):.2f}%")
        
        self.statusBar().showMessage("✅ Intraday ML analysis complete!")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: 🚦 COMMAND CENTER — Traffic Light + Morning Briefing
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_command_center_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {DARK_BG}; }}")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        title = QLabel("🚦 Command Center")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 22px; font-weight: bold;")
        layout.addWidget(title)
        
        subtitle = QLabel("One look. One decision. Zero anxiety.")
        subtitle.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px; margin-bottom: 10px;")
        layout.addWidget(subtitle)
        
        # ── Giant Traffic Light ──
        self.traffic_light_frame = QFrame()
        self.traffic_light_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
                border-radius: 16px;
                padding: 30px;
            }}
        """)
        tf_layout = QVBoxLayout(self.traffic_light_frame)
        
        self.traffic_light_label = QLabel("⚪")
        self.traffic_light_label.setStyleSheet("font-size: 80px;")
        self.traffic_light_label.setAlignment(Qt.AlignCenter)
        tf_layout.addWidget(self.traffic_light_label)
        
        self.traffic_light_text = QLabel("Press Fetch + Predict to get your signal")
        self.traffic_light_text.setStyleSheet(f"color: {TEXT_DIM}; font-size: 18px; font-weight: bold;")
        self.traffic_light_text.setAlignment(Qt.AlignCenter)
        tf_layout.addWidget(self.traffic_light_text)
        
        self.traffic_light_action = QLabel("")
        self.traffic_light_action.setStyleSheet(f"color: {TEXT_SEC}; font-size: 14px;")
        self.traffic_light_action.setAlignment(Qt.AlignCenter)
        self.traffic_light_action.setWordWrap(True)
        tf_layout.addWidget(self.traffic_light_action)
        
        layout.addWidget(self.traffic_light_frame)
        
        # ── Morning Briefing Card ──
        briefing_group = QGroupBox("📋 Morning Briefing")
        bg_layout = QGridLayout(briefing_group)
        
        self.cc_regime = MetricCard("REGIME", "—", ACCENT_CYAN)
        self.cc_strategy = MetricCard("BEST STRATEGY", "—", ACCENT_GREEN)
        self.cc_risk = MetricCard("RISK LEVEL", "—", ACCENT_GOLD)
        self.cc_vix = MetricCard("VIX", "—", TEXT_PRIMARY)
        
        bg_layout.addWidget(self.cc_regime, 0, 0)
        bg_layout.addWidget(self.cc_strategy, 0, 1)
        bg_layout.addWidget(self.cc_risk, 0, 2)
        bg_layout.addWidget(self.cc_vix, 0, 3)
        layout.addWidget(briefing_group)
        
        # ── Strategy Reasoning ──
        self.cc_reasoning = QLabel("")
        self.cc_reasoning.setWordWrap(True)
        self.cc_reasoning.setStyleSheet(f"""
            color: {TEXT_SEC}; font-size: 13px; padding: 12px;
            background-color: {DARK_CARD}; border-radius: 8px;
        """)
        layout.addWidget(self.cc_reasoning)
        
        # ── Signal Reasons List ──
        self.cc_reasons_label = QLabel("")
        self.cc_reasons_label.setWordWrap(True)
        self.cc_reasons_label.setStyleSheet(f"""
            color: {TEXT_SEC}; font-size: 12px; padding: 10px;
            background-color: {DARK_CARD}; border-radius: 8px;
        """)
        layout.addWidget(self.cc_reasons_label)
        
        # ── Position Sizing Calculator ──
        sizing_group = QGroupBox("💰 Position Sizing Calculator")
        sz_layout = QHBoxLayout(sizing_group)
        
        sz_layout.addWidget(QLabel("Your Capital (₹):"))
        self.capital_input = QSpinBox()
        self.capital_input.setRange(10000, 100000000)
        self.capital_input.setSingleStep(50000)
        self.capital_input.setValue(200000)
        self.capital_input.setPrefix("₹ ")
        sz_layout.addWidget(self.capital_input)
        
        self.btn_calc_sizing = QPushButton("Calculate")
        self.btn_calc_sizing.setStyleSheet(f"background-color: {ACCENT_CYAN}; color: #000; font-weight: bold; padding: 8px 16px; border-radius: 6px;")
        self.btn_calc_sizing.clicked.connect(self._on_calc_position_sizing)
        sz_layout.addWidget(self.btn_calc_sizing)
        
        layout.addWidget(sizing_group)
        
        self.cc_sizing_result = QLabel("")
        self.cc_sizing_result.setWordWrap(True)
        self.cc_sizing_result.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; padding: 10px; background-color: {DARK_CARD}; border-radius: 8px;")
        layout.addWidget(self.cc_sizing_result)
        
        layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab
    
    def _on_calc_position_sizing(self):
        from analyzers.trade_journal import compute_position_sizing, save_settings
        capital = self.capital_input.value()
        save_settings({"capital": capital, "lot_size": 65})
        
        # Get current traffic light color
        color = getattr(self, '_last_traffic_color', 'YELLOW')
        result = compute_position_sizing(capital, color)
        
        self.cc_sizing_result.setText(
            f"<b>{result['recommendation']}</b><br>"
            f"Max Lots: <b>{result['max_lots']}</b> | "
            f"Max Risk: <b>₹{result['max_risk']:,.0f}</b> ({result['risk_pct']:.1f}% of capital) | "
            f"Risk/Lot: ₹{result['risk_per_lot']:,.0f}"
        )
    
    def _update_command_center(self, pred):
        """Populate Command Center with prediction data."""
        briefing = backend.generate_morning_briefing(pred)
        traffic = briefing["traffic_light"]
        strategy = briefing["strategy"]
        
        # Traffic Light
        color = traffic.get("color", "RED")
        self._last_traffic_color = color
        
        emoji_map = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
        action_map = {
            "GREEN": "✅ ENTER — Signal is strong. Trade the recommended strategy.",
            "YELLOW": "⚠️ WAIT — Mixed signals. Reduce size or wait for clarity.",
            "RED": "🚫 DO NOT TRADE — Sit this one out. Capital preservation.",
        }
        color_css = {"GREEN": "#00FF7F", "YELLOW": "#FFD700", "RED": "#FF4B4B"}
        
        self.traffic_light_label.setText(emoji_map.get(color, "⚪"))
        self.traffic_light_label.setStyleSheet(f"font-size: 80px;")
        self.traffic_light_text.setText(f"{color} SIGNAL")
        self.traffic_light_text.setStyleSheet(f"color: {color_css.get(color, TEXT_DIM)}; font-size: 22px; font-weight: bold;")
        self.traffic_light_action.setText(action_map.get(color, ""))
        
        # Briefing cards
        self.cc_regime.set_value(briefing["regime"])
        self.cc_strategy.set_value(strategy["strategy"], ACCENT_GREEN)
        
        risk_color = {"LOW": ACCENT_GREEN, "MODERATE": ACCENT_GOLD, "HIGH": ACCENT_RED}.get(strategy["risk_level"], TEXT_DIM)
        self.cc_risk.set_value(strategy["risk_level"], risk_color)
        self.cc_vix.set_value(f"{briefing['vix']:.1f}" if briefing['vix'] else "—")
        
        # Reasoning
        self.cc_reasoning.setText(f"💡 <b>Why {strategy['strategy']}?</b><br>{strategy['reasoning']}")
        
        # Signal reasons
        reasons = traffic.get("reasons", [])
        if reasons:
            reasons_html = "<br>".join([f"• {r}" for r in reasons])
            self.cc_reasons_label.setText(f"<b>Signal Analysis:</b><br>{reasons_html}")
        
        # Auto-calculate sizing
        self._on_calc_position_sizing()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: ⚔️ TRADING WAR ROOM — Options Strategy Builder + AI Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_war_room_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {DARK_BG}; }}")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(12)
        
        title = QLabel("⚔️ Trading War Room")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 22px; font-weight: bold;")
        layout.addWidget(title)
        
        subtitle = QLabel("Build your strategy. Let David judge it.")
        subtitle.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px;")
        layout.addWidget(subtitle)
        
        # ── Leg Builder ──
        legs_group = QGroupBox("📐 Strategy Legs")
        self.legs_layout = QVBoxLayout(legs_group)
        
        self.war_grid = QGridLayout()
        self.legs_layout.addLayout(self.war_grid)
        
        # Add Header Row
        headers = ["Leg", "Strike", "Type", "Action", "Entry ₹", "Exit ₹", "Lots", "Status"]
        for col, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 13px; padding-bottom: 5px;")
            self.war_grid.addWidget(lbl, 0, col)
        
        self.war_legs = []  # list of leg widget dicts
        self._add_war_leg()  # start with 1 leg
        self._add_war_leg()  # add 2nd leg for spread
        
        btn_row = QHBoxLayout()
        btn_add_leg = QPushButton("➕ Add Leg")
        btn_add_leg.setStyleSheet(f"background-color: {ACCENT_CYAN}; color: #000; font-weight: bold; padding: 6px 14px; border-radius: 6px;")
        btn_add_leg.clicked.connect(self._add_war_leg)
        btn_row.addWidget(btn_add_leg)
        
        btn_remove_leg = QPushButton("➖ Remove Last")
        btn_remove_leg.setStyleSheet(f"background-color: {ACCENT_RED}; color: #fff; font-weight: bold; padding: 6px 14px; border-radius: 6px;")
        btn_remove_leg.clicked.connect(self._remove_war_leg)
        btn_row.addWidget(btn_remove_leg)
        btn_row.addStretch()
        self.legs_layout.addLayout(btn_row)
        
        # ── Closed Legs P&L ──
        closed_row = QHBoxLayout()
        closed_row.addWidget(QLabel("Realized P&L from Closed/Hedged Legs (offset):"))
        self.war_realized_pnl = QDoubleSpinBox()
        self.war_realized_pnl.setRange(-500000, 500000)
        self.war_realized_pnl.setSingleStep(100)
        self.war_realized_pnl.setDecimals(2)
        self.war_realized_pnl.setValue(0.00)
        self.war_realized_pnl.setPrefix("₹ ")
        self.war_realized_pnl.setStyleSheet(f"background-color: {DARK_CARD}; color: {TEXT_PRIMARY}; border: 1px solid {DARK_BORDER}; border-radius: 4px; padding: 4px;")
        closed_row.addWidget(self.war_realized_pnl)
        closed_row.addStretch()
        self.legs_layout.addLayout(closed_row)
        
        # ── Trade Notes / Ledger ──
        notes_lbl = QLabel("Ledger / Closed Legs Notes:")
        notes_lbl.setStyleSheet(f"color: {TEXT_SEC}; font-size: 13px; margin-top: 10px;")
        self.legs_layout.addWidget(notes_lbl)
        
        from PyQt5.QtWidgets import QTextEdit
        self.war_trade_notes = QTextEdit()
        self.war_trade_notes.setMaximumHeight(60)
        self.war_trade_notes.setPlaceholderText("E.g., Closed 23500 PE for +₹1500, rolled to 23000 PE...")
        self.war_trade_notes.setStyleSheet(f"background-color: {DARK_CARD}; color: {TEXT_PRIMARY}; border: 1px solid {DARK_BORDER}; border-radius: 4px; padding: 4px;")
        self.legs_layout.addWidget(self.war_trade_notes)
        
        layout.addWidget(legs_group)
        
        # ── Date inputs ──
        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("Entry Date:"))
        self.war_entry_date = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        self.war_entry_date.setStyleSheet(f"background-color: {DARK_CARD}; color: {TEXT_PRIMARY}; border: 1px solid {DARK_BORDER}; border-radius: 4px; padding: 6px;")
        date_row.addWidget(self.war_entry_date)
        
        date_row.addWidget(QLabel("Expiry Date:"))
        # Default to next Thursday
        from datetime import timedelta
        today = datetime.now()
        days_until_thu = (3 - today.weekday()) % 7
        if days_until_thu == 0:
            days_until_thu = 7
        next_thu = today + timedelta(days=days_until_thu)
        self.war_expiry_date = QLineEdit(next_thu.strftime("%Y-%m-%d"))
        self.war_expiry_date.setStyleSheet(f"background-color: {DARK_CARD}; color: {TEXT_PRIMARY}; border: 1px solid {DARK_BORDER}; border-radius: 4px; padding: 6px;")
        date_row.addWidget(self.war_expiry_date)
        
        # ── Portfolio Manager ──
        pf_row = QHBoxLayout()
        pf_row.addWidget(QLabel("Portfolio Name:"))
        self.war_pf_name = QLineEdit()
        self.war_pf_name.setPlaceholderText("E.g., Iron Condor 25Apr")
        self.war_pf_name.setStyleSheet(f"background-color: {DARK_CARD}; color: {TEXT_PRIMARY}; border: 1px solid {DARK_BORDER}; border-radius: 4px; padding: 4px;")
        pf_row.addWidget(self.war_pf_name)
        
        self.btn_save_pf = QPushButton("💾 Save")
        self.btn_save_pf.clicked.connect(self._save_war_portfolio)
        pf_row.addWidget(self.btn_save_pf)
        
        pf_row.addWidget(QLabel(" | Saved Portfolios:"))
        self.war_pf_combo = QComboBox()
        pf_row.addWidget(self.war_pf_combo)
        
        self.btn_load_pf = QPushButton("📂 Load")
        self.btn_load_pf.clicked.connect(self._load_war_portfolio)
        pf_row.addWidget(self.btn_load_pf)
        
        self.btn_delete_pf = QPushButton("🗑️ Delete")
        self.btn_delete_pf.setStyleSheet(f"background-color: {ACCENT_RED}; color: white; padding: 4px 10px; border-radius: 4px;")
        self.btn_delete_pf.clicked.connect(self._delete_war_portfolio)
        pf_row.addWidget(self.btn_delete_pf)
        
        self.btn_export_csv = QPushButton("📊 Export CSV")
        self.btn_export_csv.clicked.connect(self._export_war_portfolio_csv)
        self.btn_export_csv.setStyleSheet(f"background-color: #f7a02c; color: #000; font-weight: bold;")
        pf_row.addWidget(self.btn_export_csv)
        
        layout.addLayout(pf_row)
        
        self.btn_analyze_war = QPushButton("🤖 Analyze Trade")
        self.btn_analyze_war.setStyleSheet(f"background-color: {ACCENT_GREEN}; color: #000; font-weight: bold; padding: 10px 20px; border-radius: 8px; font-size: 14px;")
        self.btn_analyze_war.clicked.connect(self._on_analyze_war_room)
        date_row.addWidget(self.btn_analyze_war)
        
        layout.addLayout(date_row)
        
        # ── Payoff Chart ──
        self.war_chart = ChartCanvas(width=8, height=3)
        layout.addWidget(self.war_chart)
        
        # ── AI Verdict Panel ──
        verdict_group = QGroupBox("🤖 AI Verdict")
        vd_layout = QVBoxLayout(verdict_group)
        
        self.war_verdict_cards = QHBoxLayout()
        self.war_win_prob = MetricCard("WIN PROBABILITY", "—", ACCENT_GREEN)
        self.war_max_profit = MetricCard("MAX PROFIT", "—", ACCENT_GREEN)
        self.war_max_loss = MetricCard("MAX LOSS", "—", ACCENT_RED)
        self.war_breakeven = MetricCard("BREAKEVEN", "—", ACCENT_CYAN)
        self.war_verdict_cards.addWidget(self.war_win_prob)
        self.war_verdict_cards.addWidget(self.war_max_profit)
        self.war_verdict_cards.addWidget(self.war_max_loss)
        self.war_verdict_cards.addWidget(self.war_breakeven)
        vd_layout.addLayout(self.war_verdict_cards)
        
        self.war_verdict_text = QLabel("Add your trade legs and click Analyze Trade")
        self.war_verdict_text.setWordWrap(True)
        self.war_verdict_text.setStyleSheet(f"color: {TEXT_SEC}; font-size: 14px; padding: 10px; background-color: {DARK_CARD}; border-radius: 8px;")
        vd_layout.addWidget(self.war_verdict_text)
        
        self.war_firefight_text = QLabel("")
        self.war_firefight_text.setWordWrap(True)
        self.war_firefight_text.setStyleSheet(f"color: {ACCENT_GOLD}; font-size: 13px; padding: 8px;")
        vd_layout.addWidget(self.war_firefight_text)
        
        self.war_reasons_text = QLabel("")
        self.war_reasons_text.setWordWrap(True)
        self.war_reasons_text.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px; padding: 8px; background-color: {DARK_CARD}; border-radius: 8px;")
        vd_layout.addWidget(self.war_reasons_text)
        
        layout.addWidget(verdict_group)
        
        # ── Risk Dashboard (for Option Sellers) ──
        risk_group = QGroupBox("📊 Risk Dashboard — Option Seller Metrics")
        rk_layout = QVBoxLayout(risk_group)
        
        self.war_risk_cards = QHBoxLayout()
        self.war_pop = MetricCard("PROB. OF PROFIT", "—", ACCENT_GREEN)
        self.war_delta = MetricCard("POS. DELTA", "—", ACCENT_CYAN)
        self.war_theta = MetricCard("THETA/DAY", "—", ACCENT_GREEN)
        self.war_rr = MetricCard("RISK:REWARD", "—", ACCENT_GOLD)
        self.war_safe_zone = MetricCard("SAFE ZONE", "—", ACCENT_CYAN)
        self.war_margin = MetricCard("EST. MARGIN", "—", TEXT_SEC)
        self.war_risk_cards.addWidget(self.war_pop)
        self.war_risk_cards.addWidget(self.war_delta)
        self.war_risk_cards.addWidget(self.war_theta)
        self.war_risk_cards.addWidget(self.war_rr)
        self.war_risk_cards.addWidget(self.war_safe_zone)
        self.war_risk_cards.addWidget(self.war_margin)
        rk_layout.addLayout(self.war_risk_cards)
        
        self.war_risk_verdict = QLabel("")
        self.war_risk_verdict.setWordWrap(True)
        self.war_risk_verdict.setStyleSheet(f"color: {TEXT_SEC}; font-size: 13px; padding: 10px; background-color: {DARK_CARD}; border-radius: 8px;")
        rk_layout.addWidget(self.war_risk_verdict)
        
        layout.addWidget(risk_group)
        
        # ── Target Strike Probability Panel ──
        target_group = QGroupBox("🎯 Target Strike Analysis — AI Historical Probability")
        tg_layout = QVBoxLayout(target_group)
        
        target_input_row = QHBoxLayout()
        target_input_row.addWidget(QLabel("I executed my hedge at strike:"))
        self.war_target_strike = QSpinBox()
        self.war_target_strike.setRange(10000, 50000)
        self.war_target_strike.setSingleStep(50)
        self.war_target_strike.setValue(24650)
        self.war_target_strike.setStyleSheet(f"background-color: {DARK_CARD}; color: {ACCENT_GOLD}; border: 1px solid {ACCENT_GOLD}; border-radius: 4px; padding: 6px; font-size: 14px; font-weight: bold;")
        target_input_row.addWidget(self.war_target_strike)
        
        self.btn_target_analyze = QPushButton("🔮 Analyze Strike Probability")
        self.btn_target_analyze.setStyleSheet(f"background-color: {ACCENT_GOLD}; color: #000; font-weight: bold; padding: 8px 16px; border-radius: 6px;")
        self.btn_target_analyze.clicked.connect(self._on_target_strike_analysis)
        target_input_row.addWidget(self.btn_target_analyze)
        target_input_row.addStretch()
        tg_layout.addLayout(target_input_row)
        
        self.war_target_result = QLabel("Enter a strike and click Analyze to see AI probability of NIFTY reaching it.")
        self.war_target_result.setWordWrap(True)
        self.war_target_result.setStyleSheet(f"color: {TEXT_SEC}; font-size: 13px; padding: 12px; background-color: {DARK_CARD}; border-radius: 8px; border-left: 3px solid {ACCENT_GOLD};")
        tg_layout.addWidget(self.war_target_result)
        
        layout.addWidget(target_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        
        # Refresh portfolio list on startup
        try:
            self._refresh_war_portfolio_list()
        except Exception:
            pass
        
        return tab
    
    def _add_war_leg(self):
        """Add a new options leg row to the war room."""
        from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox
        
        leg_num = len(self.war_legs) + 1
        grid_row = leg_num
        
        lbl = QLabel(f"L{leg_num}")
        lbl.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold;")
        self.war_grid.addWidget(lbl, grid_row, 0)
        
        strike = QSpinBox()
        strike.setRange(10000, 50000)
        strike.setSingleStep(50)
        strike.setValue(24400 if leg_num == 1 else 24300)
        self.war_grid.addWidget(strike, grid_row, 1)
        
        opt_type = QComboBox()
        opt_type.addItems(["PE", "CE"])
        self.war_grid.addWidget(opt_type, grid_row, 2)
        
        action = QComboBox()
        action.addItems(["SELL", "BUY"])
        if leg_num == 2:
            action.setCurrentIndex(1)  # BUY for 2nd leg
        self.war_grid.addWidget(action, grid_row, 3)
        
        premium = QDoubleSpinBox()
        premium.setRange(0, 5000)
        premium.setSingleStep(0.5)
        premium.setDecimals(2)
        premium.setValue(85.50 if leg_num == 1 else 55.20)
        self.war_grid.addWidget(premium, grid_row, 4)
        
        exit_price = QDoubleSpinBox()
        exit_price.setRange(0, 5000)
        exit_price.setSingleStep(0.5)
        exit_price.setDecimals(2)
        exit_price.setValue(0.0)
        exit_price.setEnabled(False)
        self.war_grid.addWidget(exit_price, grid_row, 5)
        
        lots = QSpinBox()
        lots.setRange(1, 100)
        lots.setValue(1)
        self.war_grid.addWidget(lots, grid_row, 6)
        
        status = QComboBox()
        status.addItems(["Open", "Closed"])
        self.war_grid.addWidget(status, grid_row, 7)
        
        # Enable exit price only if closed
        status.currentTextChanged.connect(
            lambda text, e=exit_price: e.setEnabled(text == "Closed")
        )
        
        leg_widget = {
            "label": lbl,
            "strike": strike,
            "opt_type": opt_type,
            "action": action,
            "premium": premium,
            "lots": lots,
            "status": status,
            "exit_price": exit_price,
        }
        self.war_legs.append(leg_widget)
    
    def _remove_war_leg(self, force_clear=False):
        """Remove the last leg from the war room."""
        if not force_clear and len(self.war_legs) <= 1:
            return
        if len(self.war_legs) == 0:
            return
        leg = self.war_legs.pop()
        
        for key, widget in leg.items():
            if hasattr(widget, "deleteLater"):
                self.war_grid.removeWidget(widget)
                widget.deleteLater()
    
    def _get_war_legs_data(self):
        """Extract leg data from the UI widgets."""
        legs = []
        for leg in self.war_legs:
            legs.append({
                "strike": leg["strike"].value(),
                "option_type": leg["opt_type"].currentText(),
                "action": leg["action"].currentText(),
                "premium": leg["premium"].value(),
                "lots": leg["lots"].value(),
                "status": leg["status"].currentText(),
                "exit_price": leg["exit_price"].value(),
            })
        return legs
    
    def _refresh_war_portfolio_list(self):
        from analyzers.options_payoff import load_war_room_trades
        trades = load_war_room_trades()
        self.war_pf_combo.clear()
        if trades:
            self.war_pf_combo.addItems([t["name"] for t in trades])
            
    def _save_war_portfolio(self):
        name = self.war_pf_name.text().strip()
        if not name:
            self.statusBar().showMessage("❌ Enter a Portfolio Name to save.")
            return
            
        legs = self._get_war_legs_data()
        entry = self.war_entry_date.text()
        expiry = self.war_expiry_date.text()
        notes = self.war_trade_notes.toPlainText()
        
        from analyzers.options_payoff import save_war_room_trade
        save_war_room_trade(name, legs, entry, expiry, notes)
        self.statusBar().showMessage(f"✅ Portfolio '{name}' saved successfully!")
        self._refresh_war_portfolio_list()
    def _delete_war_portfolio(self):
        name = self.war_pf_combo.currentText()
        if not name:
            return
            
        from analyzers.options_payoff import delete_war_room_trade
        delete_war_room_trade(name)
        self.statusBar().showMessage(f"🗑️ Portfolio '{name}' deleted.")
        self._refresh_war_portfolio_list()
        
    def _load_war_portfolio(self):
        name = self.war_pf_combo.currentText()
        if not name:
            return
            
        from analyzers.options_payoff import load_war_room_trades
        trades = load_war_room_trades()
        for t in trades:
            if t["name"] == name:
                self.war_pf_name.setText(t["name"])
                self.war_entry_date.setText(t.get("entry_date", ""))
                self.war_expiry_date.setText(t.get("expiry_date", ""))
                self.war_trade_notes.setPlainText(t.get("notes", ""))
                
                # Clear and rebuild legs
                while len(self.war_legs) > 0:
                    self._remove_war_leg(force_clear=True)
                
                for leg_data in t["legs"]:
                    self._add_war_leg()
                    new_leg = self.war_legs[-1]
                    new_leg["strike"].setValue(leg_data.get("strike", 24400))
                    new_leg["opt_type"].setCurrentText(leg_data.get("option_type", "PE"))
                    new_leg["action"].setCurrentText(leg_data.get("action", "BUY"))
                    new_leg["premium"].setValue(leg_data.get("premium", 50.0))
                    new_leg["lots"].setValue(leg_data.get("lots", 1))
                    new_leg["status"].setCurrentText(leg_data.get("status", "Open"))
                    new_leg["exit_price"].setValue(leg_data.get("exit_price", 0.0))
                
                self.statusBar().showMessage(f"✅ Portfolio '{name}' loaded!")
                break
                
    def _export_war_portfolio_csv(self):
        import csv
        from PyQt5.QtWidgets import QFileDialog
        
        name = self.war_pf_name.text().strip() or "WarRoom_Trade"
        path, _ = QFileDialog.getSaveFileName(self, "Export Portfolio to CSV", f"{name}.csv", "CSV Files (*.csv)")
        
        if path:
            legs = self._get_war_legs_data()
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Portfolio Name", name])
                    writer.writerow(["Entry Date", self.war_entry_date.text()])
                    writer.writerow(["Expiry Date", self.war_expiry_date.text()])
                    writer.writerow(["Realized P&L Offset", f"₹ {self.war_realized_pnl.value():.2f}"])
                    writer.writerow([])
                    
                    headers = ["Leg", "Strike", "Type", "Action", "Entry Price", "Exit Price", "Lots", "Status"]
                    writer.writerow(headers)
                    
                    for i, leg in enumerate(legs, 1):
                        writer.writerow([
                            f"Leg {i}",
                            leg["strike"],
                            leg["option_type"],
                            leg["action"],
                            leg["premium"],
                            leg["exit_price"],
                            leg["lots"],
                            leg["status"]
                        ])
                        
                    writer.writerow([])
                    writer.writerow(["Trade Notes"])
                    writer.writerow([self.war_trade_notes.toPlainText()])
                    
                self.statusBar().showMessage(f"✅ Portfolio exported to {path}")
            except Exception as e:
                self.statusBar().showMessage(f"❌ Export failed: {str(e)}")
                
    def _on_target_strike_analysis(self):
        """AI analysis: will NIFTY reach my specific hedge strike?"""
        import numpy as np
        
        target = self.war_target_strike.value()
        spot = getattr(self, '_last_spot', 24500)
        pred = getattr(self, '_last_prediction', None)
        
        # Days to expiry
        try:
            expiry = datetime.strptime(self.war_expiry_date.text(), "%Y-%m-%d")
            today_dt = datetime.now()
            days_to_expiry = max(1, (expiry - today_dt).days)
        except ValueError:
            days_to_expiry = 5
        
        # Get VIX
        vix = 15.0
        if pred and pred.get("vix_value"):
            vix = pred["vix_value"]
        
        # Expected move calculation
        daily_vol = (vix / 100.0) / np.sqrt(365)
        period_vol = daily_vol * np.sqrt(days_to_expiry)
        expected_move = spot * period_vol
        
        dist = abs(spot - target)
        direction_to_target = "UP ↑" if target > spot else "DOWN ↓"
        sd_away = dist / max(1, expected_move)
        
        # Touch probability (higher than expiry probability)
        if sd_away <= 0.3:
            touch_prob = 75 + (0.3 - sd_away) * 50
        elif sd_away <= 0.5:
            touch_prob = 60 + (0.5 - sd_away) * 75
        elif sd_away <= 1.0:
            touch_prob = 30 + (1.0 - sd_away) * 60
        elif sd_away <= 1.5:
            touch_prob = 15 + (1.5 - sd_away) * 30
        elif sd_away <= 2.0:
            touch_prob = 5 + (2.0 - sd_away) * 20
        else:
            touch_prob = 3
        touch_prob = min(95, max(3, touch_prob))
        
        # ML direction alignment
        ml_direction = "SIDEWAYS"
        ml_confidence = 50.0
        regime = "UNKNOWN"
        if pred:
            regime = pred.get("regime", "CHOPPY")
            ep = pred.get("ensemble_prediction")
            tp = pred.get("tree_prediction")
            if ep:
                ml_direction = ep.get("direction", "SIDEWAYS")
                ml_confidence = ep.get("confidence", 0.5) * 100
            elif tp:
                ml_direction = tp.get("direction", "SIDEWAYS")
                ml_confidence = tp.get("confidence", 0.5) * 100
        
        # Alignment bonus
        aligned = False
        if target > spot and ml_direction == "UP":
            aligned = True
            touch_prob = min(95, touch_prob + 10)
        elif target < spot and ml_direction == "DOWN":
            aligned = True
            touch_prob = min(95, touch_prob + 10)
        elif (target > spot and ml_direction == "DOWN") or (target < spot and ml_direction == "UP"):
            touch_prob = max(3, touch_prob - 10)
        
        # Estimated time to reach based on daily volatility
        avg_daily_pts = spot * (vix / 100.0) / np.sqrt(252)
        est_days = max(1, int(dist / max(1, avg_daily_pts)))
        if aligned:
            est_days = max(1, int(est_days * 0.8)) # Faster if aligned with trend
            
        # Build display
        if touch_prob >= 65:
            prob_color = ACCENT_GREEN
            verdict = "HIGH probability"
            emoji = "🟢"
        elif touch_prob >= 40:
            prob_color = ACCENT_GOLD
            verdict = "MODERATE probability"
            emoji = "🟡"
        else:
            prob_color = ACCENT_RED
            verdict = "LOW probability"
            emoji = "🔴"
        
        html = (
            f"<div style='font-size: 14px;'>"
            f"<b style='font-size: 16px;'>{emoji} {verdict} ({touch_prob:.0f}%) of NIFTY reaching {target:,}</b><br><br>"
            f"<b>📊 Statistical Physics:</b><br>"
            f"&nbsp;&nbsp;• Current Spot: <b>{spot:,.2f}</b><br>"
            f"&nbsp;&nbsp;• Distance to Target: <b>{dist:,.0f} pts</b> {direction_to_target}<br>"
            f"&nbsp;&nbsp;• Expected {days_to_expiry}-day Move (VIX {vix:.1f}): <b>±{expected_move:.0f} pts</b><br>"
            f"&nbsp;&nbsp;• Standard Deviations Away: <b>{sd_away:.2f} σ</b><br>"
            f"&nbsp;&nbsp;• Touch Probability: <b style='color: {prob_color};'>{touch_prob:.0f}%</b><br><br>"
            f"<b>🧠 AI Regime Context:</b><br>"
            f"&nbsp;&nbsp;• ML Direction: <b>{ml_direction}</b> ({ml_confidence:.0f}% confidence)<br>"
            f"&nbsp;&nbsp;• Current Regime: <b>{regime}</b><br>"
            f"&nbsp;&nbsp;• AI {'✅ ALIGNED' if aligned else '❌ OPPOSING'} with your target direction<br>"
        )
        
        if aligned and touch_prob >= 50:
            html += f"&nbsp;&nbsp;• <b style='color: {ACCENT_GREEN};'>🛡️ HOLD your position. AI supports this bounce. Est. ~{est_days} trading days.</b><br>"
        elif touch_prob >= 40:
            html += f"&nbsp;&nbsp;• <b style='color: {ACCENT_GOLD};'>⚠️ Within reach but uncertain. Consider partial hedge.</b><br>"
        else:
            html += f"&nbsp;&nbsp;• <b style='color: {ACCENT_RED};'>🚨 Very unlikely to reach {target:,}. Consider rolling/closing.</b><br>"
        
        html += "</div>"
        
        self.war_target_result.setText(html)
        self.war_target_result.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; padding: 12px; "
            f"background-color: {DARK_CARD}; border-radius: 8px; "
            f"border-left: 3px solid {prob_color};"
        )
                
    def _on_analyze_war_room(self):
        """Analyze the war room trade with AI."""
        from analyzers.options_payoff import compute_payoff, ai_win_probability, suggest_firefight
        
        legs = self._get_war_legs_data()
        if not legs:
            return
        
        # Get spot price from last prediction
        spot = getattr(self, '_last_spot', 24500)
        pred = getattr(self, '_last_prediction', None)
        
        # Compute days to expiry
        try:
            expiry = datetime.strptime(self.war_expiry_date.text(), "%Y-%m-%d")
            entry = datetime.strptime(self.war_entry_date.text(), "%Y-%m-%d")
            days_to_expiry = max(1, (expiry - entry).days)
        except ValueError:
            days_to_expiry = 5
        
        # Compute payoff
        realized_pnl = self.war_realized_pnl.value()
        payoff = compute_payoff(legs, spot_price=spot, realized_pnl=realized_pnl)
        
        # Draw chart
        self.war_chart.ax.clear()
        self.war_chart.ax.fill_between(
            payoff["prices"], payoff["payoff"], 0,
            where=payoff["payoff"] >= 0, alpha=0.3, color='#00FF7F'
        )
        self.war_chart.ax.fill_between(
            payoff["prices"], payoff["payoff"], 0,
            where=payoff["payoff"] < 0, alpha=0.3, color='#FF4B4B'
        )
        self.war_chart.ax.plot(payoff["prices"], payoff["payoff"], color='#00CED1', linewidth=2)
        self.war_chart.ax.axhline(y=0, color='#555', linewidth=0.8)
        self.war_chart.ax.axvline(x=spot, color='#FFD700', linewidth=1.5, linestyle='--', label=f'Spot: {spot:,.0f}')
        
        # Mark breakevens
        for be in payoff["breakevens"]:
            self.war_chart.ax.axvline(x=be, color='#FF4B4B', linewidth=1, linestyle=':', alpha=0.7)
            self.war_chart.ax.annotate(f'BE: {be:,.0f}', xy=(be, 0), fontsize=8, color='#FF4B4B',
                                       ha='center', va='bottom')
        
        # Mark each leg's strike on the chart
        strike_colors = {'SELL': '#FF6B6B', 'BUY': '#00CED1'}
        for leg in legs:
            if leg.get('status', 'Open') == 'Open':
                clr = strike_colors.get(leg['action'].upper(), '#888')
                self.war_chart.ax.axvline(x=leg['strike'], color=clr, linewidth=1, linestyle='--', alpha=0.5)
                self.war_chart.ax.annotate(
                    f"{leg['action'][0]}{leg['option_type']} {leg['strike']}",
                    xy=(leg['strike'], payoff['max_profit'] * 0.8), fontsize=7, color=clr,
                    ha='center', va='top', rotation=90, alpha=0.8
                )
        
        self.war_chart.ax.set_xlabel('NIFTY Spot Price', color='#888')
        self.war_chart.ax.set_ylabel('P&L (₹)', color='#888')
        self.war_chart.ax.set_title('Payoff at Expiry', color='#fff', fontsize=12)
        self.war_chart.ax.legend(loc='upper left', fontsize=9)
        self.war_chart.ax.tick_params(colors='#888')
        self.war_chart.draw()
        
        # AI Verdict
        ai_result = ai_win_probability(legs, pred, spot, days_to_expiry, realized_pnl=realized_pnl)
        
        # Update cards
        win_color = ACCENT_GREEN if ai_result["win_prob"] >= 60 else ACCENT_GOLD if ai_result["win_prob"] >= 40 else ACCENT_RED
        self.war_win_prob.set_value(f"{ai_result['win_prob']:.0f}%", win_color)
        self.war_max_profit.set_value(f"₹{ai_result['max_profit']:,.0f}", ACCENT_GREEN)
        self.war_max_loss.set_value(f"₹{ai_result['max_loss']:,.0f}", ACCENT_RED)
        
        if ai_result["breakevens"]:
            be_str = " / ".join([f"{b:,.0f}" for b in ai_result["breakevens"]])
            self.war_breakeven.set_value(be_str, ACCENT_CYAN)
        else:
            self.war_breakeven.set_value("N/A")
        
        # Verdict text
        self.war_verdict_text.setText(f"<b>{ai_result['verdict']}</b>")
        
        # Firefight
        bounce_html = ""
        if ai_result.get("bounce_analysis"):
            bounce_html = f"<br><br>{ai_result['bounce_analysis']}"
            
        if ai_result["firefight_needed"]:
            firefight_suggestions = suggest_firefight(legs, spot, pred)
            ff_html = "<br>".join(firefight_suggestions)
            self.war_firefight_text.setText(f"🔥 <b>FIREFIGHT NEEDED:</b> {ai_result['firefight_reason']}<br>{ff_html}{bounce_html}")
            self.war_firefight_text.setStyleSheet(f"color: {ACCENT_RED}; font-size: 13px; padding: 8px; background-color: #2a1a1a; border-radius: 8px;")
        else:
            self.war_firefight_text.setText("🛡️ No firefight needed. Trade looks manageable.")
            self.war_firefight_text.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 13px; padding: 8px;")
        
        # Reasons
        reasons = ai_result.get("reasoning", [])
        if reasons:
            reasons_html = "<br>".join([f"• {r}" for r in reasons])
            self.war_reasons_text.setText(f"<b>AI Analysis Breakdown:</b><br>{reasons_html}")
        
        # ── Risk Dashboard ──
        from analyzers.options_payoff import compute_risk_metrics
        vix = pred.get("vix_value", 15.0) if pred else 15.0
        risk = compute_risk_metrics(legs, spot, vix or 15.0, days_to_expiry, realized_pnl)
        
        pop_color = ACCENT_GREEN if risk["pop"] >= 60 else ACCENT_GOLD if risk["pop"] >= 40 else ACCENT_RED
        self.war_pop.set_value(f"{risk['pop']:.1f}%", pop_color)
        
        delta_color = ACCENT_CYAN if abs(risk["positional_delta"]) < 1.0 else ACCENT_GOLD
        self.war_delta.set_value(f"{risk['positional_delta']:+.2f}", delta_color)
        
        theta_color = ACCENT_GREEN if risk["theta_per_day"] > 0 else ACCENT_RED
        self.war_theta.set_value(f"₹{risk['theta_per_day']:+,.0f}", theta_color)
        
        rr_color = ACCENT_GREEN if risk["risk_reward"] >= 0.3 else ACCENT_RED
        self.war_rr.set_value(f"{risk['risk_reward']:.2f}", rr_color)
        
        if risk["safe_zone"][0] and risk["safe_zone"][1]:
            self.war_safe_zone.set_value(f"{risk['safe_zone'][0]:,.0f}–{risk['safe_zone'][1]:,.0f}", ACCENT_CYAN)
        else:
            self.war_safe_zone.set_value("N/A", ACCENT_RED)
        
        self.war_margin.set_value(f"₹{risk['margin_estimate']:,.0f}", TEXT_SEC)
        
        # Risk Verdict
        risk_verdicts = []
        if risk["pop"] >= 65:
            risk_verdicts.append(f"✅ <b>High PoP ({risk['pop']:.0f}%)</b> — Probability favors your credit strategy.")
        elif risk["pop"] < 40:
            risk_verdicts.append(f"⚠️ <b>Low PoP ({risk['pop']:.0f}%)</b> — Consider widening your strikes or reducing position size.")
        
        if risk["theta_per_day"] > 0:
            risk_verdicts.append(f"✅ <b>Positive Theta (₹{risk['theta_per_day']:+,.0f}/day)</b> — Time decay is working for you.")
        else:
            risk_verdicts.append(f"⚠️ <b>Negative Theta</b> — Time decay is working against you (debit position).")
        
        if abs(risk["positional_delta"]) < 0.5:
            risk_verdicts.append(f"✅ <b>Delta Neutral ({risk['positional_delta']:+.2f})</b> — Position is well hedged.")
        elif abs(risk["positional_delta"]) >= 1.5:
            risk_verdicts.append(f"⚠️ <b>High Delta ({risk['positional_delta']:+.2f})</b> — Position is directionally exposed.")
        
        if risk["safe_zone"][0] and risk["safe_zone"][1]:
            zone_width = risk["safe_zone"][1] - risk["safe_zone"][0]
            exp_move = risk["expected_move"]
            if zone_width >= exp_move * 2:
                risk_verdicts.append(f"✅ <b>Wide Safe Zone ({zone_width:,.0f} pts)</b> — Comfortably covers the Expected Move (±{exp_move:,.0f}).")
            else:
                risk_verdicts.append(f"⚠️ <b>Narrow Safe Zone ({zone_width:,.0f} pts)</b> — Expected Move (±{exp_move:,.0f}) could breach it.")
        
        self.war_risk_verdict.setText("<br>".join(risk_verdicts))

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: 📊 WEEKLY REPORT — Performance Tracking + Drawdown Shield
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_weekly_report_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {DARK_BG}; }}")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        title = QLabel("📊 Weekly Report Card")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 22px; font-weight: bold;")
        layout.addWidget(title)
        
        # ── Drawdown Shield Status ──
        self.drawdown_status = QLabel("🛡️ Drawdown Shield: Loading...")
        self.drawdown_status.setWordWrap(True)
        self.drawdown_status.setStyleSheet(f"""
            color: {TEXT_PRIMARY}; font-size: 14px; padding: 12px;
            background-color: {DARK_CARD}; border-radius: 8px;
            border-left: 4px solid {ACCENT_CYAN};
        """)
        layout.addWidget(self.drawdown_status)
        
        # ── Performance Summary Cards ──
        perf_row = QHBoxLayout()
        self.wr_win_rate = MetricCard("WIN RATE", "—", ACCENT_GREEN)
        self.wr_net_pnl = MetricCard("NET P&L", "—", ACCENT_CYAN)
        self.wr_streak = MetricCard("STREAK", "—", ACCENT_GOLD)
        self.wr_total = MetricCard("TOTAL SIGNALS", "—", TEXT_PRIMARY)
        
        perf_row.addWidget(self.wr_win_rate)
        perf_row.addWidget(self.wr_net_pnl)
        perf_row.addWidget(self.wr_streak)
        perf_row.addWidget(self.wr_total)
        layout.addLayout(perf_row)
        
        # ── Weekly Table ──
        self.wr_table = QTableWidget(0, 5)
        self.wr_table.setHorizontalHeaderLabels(["Date", "Signal", "Predicted", "Actual", "Result"])
        self.wr_table.horizontalHeader().setStretchLastSection(True)
        self.wr_table.verticalHeader().setVisible(False)
        self.wr_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.wr_table.setMaximumHeight(250)
        layout.addWidget(self.wr_table)
        
        # ── Refresh Button ──
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Refresh Report")
        btn_refresh.setStyleSheet(f"background-color: {ACCENT_CYAN}; color: #000; font-weight: bold; padding: 8px 16px; border-radius: 6px;")
        btn_refresh.clicked.connect(self._refresh_weekly_report)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        
        # ── Info ──
        info = QLabel(
            "💡 <b>How it works:</b> Every time you Fetch + Predict, "
            "David automatically logs the signal. After market close, "
            "update the 'Actual' direction to see your win rate grow."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px; padding: 10px;")
        layout.addWidget(info)
        
        layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        
        # Initial load
        QTimer.singleShot(500, self._refresh_weekly_report)
        
        return tab
    
    def _refresh_weekly_report(self):
        """Refresh the weekly report tab with latest data."""
        from analyzers.trade_journal import generate_weekly_report, get_drawdown_status
        
        # Drawdown Shield
        dd = get_drawdown_status()
        if dd["active"]:
            self.drawdown_status.setText(dd["message"])
            self.drawdown_status.setStyleSheet(f"""
                color: {TEXT_PRIMARY}; font-size: 14px; padding: 12px;
                background-color: #2a1a1a; border-radius: 8px;
                border-left: 4px solid {ACCENT_RED};
            """)
        else:
            self.drawdown_status.setText(dd["message"])
            self.drawdown_status.setStyleSheet(f"""
                color: {TEXT_PRIMARY}; font-size: 14px; padding: 12px;
                background-color: {DARK_CARD}; border-radius: 8px;
                border-left: 4px solid {ACCENT_GREEN};
            """)
        
        # Weekly Report
        report = generate_weekly_report()
        
        wr = report["win_rate"]
        wr_color = ACCENT_GREEN if wr >= 60 else ACCENT_GOLD if wr >= 40 else ACCENT_RED
        self.wr_win_rate.set_value(f"{wr:.0f}%", wr_color)
        
        pnl = report["net_pnl"]
        pnl_color = ACCENT_GREEN if pnl >= 0 else ACCENT_RED
        self.wr_net_pnl.set_value(f"₹{pnl:,.0f}", pnl_color)
        
        streak = report["streak"]
        if streak > 0:
            self.wr_streak.set_value(f"{streak}W 🔥", ACCENT_GREEN)
        elif streak < 0:
            self.wr_streak.set_value(f"{abs(streak)}L", ACCENT_RED)
        else:
            self.wr_streak.set_value("—")
        
        self.wr_total.set_value(str(report["total_entries"]))
        
        # Table
        entries = report["entries"]
        self.wr_table.setRowCount(len(entries))
        for i, entry in enumerate(entries):
            self.wr_table.setItem(i, 0, QTableWidgetItem(entry.get("date", "")))
            
            signal = entry.get("signal", "")
            signal_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(signal, "")
            self.wr_table.setItem(i, 1, QTableWidgetItem(f"{signal_emoji} {signal}"))
            
            self.wr_table.setItem(i, 2, QTableWidgetItem(entry.get("predicted", "—")))
            self.wr_table.setItem(i, 3, QTableWidgetItem(entry.get("actual", "—")))
            
            result = entry.get("result", "—")
            result_emoji = {"WIN": "✅ WIN", "LOSS": "❌ LOSS", "AVOIDED": "⏸️ AVOIDED"}.get(result, "—")
            self.wr_table.setItem(i, 4, QTableWidgetItem(result_emoji))

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6: DAVID CODEX — A-to-Z Trading Guide
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_codex_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background-color: {DARK_BG}; }}
        """)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Title
        title = QLabel("📖 DAVID CODEX — The Complete Trading Manual")
        title.setStyleSheet(f"color: {ACCENT_GOLD}; font-size: 22px; font-weight: bold;")
        layout.addWidget(title)
        
        subtitle = QLabel("Everything you need to trade with David Oracle. From A to Z.")
        subtitle.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px;")
        layout.addWidget(subtitle)
        
        # Helper to create styled sections
        def add_section(title_text, content_text, color=ACCENT_CYAN):
            group = QGroupBox(title_text)
            group.setStyleSheet(f"""
                QGroupBox {{
                    border: 1px solid {DARK_BORDER};
                    border-radius: 8px;
                    margin-top: 12px;
                    padding: 16px 12px 12px 12px;
                    font-size: 15px;
                    font-weight: bold;
                    color: {color};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 14px;
                    padding: 0 8px;
                }}
            """)
            g_layout = QVBoxLayout(group)
            lbl = QLabel(content_text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; line-height: 1.6;")
            lbl.setTextFormat(Qt.RichText)
            g_layout.addWidget(lbl)
            layout.addWidget(group)
        
        # ── CHAPTER 1: How David Works ──
        add_section("📡 Chapter 1: How David Works", f"""
<p>David Oracle is a <b style='color:{ACCENT_CYAN}'>hybrid AI engine</b> that combines multiple machine learning models to predict NIFTY's next-day direction.</p>

<p><b>The 3-Button Workflow:</b></p>
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr><td style='color:{ACCENT_CYAN}'>📊 Sync Data (F6)</td><td>Downloads 15 years of NIFTY, VIX, SP500, FII/DII, PCR data into CSVs</td></tr>
<tr><td style='color:{ACCENT_GOLD}'>🧠 Train (F7)</td><td>Trains 4 ML models (XGBoost, LightGBM, CatBoost, HMM) from CSVs → saves .pkl</td></tr>
<tr><td style='color:{ACCENT_RED}'>🔴 Fetch Spot (F5)</td><td>Gets live NIFTY + VIX → mixes with trained .pkl → <b>instant verdict</b></td></tr>
</table>

<p style='color:{TEXT_DIM}'><i>Sync weekly. Train weekly. Fetch Spot daily at 3:15 PM.</i></p>
""")
        
        # ── CHAPTER 2: Reading the Dashboard ──
        add_section("🧠 Chapter 2: Reading the Dashboard Signals", f"""
<p><b style='color:{ACCENT_GREEN}'>Signal 1 — The Verdict</b></p>
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr><td style='color:{ACCENT_GREEN}'>🟢 UP</td><td>AI expects NIFTY to close higher tomorrow</td><td>→ Bull Call Spread</td></tr>
<tr><td style='color:{ACCENT_RED}'>🔴 DOWN</td><td>AI expects NIFTY to close lower tomorrow</td><td>→ Bear Put Spread</td></tr>
<tr><td style='color:{ACCENT_GOLD}'>🟡 SIDEWAYS</td><td>AI can't decide — market is flat</td><td>→ Short Iron Condor</td></tr>
</table>

<p><b style='color:{ACCENT_GREEN}'>Signal 2 — Confidence</b></p>
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr><td style='color:{ACCENT_GREEN}'>60%+</td><td>⭐ High conviction</td><td>Full 2-lot position</td></tr>
<tr><td style='color:{ACCENT_GOLD}'>50-60%</td><td>◆ Moderate</td><td>1-lot position</td></tr>
<tr><td style='color:{ACCENT_RED}'>40-50%</td><td>○ Low conviction</td><td>Iron Condor only</td></tr>
<tr><td style='color:{ACCENT_RED}'>&lt;40%</td><td>❌ No signal</td><td><b>DO NOT TRADE</b></td></tr>
</table>

<p><b style='color:{ACCENT_GREEN}'>Signal 3 — Regime</b></p>
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr><td style='color:{ACCENT_GREEN}'>TRENDING</td><td>Strong directional move</td><td>→ Follow verdict with directional spreads</td></tr>
<tr><td style='color:{ACCENT_GOLD}'>CHOPPY</td><td>Grinding sideways</td><td>→ Iron Condors, sell premium</td></tr>
<tr><td style='color:{ACCENT_RED}'>VOLATILE</td><td>Big swings both ways</td><td>→ Wider spreads, smaller size, or SKIP</td></tr>
</table>

<p><br/><b style='color:{ACCENT_CYAN}'>Signal 4 — Support & Resistance</b></p>
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr><td style='color:{ACCENT_GOLD}'>Level</td><td>The actual NIFTY price level</td></tr>
<tr><td style='color:{ACCENT_GOLD}'>Distance</td><td>How far away this level is from the current price (%)</td></tr>
<tr><td style='color:{ACCENT_GOLD}'>Strength</td><td>How many times the market bounced or rejected here in the past (higher score = stronger level)</td></tr>
</table>

<p style='color:{TEXT_DIM}'><i>Tip: If Verdict says SIDEWAYS but Forecast tilts UP — it's NOT a bug. They come from different models. SIDEWAYS at low confidence = Iron Condor.</i></p>
""")
        # ── CHAPTER 3: Zero Anxiety System (New) ──
        add_section("🛡️ Chapter 3: Zero Anxiety System", f"""
<p><b style='color:{ACCENT_CYAN}'>1. Command Center & Traffic Light</b></p>
<p>The smartest page in the app. David analyzes all 15 metrics and gives you a single signal:</p>
<table style='color:{TEXT_PRIMARY}; width:100%; border: 1px solid #333;'>
<tr><td style='color:{ACCENT_GREEN}'>🟢 GREEN</td><td>High Confidence (>60%), Low VIX, Regime Aligned</td><td><b>GO</b> (Entry Safe, Full Size)</td></tr>
<tr><td style='color:{ACCENT_GOLD}'>🟡 YELLOW</td><td>Mixed signals, Medium VIX</td><td><b>CAUTION</b> (Half Size or Wait)</td></tr>
<tr><td style='color:{ACCENT_RED}'>🔴 RED</td><td>Low Confidence (<45%), High VIX, Conflicting Data</td><td><b>STOP</b> (Do Not Trade)</td></tr>
</table>

<p><b style='color:{ACCENT_CYAN}'>2. Trading War Room</b></p>
<p>Once you get a strategy from the Command Center, enter the exact strikes and premium prices into the War Room. David will show you the exact Payoff Graph at expiry, your Max Profit/Loss, and the <b>AI Win Probability</b>. If the trade starts losing after entry, the War Room will offer <b>Firefight Suggestions</b> (like adding a hedge or rolling strikes).</p>

<p><b style='color:{ACCENT_CYAN}'>3. Position Sizing & Drawdown Shield</b></p>
<p>Input your trading capital. David mathematically caps your risk based on the traffic light color (e.g., max 3% risk on GREEN, 0% on RED). If you lose 3 trades in a row, the <b>Drawdown Shield</b> activates and locks you out of new signals for a 2-day cooling period.</p>
""")

        # ── CHAPTER 4: Daily Checklist ──
        add_section("📋 Chapter 3: The Daily Trading Checklist", f"""
<p style='color:{ACCENT_GOLD}'><b>Do this every trading day at 3:15 PM:</b></p>
<ol style='color:{TEXT_PRIMARY}'>
<li>Press <b style='color:{ACCENT_RED}'>🔴 Fetch Spot (F5)</b> to get the latest prediction</li>
<li>Check <b>Confidence</b>:
    <br/>→ &lt;40%: <b style='color:{ACCENT_RED}'>SKIP. Cash is King.</b>
    <br/>→ 40-60%: Proceed with 1 lot only
    <br/>→ 60%+: Full conviction (2 lots)</li>
<li>Check <b>Regime</b>:
    <br/>→ TRENDING: Directional spreads
    <br/>→ CHOPPY: Iron Condor
    <br/>→ VOLATILE: Reduce size or skip</li>
<li>Check <b>Whipsaw</b>:
    <br/>→ CHOPPY whipsaw: Don't chase breakouts
    <br/>→ CLEAN trend: Follow the signal</li>
<li>Check <b>Support &amp; Resistance</b> — use these as strike selection anchors</li>
<li>Enter trade using the Strategy Matrix (Chapter 4)</li>
<li>Set exit rules (Chapter 6)</li>
</ol>
""")

        # ── CHAPTER 4: Strategy Matrix ──
        add_section("🎯 Chapter 4: Strategy Matrix & Trade Examples", f"""
<p><b style='color:{ACCENT_GREEN}'>When Verdict = UP → Bull Call Spread</b></p>
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr><td>Buy</td><td>ATM or 1-strike ITM Call</td></tr>
<tr><td>Sell</td><td>1-2 strikes OTM Call</td></tr>
<tr><td>Spread Width</td><td>100-150 points</td></tr>
<tr><td>Max Risk</td><td>Spread width minus premium received</td></tr>
<tr><td>Target</td><td>60-70% of max profit</td></tr>
<tr><td>Stop</td><td>Exit if NIFTY breaks below nearest Support</td></tr>
</table>
<p style='color:{TEXT_DIM}'><i>Example (NIFTY 24,450): Buy 24400 CE, Sell 24550 CE → Net debit ~₹55 → Max profit ₹95</i></p>

<p><b style='color:{ACCENT_RED}'>When Verdict = DOWN → Bear Put Spread</b></p>
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr><td>Buy</td><td>ATM or 1-strike ITM Put</td></tr>
<tr><td>Sell</td><td>1-2 strikes OTM Put</td></tr>
<tr><td>Target</td><td>60-70% of max profit</td></tr>
<tr><td>Stop</td><td>Exit if NIFTY breaks above nearest Resistance</td></tr>
</table>

<p><b style='color:{ACCENT_GOLD}'>When Verdict = SIDEWAYS / Low Confidence → Short Iron Condor</b></p>
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr><td>Sell</td><td>OTM Call at nearest Resistance</td></tr>
<tr><td>Buy</td><td>OTM Call 100pts above (protection)</td></tr>
<tr><td>Sell</td><td>OTM Put at nearest Support</td></tr>
<tr><td>Buy</td><td>OTM Put 100pts below (protection)</td></tr>
<tr><td>Target</td><td>50% of premium collected</td></tr>
<tr><td>Holding</td><td>1-3 days</td></tr>
</table>
""")

        # ── CHAPTER 5: Intraday ML & Entry Timing ──
        add_section("⚡ Chapter 5: Intraday ML & Entry Timing", f"""
<p>The <b style='color:{ACCENT_CYAN}'>15-Minute Intraday ML</b> tab uses a specialized XGBoost model trained on 67,000+ bars of historical intraday data. Use it to time your entries.</p>

<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr style='color:{ACCENT_CYAN}'><td><b>Signal</b></td><td><b>Meaning</b></td><td><b>Action</b></td></tr>
<tr><td style='color:{ACCENT_GREEN}'>✅ GOOD</td><td>Optimal entry window (usually 2:30 PM - 3:30 PM)</td><td>Enter your spread/condor</td></tr>
<tr><td style='color:{ACCENT_RED}'>⚠️ WAIT</td><td>Market is overheated (e.g., Morning gap-up with RSI > 70)</td><td>Do not enter yet. Wait for mean reversion.</td></tr>
<tr><td style='color:{ACCENT_GOLD}'>⚠️ CAUTION</td><td>VIX is elevated or conditions are unstable</td><td>Wait for stability before entering</td></tr>
</table>

<p><b style='color:{ACCENT_RED}'>What is REVERSAL_RISK?</b></p>
<p style='color:{TEXT_PRIMARY}'>If the 15-Minute Regime says <b>REVERSAL_RISK</b>, it means the intraday momentum is wildly overbought or oversold (like a rubber band stretched too far). A snap-back is highly likely. <b>Do not enter a trend-following trade right now.</b> Wait for the mean reversion to finish.</p>

<p style='color:{TEXT_DIM}'><i>Rule of Thumb: The Daily Verdict tells you WHAT to trade. The Intraday ML tells you WHEN to enter.</i></p>
""")

        # ── CHAPTER 6: Position Manager ──
        add_section("🛡️ Chapter 6: Position Manager (Hold/Exit Engine)", f"""
<p>The <b style='color:{ACCENT_CYAN}'>Position Manager</b> is your trading backbone. It tracks your open trades and tells you when to hold or cut.</p>

<p><b style='color:{ACCENT_GOLD}'>1. 🎯 David's Verdict (Hold/Exit Signal)</b><br/>
<span style='color:{TEXT_SEC}'><b>Step-by-step:</b> Log your exact trade type (Debit vs Credit Spread). Click "Fetch Spot". David compares your bet against the latest ML prediction.<br/>
<b>ELI5:</b> David checks if his latest guess matches your trade. If you hold a Debit spread, you need momentum, so Sideways = EXIT. If you hold a Credit spread, Sideways is fine (theta decay), so Sideways = HOLD.</span></p>

<p><b style='color:{ACCENT_GOLD}'>2. 📈 Recovery Probability (Panic Preventer)</b><br/>
<span style='color:{TEXT_SEC}'><b>Step-by-step:</b> If you are in a drawdown (e.g., down 1.5%), David scans 15 years of history for similar dips and calculates the recovery rate.<br/>
<b>ELI5:</b> It tells you if this dip is just normal noise (high probability of bouncing back) or a real market crash (low probability, cut it).</span></p>

<p><b style='color:{ACCENT_GOLD}'>3. ⚠️ Drawdown Alert System (Circuit Breaker)</b><br/>
<span style='color:{TEXT_SEC}'><b>Step-by-step:</b> Tracks your drawdown severity (SAFE to EMERGENCY) and consecutive loss streak.<br/>
<b>ELI5:</b> If it flashes Red/Emergency, or if your Loss Streak shines red, your risk rules failed. Stop trading and walk away.</span></p>

<p><b style='color:{ACCENT_GOLD}'>4. ⏱️ 15-Min Intraday Regime (Heartbeat Monitor)</b><br/>
<span style='color:{TEXT_SEC}'><b>Step-by-step:</b> Uses 15-minute candles to calculate an Entry Quality score (0-100) based on RSI and VIX.<br/>
<b>ELI5:</b> Even if the Daily model says "Go UP!", if the intraday heartbeat is too fast (overheated), it tells you to wait a few hours before entering.</span></p>
""")

        # ── CHAPTER 7: Risk Management ──
        add_section("💰 Chapter 7: Risk Management & Capital Allocation", f"""
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr style='color:{ACCENT_CYAN}'><td><b>Capital</b></td><td><b>Per Trade</b></td><td><b>Max Open</b></td></tr>
<tr><td>₹1 Lakh</td><td>1 lot (₹5,000-7,000 margin)</td><td>2 positions</td></tr>
<tr><td>₹3 Lakh</td><td>2 lots</td><td>3 positions</td></tr>
<tr><td>₹5 Lakh</td><td>3-4 lots</td><td>4 positions</td></tr>
<tr><td>₹10 Lakh+</td><td>5 lots max</td><td>5 positions</td></tr>
</table>

<p style='color:{ACCENT_RED}'><b>Golden Rule: Never risk more than 2% of capital on a single trade.</b></p>
<p style='color:{TEXT_PRIMARY}'>For Spreads: Your max loss = spread width – premium received. Always buy the protection wing.</p>
<p style='color:{TEXT_PRIMARY}'>For Iron Condors: Your max loss per side = spread width – premium collected.</p>
""")

        # ── CHAPTER 8: Exit Rules ──
        add_section("⏱️ Chapter 8: When to Exit — The Exit Playbook", f"""
<p><b style='color:{ACCENT_GREEN}'>EXIT WITH PROFIT:</b></p>
<ol style='color:{TEXT_PRIMARY}'>
<li>Target hit (60-70% of max) → <b>Exit immediately. Don't wait for more.</b></li>
<li>Iron Condor premium decayed 50% → Book it.</li>
</ol>

<p><b style='color:{ACCENT_RED}'>EXIT FOR PROTECTION:</b></p>
<ol style='color:{TEXT_PRIMARY}'>
<li><b>Confidence drops sharply</b> (e.g., 65% UP → 42% SIDEWAYS next day) → Close directional trade</li>
<li><b>Regime changes</b> (TRENDING → VOLATILE) → Close all directional. Switch to Iron Condor</li>
<li><b>The 2-Loss Rule</b>: If you lose on 2 Iron Condors in a row → Stop Condors for 1 week</li>
<li><b>Never hold over expiry</b> if your strikes are anywhere near spot</li>
<li>On directional spreads: <b>Cut losses at 100% of premium paid</b></li>
</ol>

<p><b>Average Holding Periods (from backtest):</b></p>
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr><td>Bull/Bear Spread</td><td>1-2 days</td><td>Exit at 60-70% of max profit</td></tr>
<tr><td>Iron Condor</td><td>2-3 days</td><td>Exit at 50% premium decay</td></tr>
</table>
""")

        # ── CHAPTER 9: When NOT to Trade ──
        add_section("🚫 Chapter 9: When NOT to Trade — Red Light Days", f"""
<table style='color:{TEXT_PRIMARY}; width:100%;'>
<tr style='color:{ACCENT_RED}'><td><b>Scenario</b></td><td><b>Why</b></td><td><b>Action</b></td></tr>
<tr><td>Confidence &lt;40%</td><td>David doesn't know</td><td style='color:{ACCENT_RED}'>Sit on hands</td></tr>
<tr><td>Budget Day / Union Budget</td><td>Unpredictable gap moves</td><td style='color:{ACCENT_RED}'>Skip entirely</td></tr>
<tr><td>RBI Policy Day</td><td>Rate decisions cause spikes</td><td style='color:{ACCENT_GOLD}'>Skip or Iron Condor only</td></tr>
<tr><td>F&amp;O Expiry (Thursday)</td><td>Gamma risk, pin risk</td><td style='color:{ACCENT_RED}'>Close open trades</td></tr>
<tr><td>VIX &gt; 25</td><td>Market is wild</td><td style='color:{ACCENT_GOLD}'>Half position size only</td></tr>
<tr><td>Verdict flips UP↔DOWN in 15 min</td><td>AI is guessing</td><td style='color:{ACCENT_RED}'>CANCEL TRADE</td></tr>
<tr><td>VIX up &gt;10% today</td><td>Fear overrides math</td><td style='color:{ACCENT_RED}'>STAY OUT</td></tr>
</table>
""", ACCENT_RED)

        # ── CHAPTER 8: Psychology ──
        add_section("🧠 Chapter 8: Trading Psychology — The Mental Game", f"""
<p style='color:{ACCENT_GOLD}'><b>Trading is 20% Math and 80% Mindset.</b></p>

<p><b style='color:{ACCENT_CYAN}'>The Monday Dip Scenario:</b></p>
<p style='color:{TEXT_PRIMARY}'>You entered a Bull Spread at 24,647. Monday morning, NIFTY drops to 24,450. You're in a loss.</p>
<ul style='color:{TEXT_PRIMARY}'>
<li>Check David at 10:30 AM</li>
<li>If Verdict is still <b>UP</b> and Confidence &gt;40% → <b>HOLD</b></li>
<li>If Verdict flips to SIDEWAYS or DOWN → <b>EXIT IMMEDIATELY</b></li>
<li>You have a protection wing (spread). You know your max loss. Let the math play out.</li>
</ul>

<p><b style='color:{ACCENT_CYAN}'>Core Mindset Rules:</b></p>
<ol style='color:{TEXT_PRIMARY}'>
<li><b>Eyes off the MTM, Eyes on the Signal.</b> Your broker's P&L gauge is designed to make you panic. David's Verdict is designed to make you money.</li>
<li><b>The 40% Floor.</b> Above 40% = trend is valid. Below 40% = AI is guessing. Never trade a guess.</li>
<li><b>Accept the Streak.</b> Backtest showed max losing streak = 8 days. Keep positions small enough to survive it. The 196% recovery follows.</li>
<li><b>The 2-Loss Shield.</b> 2 consecutive losses → stop trading for 3 days. This prevents revenge trading during regime shifts.</li>
</ol>

<p style='color:{ACCENT_GOLD}'><b>🦅 Fly with the math. Survive the chop. Grow the capital.</b></p>
""")

        # ── KEY TAKEAWAYS ──
        add_section("🔑 Key Takeaways", f"""
<ol style='color:{TEXT_PRIMARY}'>
<li>David predicts <b>1-day moves</b> — don't hold spreads for weeks</li>
<li>Confidence below 40% = <b>No Trade</b></li>
<li>Iron Condor is your <b>default</b> when David is uncertain — you profit from time passing</li>
<li>Always check <b>Regime</b> — TRENDING = directional, CHOPPY = Iron Condors</li>
<li>Support/Resistance levels = your <b>strike selection guide</b></li>
<li>Average holding is <b>1-2 days</b> — this is NOT swing trading</li>
<li>David's accuracy is <b>62-66%</b>. 1 in 3 will be wrong. Win with position sizing + conviction filtering</li>
<li><b style='color:{ACCENT_RED}'>Paper trade 2 weeks</b> before using real money</li>
</ol>
""", ACCENT_GOLD)

        layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: 💡 TRADE RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_trade_recommendations_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {DARK_BG}; }}")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(12)
        
        title = QLabel("💡 Trade Recommendations")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 22px; font-weight: bold;")
        layout.addWidget(title)
        
        subtitle = QLabel("AI-driven dynamic strike selection with Black-Scholes pricing, whipsaw detection, and Kelly sizing.")
        subtitle.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        layout.addWidget(subtitle)
        
        # ── Row 1: Risk Intelligence Meters ──
        meters_row = QHBoxLayout()
        
        self.tr_whipsaw_card = MetricCard("⚡ WHIPSAW RISK", "—", ACCENT_GOLD)
        self.tr_whipsaw_card.setToolTip("Probability of price reversals in the next few days. >40% auto-downgrades to Iron Condor.")
        meters_row.addWidget(self.tr_whipsaw_card)
        
        self.tr_regime_card = MetricCard("🛡️ REGIME STABILITY", "—", ACCENT_CYAN)
        self.tr_regime_card.setToolTip("How stable is the current market regime? Low = regime flip imminent.")
        meters_row.addWidget(self.tr_regime_card)
        
        self.tr_sizing_card = MetricCard("📐 POSITION SIZE", "—", ACCENT_GREEN)
        self.tr_sizing_card.setToolTip("Kelly-weighted lots based on AI confidence. 0 = skip trade.")
        meters_row.addWidget(self.tr_sizing_card)
        
        self.tr_expiry_card = MetricCard("📅 DAYS TO EXPIRY", "—", TEXT_SEC)
        self.tr_expiry_card.setToolTip("Trading days until the nearest NIFTY weekly expiry (Thursday).")
        meters_row.addWidget(self.tr_expiry_card)
        
        layout.addLayout(meters_row)
        
        # ── Row 1.5: Greeks Dashboard (Feature 10) ──
        greeks_row = QHBoxLayout()
        
        self.tr_delta_card = MetricCard("δ NET DELTA", "—", ACCENT_CYAN)
        self.tr_delta_card.setToolTip("Directional bias. +10 means you gain ₹10 for every 1pt NIFTY move up.")
        greeks_row.addWidget(self.tr_delta_card)
        
        self.tr_theta_card = MetricCard("θ DAILY DECAY", "—", ACCENT_GREEN)
        self.tr_theta_card.setToolTip("Expected daily income from time decay. Positive is good for sellers.")
        greeks_row.addWidget(self.tr_theta_card)
        
        self.tr_gamma_card = MetricCard("γ GAMMA RISK", "—", ACCENT_GOLD)
        self.tr_gamma_card.setToolTip("Sensitivity of Delta. High Gamma means risk accelerates as prices move.")
        greeks_row.addWidget(self.tr_gamma_card)
        
        layout.addLayout(greeks_row)
        
        # ── Row 2: Strategy Recommendation Card ──
        self.tr_card = QFrame()
        self.tr_card.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_CARD};
                border: 1px solid {DARK_BORDER};
                border-radius: 8px;
            }}
        """)
        tr_layout = QVBoxLayout(self.tr_card)
        tr_layout.setContentsMargins(20, 15, 20, 15)
        
        self.tr_title = QLabel("NO PREDICTION YET")
        self.tr_title.setStyleSheet(f"color: {ACCENT_GOLD}; font-size: 20px; font-weight: bold; border: none;")
        tr_layout.addWidget(self.tr_title)
        
        self.tr_reasoning = QLabel("Run Fetch Spot + Predict to see the AI's exact strike recommendations.")
        self.tr_reasoning.setWordWrap(True)
        self.tr_reasoning.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; border: none; margin-top: 8px;")
        tr_layout.addWidget(self.tr_reasoning)
        
        self.tr_legs_label = QLabel("")
        self.tr_legs_label.setWordWrap(True)
        self.tr_legs_label.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 14px; font-family: monospace; border: none; margin-top: 8px;")
        tr_layout.addWidget(self.tr_legs_label)
        
        self.tr_firefight = QLabel("")
        self.tr_firefight.setWordWrap(True)
        self.tr_firefight.setStyleSheet(f"color: {TEXT_SEC}; font-size: 12px; border: none; margin-top: 8px; padding: 10px; background-color: rgba(255, 75, 75, 0.1); border-left: 3px solid {ACCENT_RED};")
        tr_layout.addWidget(self.tr_firefight)
        
        layout.addWidget(self.tr_card)
        
        # ── Row 3: Bounce-Back Estimator ──
        self.tr_bounce_card = QFrame()
        self.tr_bounce_card.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(0, 206, 209, 0.05);
                border: 1px solid rgba(0, 206, 209, 0.3);
                border-radius: 8px;
            }}
        """)
        bounce_layout = QVBoxLayout(self.tr_bounce_card)
        bounce_layout.setContentsMargins(15, 12, 15, 12)
        
        self.tr_bounce_label = QLabel("📊 Historical Bounce-Back Estimator: Run prediction to analyze...")
        self.tr_bounce_label.setWordWrap(True)
        self.tr_bounce_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; border: none;")
        bounce_layout.addWidget(self.tr_bounce_label)
        
        layout.addWidget(self.tr_bounce_card)
        
        # ── Row 4: Payoff Graph ──
        graph_title = QLabel("📊 Projected Payoff Graph at Expiry")
        graph_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 16px; font-weight: bold; margin-top: 5px;")
        layout.addWidget(graph_title)
        
        self.tr_chart = ChartCanvas(width=10, height=4)
        layout.addWidget(self.tr_chart, stretch=1)
        
        # ── Row 5: Export to War Room + History ──
        bottom_row = QHBoxLayout()
        
        self.tr_export_btn = QPushButton("⚔️  Export to War Room")
        self.tr_export_btn.setToolTip("Send these recommended legs to the War Room for live tracking.")
        self.tr_export_btn.clicked.connect(self._on_export_to_war_room)
        self.tr_export_btn.setStyleSheet(f"""
            QPushButton {{
                border-left: 3px solid {ACCENT_GOLD};
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }}
        """)
        bottom_row.addWidget(self.tr_export_btn)
        bottom_row.addStretch()
        
        layout.addLayout(bottom_row)
        
        # ── Row 6: Recommendation History ──
        history_title = QLabel("📜 Recent Recommendations")
        history_title.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px; font-weight: bold; margin-top: 5px;")
        layout.addWidget(history_title)
        
        self.tr_history_label = QLabel("No history yet. Recommendations will be logged after each prediction.")
        self.tr_history_label.setWordWrap(True)
        self.tr_history_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; border: none;")
        layout.addWidget(self.tr_history_label)
        
        layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab

    def _on_export_to_war_room(self):
        """Export the AI-recommended legs to the War Room tab."""
        if not hasattr(self, '_last_trade_rec') or not self._last_trade_rec:
            self.statusBar().showMessage("⚠️ No recommendation to export. Run Fetch + Predict first.")
            return
        
        rec = self._last_trade_rec
        legs = rec.get("legs", [])
        if not legs:
            self.statusBar().showMessage("⚠️ No legs in recommendation to export.")
            return
        
        # Clear existing war room legs
        while len(self.war_legs) > 0:
            self._remove_war_leg(force_clear=True)
        
        # Populate war room with recommended legs
        for leg in legs:
            self._add_war_leg()
            row = self.war_legs[-1]
            row["strike"].setValue(int(leg["strike"]))
            row["type"].setCurrentText(leg["option_type"])
            row["action"].setCurrentText(leg["action"])
            added_leg = self.war_legs[-1]
            added_leg["strike"].setValue(int(leg["strike"]))
            added_leg["opt_type"].setCurrentText(leg["option_type"])
            added_leg["action"].setCurrentText(leg["action"])
            added_leg["premium"].setValue(float(leg["premium"]))
            added_leg["lots"].setValue(int(leg.get("lots", lots))) # Use lots from meta if not specified per leg
        
        # Switch to War Room tab
        for i in range(self.tabs.count()):
            if "War Room" in self.tabs.tabText(i):
                self.tabs.setCurrentIndex(i)
                break
        
        self.statusBar().showMessage(f"✅ Exported {len(legs)} legs to War Room. Click 'Analyze Trade' to see full AI verdict.")

    def _build_price_action_tab(self):
        """Phase 5: Price Action Analysis UI."""
        tab = QWidget()
        tab.setObjectName("PriceActionWidget")
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {DARK_BG}; }}")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)
        
        title = QLabel("📈 Intraday Price Action & Correlation")
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        subtitle = QLabel("Independent technical analysis based on 15m NIFTY and VIX correlation. Not model-driven.")
        subtitle.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        layout.addWidget(subtitle)
        
        # ── Row 1: Quant Gauges ──
        gauges_row = QHBoxLayout()
        
        self.pa_corr_card = MetricCard("🔗 VIX CORRELATION", "—", ACCENT_GOLD)
        self.pa_corr_card.setToolTip("Inverse Relationship: When NIFTY goes up, VIX should go down. If correlation is positive, it's a major warning of a market top.")
        gauges_row.addWidget(self.pa_corr_card)
        
        self.pa_vwap_card = MetricCard("🌀 VWAP DEVIATION", "—", ACCENT_CYAN)
        self.pa_vwap_card.setToolTip("Mean Reversion: Intraday price usually returns to its VWAP. >1.2% is overstretched and due for a pullback.")
        gauges_row.addWidget(self.pa_vwap_card)
        
        self.pa_obv_card = MetricCard("📊 OBV TREND", "—", ACCENT_GOLD)
        self.pa_obv_card.setToolTip("Volume Flow: On-Balance Volume. Cumulative volume that follows price. Bullish if OBV rises with price.")
        gauges_row.addWidget(self.pa_obv_card)

        self.pa_divergence_card = MetricCard("🕵️ DIVERGENCE", "NONE", TEXT_SEC)
        self.pa_divergence_card.setToolTip("Hidden Divergence: Flags when NIFTY makes a new high but VIX stops falling (Smart Money exit) or vice-versa.")
        gauges_row.addWidget(self.pa_divergence_card)
        
        layout.addLayout(gauges_row)
        
        # ── Row 2: Patterns, Levels, & FVGs ──
        mid_row = QHBoxLayout()
        
        # Pattern Card
        pattern_box = QFrame()
        pattern_box.setStyleSheet(f"background-color: {DARK_CARD}; border: 1px solid {DARK_BORDER}; border-radius: 8px;")
        pb_layout = QVBoxLayout(pattern_box)
        pb_title = QLabel("🕯️ CANDLE PATTERNS")
        pb_title.setStyleSheet("font-weight: bold; color: #ddd; border: none;")
        pb_layout.addWidget(pb_title)
        self.pa_patterns_label = QLabel("No 15m patterns detected yet.")
        self.pa_patterns_label.setWordWrap(True)
        self.pa_patterns_label.setStyleSheet("color: #aaa; font-style: italic; border: none;")
        pb_layout.addWidget(self.pa_patterns_label)
        mid_row.addWidget(pattern_box)
        
        # FVG Card (Feature 5.1)
        fvg_box = QFrame()
        fvg_box.setStyleSheet(f"background-color: {DARK_CARD}; border: 1px solid {DARK_BORDER}; border-radius: 8px;")
        fb_layout = QVBoxLayout(fvg_box)
        fb_title = QLabel("🕳️ FAIR VALUE GAPS")
        fb_title.setStyleSheet("font-weight: bold; color: #ddd; border: none;")
        fb_layout.addWidget(fb_title)
        self.pa_fvg_label = QLabel("No imbalances found.")
        self.pa_fvg_label.setWordWrap(True)
        self.pa_fvg_label.setStyleSheet("color: #aaa; font-style: italic; border: none;")
        fb_layout.addWidget(self.pa_fvg_label)
        mid_row.addWidget(fvg_box)

        # Levels Card
        level_box = QFrame()
        level_box.setStyleSheet(f"background-color: {DARK_CARD}; border: 1px solid {DARK_BORDER}; border-radius: 8px;")
        lb_layout = QVBoxLayout(level_box)
        lb_title = QLabel("🧱 INTRADAY LEVELS")
        lb_title.setStyleSheet("font-weight: bold; color: #ddd; border: none;")
        lb_layout.addWidget(lb_title)
        self.pa_levels_label = QLabel("Support: —\nResistance: —")
        self.pa_levels_label.setStyleSheet(f"color: {ACCENT_CYAN}; font-family: monospace; border: none;")
        lb_layout.addWidget(self.pa_levels_label)
        mid_row.addWidget(level_box)
        
        layout.addLayout(mid_row)
        
        # ── Row 3: AI vs PA Alignment ──
        self.pa_alignment_card = QFrame()
        self.pa_alignment_card.setStyleSheet(f"background-color: rgba(255, 215, 0, 0.05); border: 1px solid {ACCENT_GOLD}; border-radius: 8px;")
        align_layout = QVBoxLayout(self.pa_alignment_card)
        
        align_title = QLabel("⚖️ AI VS PRICE ACTION ALIGNMENT")
        align_title.setStyleSheet("font-weight: bold; border: none;")
        align_layout.addWidget(align_title)
        
        self.pa_alignment_text = QLabel("Calculating alignment between ML models and raw price action...")
        self.pa_alignment_text.setWordWrap(True)
        self.pa_alignment_text.setStyleSheet("border: none; font-size: 13px;")
        align_layout.addWidget(self.pa_alignment_text)
        
        layout.addWidget(self.pa_alignment_card)
        
        # ── Row 4: Technical Guide ──
        guide_box = QFrame()
        guide_box.setStyleSheet(f"background-color: {DARK_CARD}; border: 1px solid {DARK_BORDER}; border-radius: 8px; margin-top: 10px;")
        g_layout = QVBoxLayout(guide_box)
        g_title = QLabel("📖 TECHNICAL GUIDE: WHAT DO THESE MEAN?")
        g_title.setStyleSheet("font-weight: bold; color: #888; border: none; font-size: 11px;")
        g_layout.addWidget(g_title)
        
        guide_text = QLabel(
            "• <b>VIX Correlation:</b> Normal is -0.80. If it turns positive (0.10+), it means VIX is rising <i>with</i> price — high risk of a crash.<br>"
            "• <b>Divergence:</b> 'Smart Money' signal. If NIFTY hits a new high but VIX doesn't hit a new low, the rally is fake.<br>"
            "• <b>Fair Value Gaps (FVG):</b> Sudden market imbalances. Price often acts as a magnet to 'fill' these gaps later.<br>"
            "• <b>OBV Trend:</b> Confirms if the move is backed by volume. Price UP + OBV UP = Strong Trend."
        )
        guide_text.setStyleSheet("color: #777; font-size: 11px; border: none;")
        guide_text.setWordWrap(True)
        g_layout.addWidget(guide_text)
        layout.addWidget(guide_box)

        layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab

    def _update_price_action_tab(self, pred):
        """Update Phase 5 tab using 15m data engine."""
        try:
            import pandas as pd
            from analyzers.price_action_engine import calculate_pa_metrics
            
            # Load 15m data
            nifty_csv = os.path.join(os.path.dirname(__file__), "data", "nifty_15m_2001_to_now.csv")
            vix_csv = os.path.join(os.path.dirname(__file__), "data", "INDIAVIX_15minute_2001_now.csv")
            
            if not os.path.exists(nifty_csv) or not os.path.exists(vix_csv):
                self.append_log("⚠️ Price Action update skipped: 15m data files not found.")
                return
                
            df_n = pd.read_csv(nifty_csv, parse_dates=["date"]).set_index("date")
            df_v = pd.read_csv(vix_csv, parse_dates=["date"]).set_index("date")
            
            pa = calculate_pa_metrics(df_n, df_v)
            if not pa["success"]:
                self.append_log(f"⚠️ Price Action metrics calculation failed: {pa.get('error', 'Unknown error')}")
                return
                
            # 1. Update Correlation
            corr = pa["correlation"]
            corr_color = ACCENT_GREEN if corr < -0.7 else ACCENT_GOLD if corr < -0.4 else ACCENT_RED
            self.pa_corr_card.set_value(f"{corr:.2f}", corr_color)
            
            # 2. Update VWAP
            dev = pa["vwap_dev"]
            dev_color = ACCENT_RED if abs(dev) > 1.2 else ACCENT_CYAN
            self.pa_vwap_card.set_value(f"{dev:+.2f}%", dev_color)
            
            # 2.1 Update OBV
            obv_status = pa.get("obv_status", "NEUTRAL")
            obv_color = ACCENT_GREEN if "ACCUMULATION" in obv_status else ACCENT_RED if "DISTRIBUTION" in obv_status else TEXT_SEC
            self.pa_obv_card.set_value(obv_status.split(" ")[0], obv_color)

            # 3. Update Divergence
            div = pa["divergence"]
            if div and div != "NONE":
                div_color = ACCENT_RED if "BEARISH" in div else ACCENT_GREEN
                self.pa_divergence_card.set_value("⚠️ ACTIVE", div_color)
                self.pa_divergence_card.setToolTip(div)
            else:
                self.pa_divergence_card.set_value("NONE", TEXT_SEC)
            
            # 4. Patterns
            patterns = pa["patterns"]
            if patterns:
                p_text = ""
                for p in patterns:
                    p_text += f"• {p[0]}: {p[2]}\n"
                self.pa_patterns_label.setText(p_text.strip())
            else:
                self.pa_patterns_label.setText("No significant 15m patterns detected.")
                
            # 5. FVGs
            fvgs = pa.get("fvgs", [])
            if fvgs:
                f_text = ""
                for f in fvgs:
                    f_text += f"• {f['type']} ({f['bottom']:.0f}-{f['top']:.0f})\n"
                self.pa_fvg_label.setText(f_text.strip())
            else:
                self.pa_fvg_label.setText("No large FVGs / Imbalances found.")

            # 6. Levels
            lvls = pa["levels"]
            sup_str = ", ".join([f"{s:,.0f}" for s in lvls["support"]])
            res_str = ", ".join([f"{r:,.0f}" for r in lvls["resistance"]])
            self.pa_levels_label.setText(f"Support: {sup_str}\nResistance: {res_str}")
            
            # 6. Alignment Score
            # Check if AI direction matches PA patterns/divergence
            ai_dir = "SIDEWAYS"
            if pred and pred.get("ensemble_prediction"):
                ai_dir = pred.get("ensemble_prediction", {}).get("direction", "SIDEWAYS")
            
            score = 100
            reasons = []
            
            if div:
                if "BEARISH" in div and ai_dir == "UP":
                    score -= 40
                    reasons.append("⚠️ AI is Bulllish but VIX-Price Divergence is Bearish.")
                if "BULLISH" in div and ai_dir == "DOWN":
                    score -= 40
                    reasons.append("⚠️ AI is Bearish but VIX-Price Divergence is Bullish.")
            
            if abs(dev) > 1.0:
                if (dev > 0 and ai_dir == "UP") or (dev < 0 and ai_dir == "DOWN"):
                    score -= 20
                    reasons.append(f"⚠️ Price is overstretched ({dev:+.1f}% from VWAP). Mean reversion risk.")

            if "DISTRIBUTION" in obv_status and ai_dir == "UP":
                score -= 15
                reasons.append("⚠️ Price is rising but Volume Flow (OBV) is falling (Bearish Divergence).")

            if score >= 80:
                align_text = f"<b style='color:{ACCENT_GREEN};'>FULL ALIGNMENT ({score}%)</b><br>ML models and price action are in sync."
                self.pa_alignment_card.setStyleSheet(f"background-color: rgba(80, 255, 171, 0.05); border: 1px solid {ACCENT_GREEN}; border-radius: 8px;")
            elif score >= 50:
                align_text = f"<b style='color:{ACCENT_GOLD};'>MODERATE ALIGNMENT ({score}%)</b><br>" + "<br>".join(reasons)
                self.pa_alignment_card.setStyleSheet(f"background-color: rgba(255, 215, 0, 0.05); border: 1px solid {ACCENT_GOLD}; border-radius: 8px;")
            else:
                align_text = f"<b style='color:{ACCENT_RED};'>CONFLICT DETECTED ({score}%)</b><br>" + "<br>".join(reasons)
                self.pa_alignment_card.setStyleSheet(f"background-color: rgba(255, 75, 75, 0.05); border: 1px solid {ACCENT_RED}; border-radius: 8px;")
            
            self.pa_alignment_text.setText(align_text)
            
        except Exception as e:
            self.append_log(f"⚠️ Price Action update failed: {e}")

    def _update_trade_recommendations_tab(self, pred):
        """Update the Trade Recommendations tab with all advanced features."""
        if not hasattr(self, 'tr_title') or pred is None or not pred.get("success"):
            return
            
        from analyzers.trade_recommender import generate_recommendation
        from analyzers.price_action_engine import calculate_pa_metrics
        from analyzers.options_payoff import compute_payoff
        
        spot = pred.get("spot_price", 24500)
        vix = pred.get("vix_value", 15.0) or 15.0
        pcr = pred.get("pcr", 1.0)
        regime = pred.get("regime", "CHOPPY")
        
        direction = "SIDEWAYS"
        confidence = 50.0
        if pred.get("ensemble_prediction"):
            direction = pred["ensemble_prediction"].get("direction", "SIDEWAYS")
            confidence = pred["ensemble_prediction"].get("confidence", 0.5) * 100
        
        # ── Feature 1 & 5: Whipsaw + Regime Stability ──
        whipsaw_prob = None
        flip_risk = 0
        chop_range = None
        try:
            from analyzers.whipsaw_detector import WhipsawDetector
            import pandas as pd
            nifty_csv = os.path.join(os.path.dirname(__file__), "data", "nifty_daily.csv")
            if os.path.exists(nifty_csv):
                df_ws = pd.read_csv(nifty_csv, parse_dates=["date"])
                # Add minimal features needed by WhipsawDetector
                df_ws["returns_1d"] = df_ws["close"].pct_change()
                df_ws["bb_width"] = df_ws["close"].rolling(20).std() / df_ws["close"].rolling(20).mean()
                df_ws["atr_14"] = (df_ws["high"] - df_ws["low"]).rolling(14).mean()
                df_ws["atr_ratio"] = df_ws["atr_14"] / df_ws["close"]
                df_ws["realized_vol_20"] = df_ws["returns_1d"].rolling(20).std() * (252 ** 0.5)
                df_ws["adx"] = 25  # Placeholder — real ADX needs DI+/DI-
                df_ws["vix"] = vix
                df_ws = df_ws.dropna()
                
                ws = WhipsawDetector()
                ws_result = ws.analyze(df_ws)
                whipsaw_prob = ws_result.get("whipsaw_prob", 0)
                flip_risk = ws_result.get("flip_risk", 0)
                chop_range = ws_result.get("chop_range")
        except Exception as e:
            self.append_log(f"⚠️ Whipsaw analysis failed: {e}")
        
        # Update Whipsaw Meter
        if whipsaw_prob is not None:
            if whipsaw_prob > 55:
                ws_color = ACCENT_RED
                ws_text = f"{whipsaw_prob:.0f}% 🔴"
            elif whipsaw_prob > 30:
                ws_color = ACCENT_GOLD
                ws_text = f"{whipsaw_prob:.0f}% ⚠️"
            else:
                ws_color = ACCENT_GREEN
                ws_text = f"{whipsaw_prob:.0f}% ✅"
            self.tr_whipsaw_card.set_value(ws_text, ws_color)
        
        # Update Regime Stability
        stability = max(0, 100 - flip_risk)
        if stability < 40:
            stab_color = ACCENT_RED
            stab_text = f"{stability:.0f}% ⚠️"
        elif stability < 70:
            stab_color = ACCENT_GOLD
            stab_text = f"{stability:.0f}%"
        else:
            stab_color = ACCENT_GREEN
            stab_text = f"{stability:.0f}% ✅"
        self.tr_regime_card.set_value(stab_text, stab_color)
        
        # ── Generate Recommendation (Features 1,3,4,6 built into trade_recommender.py) ──
        rec = generate_recommendation(spot, vix, pcr, direction, confidence, regime,
                                       whipsaw_prob=whipsaw_prob)
        
        self._last_trade_rec = rec  # Store for Export to War Room
        
        meta = rec.get("meta", {})
        dte = meta.get("days_to_expiry", 5)
        lots = meta.get("lots", 1)
        
        # Update sizing and expiry cards
        is_caution = meta.get("is_caution", False)
        if lots >= 2:
            self.tr_sizing_card.set_value(f"{lots} Lots 🔥", ACCENT_GREEN)
        elif is_caution:
            self.tr_sizing_card.set_value(f"{lots} Lot ⚠️", ACCENT_GOLD)
        else:
            self.tr_sizing_card.set_value(f"{lots} Lot", TEXT_PRIMARY)
        
        expiry_str = meta.get("expiry_str", "")
        self.tr_expiry_card.set_value(f"{dte}d ({expiry_str})", ACCENT_CYAN if dte > 2 else ACCENT_RED)
        
        # Update Greeks Cards (Feature 10)
        greeks = meta.get("greeks", {})
        if greeks:
            d = greeks.get("delta", 0)
            t = greeks.get("theta", 0)
            g = greeks.get("gamma", 0)
            
            self.tr_delta_card.set_value(f"{d:+.0f}", ACCENT_CYAN if abs(d) < 50 else ACCENT_GOLD)
            self.tr_theta_card.set_value(f"₹{t:+.0f}", ACCENT_GREEN if t > 0 else ACCENT_RED)
            self.tr_gamma_card.set_value(f"{g:.3f}", ACCENT_GOLD if g > 0.05 else TEXT_PRIMARY)
        
        # ── Update Strategy Card ──
        self.tr_title.setText(f"💡 STRATEGY: {rec['strategy']}")
        
        reasoning_html = "<b>Why this trade?</b><ul>"
        for r in rec['reasoning']:
            reasoning_html += f"<li>{r}</li>"
        reasoning_html += "</ul>"
        self.tr_reasoning.setText(reasoning_html)
        
        if rec['legs']:
            legs_text = "<b>Recommended Legs:</b><br>"
            for leg in rec['legs']:
                color = ACCENT_RED if leg['action'] == "SELL" else ACCENT_GREEN
                legs_text += f"&nbsp;&nbsp;• <span style='color:{color};'><b>{leg['action']}</b></span> {leg['strike']} {leg['option_type']} @ ₹{leg['premium']:.0f} × {leg.get('lots', 1)} lot(s)<br>"
            self.tr_legs_label.setText(legs_text)
        else:
            self.tr_legs_label.setText("<b>No legs — AI says skip this session.</b>")
        
        ff_html = "<b>🔥 Firefight Triggers:</b><ul>"
        for f_item in rec['firefight']:
            ff_html += f"<li>{f_item}</li>"
        ff_html += "</ul>"
        if whipsaw_prob and whipsaw_prob > 40:
            ff_html += f"<p style='color:{ACCENT_RED};'><b>⚡ Whipsaw Override Active:</b> Strategy was auto-downgraded to Iron Condor due to {whipsaw_prob:.0f}% whipsaw risk.</p>"
        if stability < 40:
            ff_html += f"<p style='color:{ACCENT_GOLD};'><b>⚠️ Regime Instability:</b> Regime may flip in 1-2 days (stability {stability:.0f}%). Consider halving position size.</p>"
        self.tr_firefight.setText(ff_html)
        
        # ── Feature 2: Bounce-Back Estimator ──
        try:
            from analyzers.bounce_analyzer import BounceAnalyzer
            import pandas as pd
            nifty_csv = os.path.join(os.path.dirname(__file__), "data", "nifty_daily.csv")
            if os.path.exists(nifty_csv) and rec['legs']:
                df_bounce = pd.read_csv(nifty_csv, parse_dates=["date"])
                # Find the sold strike (most at-risk price level)
                sold_legs = [l for l in rec['legs'] if l['action'] == 'SELL']
                if sold_legs:
                    target = sold_legs[0]['strike']
                    ba = BounceAnalyzer()
                    bounce_result = ba.analyze(df_bounce, target, days_list=[3, 5, 10])
                    
                    bounce_html = f"<b>📊 If NIFTY drops to {target:,.0f} (your sold strike):</b><br>"
                    for days, data in bounce_result.get("timeframes", {}).items():
                        prob = data["recovery_prob"]
                        avg_d = data["avg_recovery_days"]
                        scenarios = data["scenarios_found"]
                        emoji = "🟢" if prob > 60 else ("🟡" if prob > 40 else "🔴")
                        bounce_html += f"&nbsp;&nbsp;{emoji} <b>{days}-day window:</b> {prob:.0f}% recovery chance (avg {avg_d:.1f} days, {scenarios} historical events)<br>"
                    self.tr_bounce_label.setText(bounce_html)
                else:
                    self.tr_bounce_label.setText("📊 No sold strikes in this recommendation — bounce analysis not applicable.")
            else:
                self.tr_bounce_label.setText("📊 Data unavailable for bounce analysis.")
        except Exception as e:
            self.tr_bounce_label.setText(f"📊 Bounce analysis unavailable: {e}")
        
        # ── Draw Payoff Graph ──
        if rec['legs']:
            payoff = compute_payoff(rec['legs'], spot_price=spot)
            
            self.tr_chart.ax.clear()
            self.tr_chart.ax.set_facecolor(DARK_BG)
            
            self.tr_chart.ax.fill_between(
                payoff["prices"], payoff["payoff"], 0,
                where=payoff["payoff"] >= 0, alpha=0.3, color='#00FF7F'
            )
            self.tr_chart.ax.fill_between(
                payoff["prices"], payoff["payoff"], 0,
                where=payoff["payoff"] < 0, alpha=0.3, color='#FF4B4B'
            )
            
            self.tr_chart.ax.plot(payoff["prices"], payoff["payoff"], color='#00CED1', linewidth=2)
            self.tr_chart.ax.axhline(y=0, color='#555', linewidth=0.8)
            self.tr_chart.ax.axvline(x=spot, color='#FFD700', linewidth=1.5, linestyle='--', label=f'Spot: {spot:,.0f}')
            
            # Mark breakevens
            for be in payoff.get("breakevens", []):
                self.tr_chart.ax.axvline(x=be, color='#FF6B6B', linewidth=1, linestyle=':', alpha=0.7, label=f'BE: {be:,.0f}')
            
            # Mark chop range if available
            if chop_range:
                self.tr_chart.ax.axvspan(chop_range[0], chop_range[1], alpha=0.08, color='#FFD700', label='Expected Chop Zone')
            
            self.tr_chart.ax.set_xlabel('NIFTY Spot Price', color=TEXT_DIM, fontsize=10)
            self.tr_chart.ax.set_ylabel('P&L (₹)', color=TEXT_DIM, fontsize=10)
            self.tr_chart.ax.set_title(f'{rec["strategy"]} — {lots} Lot(s), {dte}d to Expiry', color=TEXT_PRIMARY, fontsize=12)
            self.tr_chart.ax.tick_params(colors=TEXT_DIM)
            self.tr_chart.ax.legend(loc='upper left', fontsize=8, facecolor=DARK_CARD, edgecolor=DARK_BORDER, labelcolor=TEXT_DIM)
            
            self.tr_chart.fig.tight_layout()
            self.tr_chart.draw()
        
        # ── Feature 8: Log Recommendation ──
        try:
            from analyzers.trade_recs_logger import log_recommendation, get_recs_history
            log_recommendation(rec, spot_price=spot, vix=vix, direction=direction, confidence=confidence)
            
            history = get_recs_history(5)
            if history:
                hist_html = ""
                for h in reversed(history):
                    ts = h.get("timestamp", "")
                    strat = h.get("strategy", "—")
                    h_dir = h.get("direction", "—")
                    h_conf = h.get("confidence", 0)
                    hist_html += f"<span style='color:{TEXT_DIM};'>{ts}</span> — <b>{strat}</b> ({h_dir} @ {h_conf:.0f}%)<br>"
                self.tr_history_label.setText(hist_html)
        except Exception:
            pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # SHORTCUTS & STATUS BAR
    # ─────────────────────────────────────────────────────────────────────────

    
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("F5"), self, self.on_fetch_spot)
        QShortcut(QKeySequence("F6"), self, self.on_sync_data)
        QShortcut(QKeySequence("F7"), self, self.on_train_models)
        QShortcut(QKeySequence("F8"), self, self.on_sync_15m)
    
    def _setup_status_bar(self):
        self.statusBar().showMessage("Ready — Press F5 to fetch spot prices")
    
    # ─────────────────────────────────────────────────────────────────────────
    # BUTTON HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _set_buttons_enabled(self, enabled):
        self.btn_fetch.setEnabled(enabled)
        self.btn_sync.setEnabled(enabled)
        self.btn_train.setEnabled(enabled)
        self.btn_sync_15m.setEnabled(enabled)
    
    def _start_worker(self, func, callback, status_msg, *args):
        """Start a background worker thread."""
        if self.worker and self.worker.isRunning():
            self.append_log("⚠️ Another operation is already running!")
            return
        
        self._set_buttons_enabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        self.statusBar().showMessage(status_msg)
        
        self.worker = WorkerThread(func, *args)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished.connect(lambda result: self._on_worker_done(result, callback))
        self.worker.start()
    
    def _on_worker_done(self, result, callback):
        self._set_buttons_enabled(True)
        self.progress.setVisible(False)
        callback(result)
        self.refresh_data_inspector()
    
    # ── BUTTON 1: FETCH SPOT ──
    def on_fetch_spot(self):
        self.append_log("\n" + "═" * 50)
        self._start_worker(
            self._fetch_and_predict,
            self._on_prediction_done,
            "🔴 Fetching spot prices..."
        )
    
    def _fetch_and_predict(self):
        """Fetch spot, then immediately predict."""
        spot = backend.fetch_spot()
        if spot["success"]:
            pred = backend.predict_now(
                spot_price=spot["nifty_price"],
                vix_value=spot["vix_value"]
            )
            pred["_spot"] = spot
            return pred
        else:
            # Try prediction without live spot
            pred = backend.predict_now()
            pred["_spot"] = spot
            return pred
    
    def _on_prediction_done(self, result):
        from datetime import datetime
        self.prediction = result
        
        # Update spot labels
        spot = result.get("_spot", {})
        if result.get("spot_price"):
            self.spot_label.setText(f"NIFTY: {result['spot_price']:,.2f}")
        if result.get("vix_value"):
            self.vix_label.setText(f"VIX: {result['vix_value']:.2f}")
        
        self.last_update_label.setText(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
        
        if result["success"]:
            self._update_dashboard(result)
            self._update_forecast(result)
            self._update_position_manager(result)
            self._update_command_center(result)
            self._update_trade_recommendations_tab(result)
            self._update_price_action_tab(result)
            self._fetch_intraday_only()  # Auto-fetch intraday data on global refresh
            
            # Store prediction state for War Room
            self._last_prediction = result
            self._last_spot = result.get("spot_price", 24500)
            
            # Auto-log daily signal to journal
            try:
                from analyzers.trade_journal import log_daily_signal
                traffic = backend.compute_traffic_light(result)
                strategy = backend.recommend_strategy(result)
                direction = traffic.get("direction", "SIDEWAYS")
                
                log_daily_signal(
                    date_str=datetime.now().strftime("%Y-%m-%d"),
                    signal_color=traffic["color"],
                    strategy=strategy["strategy"],
                    predicted_dir=direction,
                )
                self._refresh_weekly_report()
            except Exception:
                pass  # Don't crash on journal errors
            
            self.statusBar().showMessage("✅ Prediction complete!")
            self.dashboard_placeholder.setVisible(False)
        else:
            self.statusBar().showMessage(f"❌ Prediction failed: {result.get('error', 'Unknown')}")
    
    # ── BUTTON 2: SYNC DATA ──
    def on_sync_data(self):
        self.append_log("\n" + "═" * 50)
        self._start_worker(
            backend.sync_all_data,
            self._on_sync_done,
            "📊 Syncing market data..."
        )
    
    def _on_sync_done(self, result):
        if result["success"]:
            self.statusBar().showMessage("✅ Data sync complete!")
        else:
            self.statusBar().showMessage(f"❌ Sync failed: {result.get('error', 'Unknown')}")
    
    # ── BUTTON 3: TRAIN ──
    def on_train_models(self):
        self.append_log("\n" + "═" * 50)
        self._start_worker(
            backend.train_all_models,
            self._on_train_done,
            "🧠 Training models (this takes ~3 minutes)..."
        )
    
    def _on_train_done(self, result):
        if result["success"]:
            count = len(result.get("models_trained", []))
            self.statusBar().showMessage(f"✅ Training complete! {count} models saved.")
        else:
            self.statusBar().showMessage(f"❌ Training failed: {result.get('error', 'Unknown')}")
    
    # ── BUTTON 4: SYNC 15M DATA ──
    def on_sync_15m(self):
        self.append_log("\n" + "═" * 50)
        self._start_worker(
            backend.sync_15m_data,
            self._on_sync_15m_done,
            "📉 Syncing 15-minute data..."
        )
    
    def _on_sync_15m_done(self, result):
        if result.get("success"):
            nifty_new = result.get("nifty", {}).get("new_rows", 0)
            vix_new = result.get("vix", {}).get("new_rows", 0)
            self.statusBar().showMessage(f"✅ 15m sync complete! NIFTY: +{nifty_new} rows, VIX: +{vix_new} rows")
            # Refresh price action tab
            self._update_price_action_tab(self.prediction)
        else:
            self.statusBar().showMessage(f"❌ 15m sync failed: {result.get('error', 'Unknown')}")
    
    # ── POSITION MANAGER HANDLERS ──
    def on_log_position(self):
        """Log a new position from the UI inputs."""
        from analyzers.drawdown_monitor import DrawdownMonitor
        
        direction_text = self.pos_direction.currentText()
        direction = direction_text # Pass the full exact string to the backend
        
        entry_price = self.pos_entry_price.value()
        entry_date = self.pos_entry_date.date().toString("yyyy-MM-dd")
        expiry_date = self.pos_expiry_date.date().toString("yyyy-MM-dd")
        
        monitor = DrawdownMonitor()
        monitor.log_position(direction, entry_price, entry_date, expiry_date)
        
        self.append_log(f"📌 Position logged: {direction} at ₹{entry_price:,} on {entry_date} (Expiry: {expiry_date})")
        self.signal_verdict.setText(f"📌 Position logged: {direction.split(' ')[0]} spread at ₹{entry_price:,}. Click Fetch Spot to get HOLD/EXIT signal.")
        self.signal_verdict.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 14px; padding: 12px;")
        self.statusBar().showMessage(f"📌 Position logged: {direction} at ₹{entry_price:,}")
    
    def on_close_position(self):
        """Close the active position."""
        from analyzers.drawdown_monitor import DrawdownMonitor
        monitor = DrawdownMonitor()
        
        if not monitor.data.get("active"):
            self.statusBar().showMessage("⚠️ No active position to close.")
            return
        
        current_price = self.prediction.get("spot_price", 0) if self.prediction else 0
        if current_price == 0:
            current_price = monitor.data["active"]["entry_price"]  # neutral close
        
        monitor.close_position(current_price)
        self.append_log(f"❌ Position closed at ₹{current_price:,.2f}")
        self.signal_verdict.setText("Position closed. Log a new one to continue.")
        self.signal_verdict.setStyleSheet(f"color: {TEXT_DIM}; font-size: 14px; padding: 12px;")
        self.card_hold_exit.set_value("—", TEXT_DIM)
        self.statusBar().showMessage("❌ Position closed.")
    
    def on_refresh_position_manager(self):
        """Manually refresh the Position Manager tab."""
        if self.prediction and self.prediction.get("success"):
            self._update_position_manager(self.prediction)
            self.statusBar().showMessage("🔄 Position Manager refreshed.")
        else:
            self.statusBar().showMessage("⚠️ Click Fetch Spot first to get prediction data.")
    
    def _update_position_manager(self, pred):
        """Update all Position Manager panels with latest prediction."""
        from analyzers.drawdown_monitor import DrawdownMonitor
        
        current_price = pred.get("spot_price", 0)
        confidence = 0
        if pred.get("tree_prediction"):
            confidence = pred["tree_prediction"]["confidence"] * 100
        vix = pred.get("vix_value", 15)
        regime = pred.get("regime", "CHOPPY")
        
        # ─── DRAWDOWN STATUS ───
        monitor = DrawdownMonitor()
        dd_status = monitor.get_status(current_price)
        
        if dd_status["has_position"]:
            pos = dd_status["position_info"]
            entry_dir = pos.get("direction", "UP")
            entry_price = pos.get("entry_price", current_price)
            entry_date = pos.get("entry_date", "")
            
            # P&L card
            pnl = dd_status["pnl_pct"]
            pnl_color = ACCENT_GREEN if pnl >= 0 else ACCENT_RED
            self.card_current_pnl.set_value(f"{pnl:+.1f}%", pnl_color)
            
            # Drawdown severity
            sev = dd_status["severity"]
            sev_colors = {"SAFE": ACCENT_GREEN, "CAUTION": ACCENT_GOLD, "DANGER": ACCENT_RED, "EMERGENCY": ACCENT_RED}
            self.card_dd_severity.set_value(f"{sev}", sev_colors.get(sev, TEXT_DIM))
            self.card_dd_drawdown.set_value(f"{dd_status['drawdown_pct']:.1f}%", ACCENT_RED if dd_status['drawdown_pct'] > 1 else TEXT_PRIMARY)
            self.card_dd_streak.set_value(str(dd_status["consecutive_losses"]), ACCENT_RED if dd_status["consecutive_losses"] >= 2 else TEXT_PRIMARY)
            self.dd_message.setText(dd_status.get("message", ""))
            
            # ─── HOLD/EXIT SIGNAL ───
            try:
                signal = backend.get_hold_exit_signal(entry_dir, entry_price, entry_date, prediction=pred)
                
                sig = signal.get("signal", "HOLD")
                sig_colors = {"HOLD": ACCENT_GREEN, "HEDGE": ACCENT_GOLD, "EXIT": ACCENT_RED}
                self.card_hold_exit.set_value(sig, sig_colors.get(sig, TEXT_DIM))
                self.card_days_held.set_value(str(signal.get("days_held", 0)))
                
                self.signal_verdict.setText(signal.get("message", ""))
                self.signal_verdict.setStyleSheet(f"color: {sig_colors.get(sig, TEXT_DIM)}; font-size: 14px; padding: 12px; font-weight: bold;")
                
                # ─── RECOVERY PROBABILITY ───
                recovery = signal.get("recovery_prob")
                if recovery:
                    self.card_recovery_pct.set_value(f"{recovery.get('recovery_prob', 0):.0f}%", ACCENT_GREEN if recovery.get('recovery_prob', 0) >= 60 else ACCENT_RED)
                    self.card_recovery_days.set_value(f"{recovery.get('avg_recovery_days', 0):.0f}d")
                    self.card_scenarios.set_value(str(recovery.get('similar_scenarios', 0)))
                    self.recovery_message.setText(recovery.get('message', ''))
            except Exception as e:
                self.signal_verdict.setText(f"Error computing signal: {e}")
        else:
            self.signal_verdict.setText("No active position. Log one above to get HOLD/EXIT signals.")
            self.signal_verdict.setStyleSheet(f"color: {TEXT_DIM}; font-size: 14px; padding: 12px;")
        
        # ─── EXPIRY ADVISOR (always update) ───
        try:
            expiry = backend.get_optimal_expiry(confidence, vix, regime)
            self.expiry_recommendation.setText(expiry.get("recommendation", ""))
            reasoning_text = "\n".join([f"• {r}" for r in expiry.get("reasoning", [])])
            self.expiry_reasoning.setText(reasoning_text)
        except Exception:
            pass
        
        # ─── 15-MIN INTRADAY REGIME ───
        try:
            regime_15m = backend.get_intraday_regime()
            
            intra_regime = regime_15m.get("intraday_regime", "UNKNOWN")
            regime_colors = {"TRENDING_UP": ACCENT_GREEN, "TRENDING_DOWN": ACCENT_RED, "CHOPPY": ACCENT_GOLD, "REVERSAL_RISK": ACCENT_RED}
            self.card_intraday_regime.set_value(intra_regime, regime_colors.get(intra_regime, TEXT_DIM))
            
            eq = regime_15m.get("entry_quality", 50)
            eq_color = ACCENT_GREEN if eq >= 60 else ACCENT_GOLD if eq >= 40 else ACCENT_RED
            self.card_entry_quality.set_value(f"{eq}/100", eq_color)
            
            self.card_rsi_15m.set_value(f"{regime_15m.get('rsi', 50):.0f}")
            self.card_vix_15m.set_value(regime_15m.get("vix_trend", "N/A")[:20])
        except Exception:
            pass
    # ─────────────────────────────────────────────────────────────────────────
    # DASHBOARD UPDATE
    # ─────────────────────────────────────────────────────────────────────────
    
    def _update_dashboard(self, pred):
        """Populate dashboard cards from prediction results."""
        
        # Verdict
        if pred.get("tree_prediction"):
            tp = pred["tree_prediction"]
            direction = tp.get("direction", "—")
            confidence = tp.get("confidence", 0) * 100
            
            if "UP" in str(direction).upper():
                color = ACCENT_GREEN
            elif "DOWN" in str(direction).upper():
                color = ACCENT_RED
            else:
                color = ACCENT_GOLD
            
            self.card_verdict.set_value(direction, color)
            self.card_confidence.set_value(f"{confidence:.0f}%", color)
        
        # Regime
        regime = pred.get("regime", "—")
        self.card_regime.set_value(regime, ACCENT_CYAN)
        
        # Whipsaw
        if pred.get("whipsaw"):
            ws = pred["whipsaw"]
            is_choppy = ws.get("is_choppy", False)
            prob = ws.get("whipsaw_prob", 0)  # already a percentage (0-100)
            chop_range = ws.get("chop_range", (0, 0))
            
            if is_choppy:
                status = f"⚠️ CHOPPY ({prob:.0f}%)"
                self.card_whipsaw.setToolTip(f"Expected Chop Range: {chop_range[0]:.0f} to {chop_range[1]:.0f}")
            else:
                status = f"✅ CLEAN ({prob:.0f}%)"
                self.card_whipsaw.setToolTip(f"Trend is clean. Low whipsaw probability.")
                
            color = ACCENT_GOLD if is_choppy else ACCENT_GREEN
            self.card_whipsaw.set_value(status, color)
        
        if pred.get("tree_prediction"):
            tp = pred["tree_prediction"]
            self.card_tree.set_value(
                f"{tp['direction']} ({tp['confidence']*100:.0f}%)",
                ACCENT_GREEN if "UP" in str(tp['direction']).upper() else ACCENT_RED
            )
        
        # Pull latest 15-minute prediction if available
        # It's an independent ML model predicting the immediate next 15-240 minutes
        df_15m = getattr(self, '_last_15m_df', None)
        features_15m = getattr(self, '_last_15m_features', None)
        if df_15m is not None and features_15m is not None:
            try:
                from models.intraday_classifier import IntradayClassifier
                model = IntradayClassifier()
                if model.load():
                    intra_pred = model.predict_today(df_15m, features_15m)
                    if intra_pred and intra_pred.get("success"):
                        i_dir = intra_pred.get("direction", "—")
                        i_conf = intra_pred.get("confidence", 0) * 100
                        i_color = ACCENT_GREEN if "UP" in i_dir else ACCENT_RED if "DOWN" in i_dir else ACCENT_GOLD
                        self.card_intraday_dash.set_value(f"{i_dir} ({i_conf:.0f}%)", i_color)
                    else:
                        self.card_intraday_dash.set_value("—")
                else:
                    self.card_intraday_dash.set_value("Model Missing", TEXT_DIM)
            except Exception as e:
                self.card_intraday_dash.set_value("Error", TEXT_DIM)
        else:
            self.card_intraday_dash.set_value("Sync 15m Data", TEXT_DIM)
        
        # Sentiment
        pcr = pred.get("pcr", 1.0)
        pcr_color = ACCENT_GREEN if pcr < 0.8 else ACCENT_RED if pcr > 1.2 else ACCENT_GOLD
        self.card_pcr.set_value(f"{pcr:.2f}", pcr_color)
        
        fii = pred.get("fii_net", 0)
        self.card_fii.set_value(f"{fii:,.0f}", ACCENT_GREEN if fii > 0 else ACCENT_RED)
        
        dii = pred.get("dii_net", 0)
        self.card_dii.set_value(f"{dii:,.0f}", ACCENT_GREEN if dii > 0 else ACCENT_RED)
        
        # Support & Resistance
        spot = pred.get("spot_price", 0)
        
        if pred.get("resistances"):
            for i, tile in enumerate(self.res_tiles):
                if i < len(pred["resistances"]):
                    r = pred["resistances"][i]
                    dist = ((r['price'] - spot) / spot) * 100
                    tile.lbl_level.setText(f"{r['price']:,.0f}")
                    tile.lbl_level.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 18px; font-weight: bold; border: none; background: transparent;")
                    tile.lbl_stats.setText(f"Dist: +{dist:.1f}%  |  Str: {r['strength']:.1f}")
                else:
                    tile.lbl_level.setText("—")
                    tile.lbl_stats.setText("Dist: —  |  Str: —")
                    tile.lbl_level.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: bold; border: none; background: transparent;")
        
        if pred.get("supports"):
            for i, tile in enumerate(self.support_tiles):
                if i < len(pred["supports"]):
                    s = pred["supports"][i]
                    dist = ((spot - s['price']) / spot) * 100
                    tile.lbl_level.setText(f"{s['price']:,.0f}")
                    tile.lbl_level.setStyleSheet(f"color: {ACCENT_RED}; font-size: 18px; font-weight: bold; border: none; background: transparent;")
                    tile.lbl_stats.setText(f"Dist: -{dist:.1f}%  |  Str: {s['strength']:.1f}")
                else:
                    tile.lbl_level.setText("—")
                    tile.lbl_stats.setText("Dist: —  |  Str: —")
                    tile.lbl_level.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: bold; border: none; background: transparent;")
    
    # ─────────────────────────────────────────────────────────────────────────
    # FORECAST UPDATE
    # ─────────────────────────────────────────────────────────────────────────
    
    def _update_forecast(self, pred):
        """Update forecast tab with range predictions and chart."""
        ranges = pred.get("ranges")
        spot = pred.get("spot_price", 0)
        
        if not ranges:
            return
        
        # 7-day cards
        if 7 in ranges:
            r = ranges[7]
            self.card_7d_low.set_value(f"{r['p10']:,.0f}", ACCENT_RED)
            self.card_7d_mid.set_value(f"{r['p50']:,.0f}", ACCENT_CYAN)
            self.card_7d_high.set_value(f"{r['p90']:,.0f}", ACCENT_GREEN)
        
        # 30-day cards
        if 30 in ranges:
            r = ranges[30]
            self.card_30d_low.set_value(f"{r['p10']:,.0f}", ACCENT_RED)
            self.card_30d_mid.set_value(f"{r['p50']:,.0f}", ACCENT_CYAN)
            self.card_30d_high.set_value(f"{r['p90']:,.0f}", ACCENT_GREEN)
        
        # Draw probability cone chart
        if 7 in ranges:
            r = ranges[7]
            ax = self.forecast_chart.ax
            ax.clear()
            ax.set_facecolor(DARK_BG)
            
            x = [0, 7]
            # 80% confidence band
            ax.fill_between(x, [spot, r['p10']], [spot, r['p90']],
                          alpha=0.15, color='cyan', label='80% Confidence')
            # 50% confidence band
            ax.fill_between(x, [spot, r['p25']], [spot, r['p75']],
                          alpha=0.25, color='cyan', label='50% Confidence')
            # Median path
            ax.plot(x, [spot, r['p50']], 'c-', linewidth=2, label='Median')
            # Current price
            ax.axhline(y=spot, color='white', linestyle='--', alpha=0.5, label='Current')
            
            ax.set_xlabel('Days', color=TEXT_DIM, fontsize=10)
            ax.set_ylabel('Nifty Level', color=TEXT_DIM, fontsize=10)
            ax.set_title('7-Day Probability Cone', color=TEXT_PRIMARY, fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8, facecolor=DARK_CARD, edgecolor=DARK_BORDER, labelcolor=TEXT_DIM)
            ax.tick_params(colors=TEXT_DIM)
            
            self.forecast_chart.fig.tight_layout()
            self.forecast_chart.draw()
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY LAB HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_analyze_strike(self):
        if not self.prediction or not self.prediction.get("df_raw") is not None:
            self.append_log("⚠️ Run Fetch Spot first to load data!")
            return
        
        strike = self.strike_input.value()
        days = self.days_input.value()
        
        try:
            res = backend.analyze_strike(self.prediction["df_raw"], strike, days)
            self.card_touch_prob.set_value(f"{res['touch_prob']:.1f}%",
                ACCENT_RED if res['touch_prob'] > 60 else ACCENT_GOLD if res['touch_prob'] > 35 else ACCENT_GREEN)
            self.card_recovery.set_value(f"{res['recovery_prob']:.1f}%", ACCENT_GREEN)
            self.card_firefight.set_value(f"{res['firefight_level']:,.0f}", ACCENT_GOLD)
            
            if res['touch_prob'] > 60:
                self.condor_verdict.setText("🚨 HIGH RISK! High probability of touching this strike.")
                self.condor_verdict.setStyleSheet(f"color: {ACCENT_RED}; font-size: 14px; font-weight: bold;")
            elif res['touch_prob'] > 35:
                self.condor_verdict.setText("⚠️ MODERATE RISK. Keep monitoring.")
                self.condor_verdict.setStyleSheet(f"color: {ACCENT_GOLD}; font-size: 14px; font-weight: bold;")
            else:
                self.condor_verdict.setText("✅ SAFE ZONE. Low touch probability.")
                self.condor_verdict.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 14px; font-weight: bold;")
        except Exception as e:
            self.append_log(f"❌ Strike analysis failed: {e}")
    
    def on_check_bounce(self):
        if not self.prediction or self.prediction.get("df_raw") is None:
            self.append_log("⚠️ Run Fetch Spot first to load data!")
            return
        
        target = self.target_input.value()
        try:
            res = backend.analyze_bounce(self.prediction["df_raw"], target)
            timeframes = res.get("timeframes", {})
            
            self.bounce_table.setRowCount(len(timeframes))
            for i, (d, vals) in enumerate(timeframes.items()):
                self.bounce_table.setItem(i, 0, QTableWidgetItem(str(d)))
                self.bounce_table.setItem(i, 1, QTableWidgetItem(f"{vals['recovery_prob']:.1f}%"))
                self.bounce_table.setItem(i, 2, QTableWidgetItem(f"{vals['avg_recovery_days']:.1f}"))
        except Exception as e:
            self.append_log(f"❌ Bounce analysis failed: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA INSPECTOR
    # ─────────────────────────────────────────────────────────────────────────
    
    def refresh_data_inspector(self):
        """Refresh all data status tables."""
        try:
            status = backend.get_data_status()
        except Exception:
            return
        
        # CSV table (full inspector tab)
        csv_data = status.get("csv", [])
        self.csv_table.setRowCount(len(csv_data))
        for i, s in enumerate(csv_data):
            self.csv_table.setItem(i, 0, QTableWidgetItem(s["name"]))
            self.csv_table.setItem(i, 1, QTableWidgetItem(str(s["rows"])))
            self.csv_table.setItem(i, 2, QTableWidgetItem(s["latest_date"]))
            self.csv_table.setItem(i, 3, QTableWidgetItem(s["file_size"]))
            
            status_text = "✅ Fresh" if s["exists"] and not s["is_stale"] else "⚠️ Stale" if s["exists"] else "❌ Missing"
            item = QTableWidgetItem(status_text)
            if s["is_stale"]:
                item.setForeground(QColor(ACCENT_GOLD))
            elif not s["exists"]:
                item.setForeground(QColor(ACCENT_RED))
            else:
                item.setForeground(QColor(ACCENT_GREEN))
            self.csv_table.setItem(i, 4, item)
        
        # Mini inspector (left panel)
        self.mini_inspector.setRowCount(len(csv_data))
        for i, s in enumerate(csv_data):
            name_item = QTableWidgetItem(s["name"])
            date_item = QTableWidgetItem(s["latest_date"])
            
            if s["is_stale"]:
                date_item.setForeground(QColor(ACCENT_GOLD))
            elif not s["exists"]:
                date_item.setForeground(QColor(ACCENT_RED))
            else:
                date_item.setForeground(QColor(ACCENT_GREEN))
            
            self.mini_inspector.setItem(i, 0, name_item)
            self.mini_inspector.setItem(i, 1, date_item)
        
        # Model table
        model_data = status.get("models", [])
        self.model_table.setRowCount(len(model_data))
        all_models_ok = True
        for i, m in enumerate(model_data):
            self.model_table.setItem(i, 0, QTableWidgetItem(m["name"]))
            self.model_table.setItem(i, 1, QTableWidgetItem(m["file_size"]))
            self.model_table.setItem(i, 2, QTableWidgetItem(m["modified"]))
            
            status_text = "✅ Ready" if m["exists"] else "❌ Missing"
            item = QTableWidgetItem(status_text)
            item.setForeground(QColor(ACCENT_GREEN if m["exists"] else ACCENT_RED))
            self.model_table.setItem(i, 3, item)
            
            if not m["exists"]:
                all_models_ok = False
        
        # Update model status label
        if all_models_ok:
            self.model_status_label.setText("🟢 Models: All Ready")
            self.model_status_label.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 11px; border: none;")
        else:
            self.model_status_label.setText("🔴 Models: Some Missing")
            self.model_status_label.setStyleSheet(f"color: {ACCENT_RED}; font-size: 11px; border: none;")
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOG CONSOLE
    # ─────────────────────────────────────────────────────────────────────────
    
    def append_log(self, msg):
        """Append a message to the log console (thread-safe)."""
        self.log_console.append(msg)
        # Auto-scroll to bottom
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(DARK_BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(DARK_CARD))
    palette.setColor(QPalette.AlternateBase, QColor(DARK_BG))
    palette.setColor(QPalette.ToolTipBase, QColor(DARK_CARD))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(DARK_CARD))
    palette.setColor(QPalette.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.BrightText, QColor(ACCENT_RED))
    palette.setColor(QPalette.Highlight, QColor(ACCENT_CYAN))
    palette.setColor(QPalette.HighlightedText, QColor(DARK_BG))
    app.setPalette(palette)
    
    app.setStyleSheet(GLOBAL_STYLE)
    
    window = DavidOracleWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
