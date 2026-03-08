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

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QTextEdit, QTableWidget,
    QTableWidgetItem, QSplitter, QFrame, QGroupBox, QGridLayout,
    QSpinBox, QSlider, QHeaderView, QSizePolicy, QSystemTrayIcon,
    QMenu, QAction, QStatusBar, QProgressBar, QShortcut, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSize
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

DARK_BG = "#0E1117"
DARK_CARD = "#1A1D29"
DARK_BORDER = "#2D3139"
ACCENT_GREEN = "#00FF7F"
ACCENT_RED = "#FF4B4B"
ACCENT_GOLD = "#FFD700"
ACCENT_CYAN = "#00CED1"
TEXT_PRIMARY = "#FAFAFA"
TEXT_DIM = "#8B8D97"

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
        border: 1px solid {DARK_BORDER};
        border-radius: 8px;
        margin-top: 10px;
        padding-top: 15px;
        font-size: 14px;
        font-weight: bold;
        color: {ACCENT_CYAN};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
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
                background-color: {DARK_CARD};
                border: 1px solid {DARK_BORDER};
                border-radius: 10px;
                padding: 12px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
    
    def set_value(self, value, color=None):
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")


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
        self.setWindowTitle("🦅 David Oracle v2.0 — Desktop")
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
        self.tab_inspector = self._build_inspector_tab()
        
        self.tabs.addTab(self.tab_dashboard, "🦅 Dashboard")
        self.tabs.addTab(self.tab_forecast, "📈 Forecast")
        self.tabs.addTab(self.tab_strategy, "🧪 Strategy Lab")
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
        
        version = QLabel("v2.0 — Desktop Edition")
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
    
    def _build_dashboard_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
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
        
        # Row 2: Tree | LSTM | Ensemble predictions
        row2 = QHBoxLayout()
        
        self.card_tree = MetricCard("🌳 TREE MODEL", "—", TEXT_PRIMARY)
        self.card_lstm = MetricCard("🧠 LSTM", "—", TEXT_PRIMARY)
        self.card_ensemble = MetricCard("📊 ENSEMBLE", "—", TEXT_PRIMARY)
        
        row2.addWidget(self.card_tree)
        row2.addWidget(self.card_lstm)
        row2.addWidget(self.card_ensemble)
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
        
        # Resistance table
        self.resistance_table = QTableWidget(3, 3)
        self.resistance_table.setHorizontalHeaderLabels(["Level", "Distance", "Strength"])
        self.resistance_table.horizontalHeader().setStretchLastSection(True)
        self.resistance_table.verticalHeader().setVisible(False)
        self.resistance_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.resistance_table.setMaximumHeight(120)
        
        # Support table
        self.support_table = QTableWidget(3, 3)
        self.support_table.setHorizontalHeaderLabels(["Level", "Distance", "Strength"])
        self.support_table.horizontalHeader().setStretchLastSection(True)
        self.support_table.verticalHeader().setVisible(False)
        self.support_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.support_table.setMaximumHeight(120)
        
        res_group = QGroupBox("Resistance (Above)")
        res_layout = QVBoxLayout(res_group)
        res_layout.addWidget(self.resistance_table)
        
        sup_group = QGroupBox("Support (Below)")
        sup_layout = QVBoxLayout(sup_group)
        sup_layout.addWidget(self.support_table)
        
        row4.addWidget(res_group)
        row4.addWidget(sup_group)
        layout.addLayout(row4)
        
        layout.addStretch()
        
        # Placeholder message
        self.dashboard_placeholder = QLabel("Press 🔴 Fetch Spot + Predict (F5) to see the full dashboard")
        self.dashboard_placeholder.setStyleSheet(f"color: {TEXT_DIM}; font-size: 14px;")
        self.dashboard_placeholder.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.dashboard_placeholder)
        
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
    # TAB 5: DAVID CODEX — A-to-Z Trading Guide
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
<tr><td style='color:{ACCENT_GOLD}'>🧠 Train (F7)</td><td>Trains 5 ML models (XGBoost, LightGBM, CatBoost, HMM, LSTM) from CSVs → saves .pkl</td></tr>
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

<p style='color:{TEXT_DIM}'><i>Tip: If Verdict says SIDEWAYS but Forecast tilts UP — it's NOT a bug. They come from different models. SIDEWAYS at low confidence = Iron Condor.</i></p>
""")

        # ── CHAPTER 3: Daily Checklist ──
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

        # ── CHAPTER 5: Risk Management ──
        add_section("🛡️ Chapter 5: Risk Management & Capital Allocation", f"""
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

        # ── CHAPTER 6: Exit Rules ──
        add_section("⏱️ Chapter 6: When to Exit — The Exit Playbook", f"""
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

        # ── CHAPTER 7: When NOT to Trade ──
        add_section("🚫 Chapter 7: When NOT to Trade — Red Light Days", f"""
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
    # SHORTCUTS & STATUS BAR
    # ─────────────────────────────────────────────────────────────────────────
    
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("F5"), self, self.on_fetch_spot)
        QShortcut(QKeySequence("F6"), self, self.on_sync_data)
        QShortcut(QKeySequence("F7"), self, self.on_train_models)
    
    def _setup_status_bar(self):
        self.statusBar().showMessage("Ready — Press F5 to fetch spot prices")
    
    # ─────────────────────────────────────────────────────────────────────────
    # BUTTON HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _set_buttons_enabled(self, enabled):
        self.btn_fetch.setEnabled(enabled)
        self.btn_sync.setEnabled(enabled)
        self.btn_train.setEnabled(enabled)
    
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
        
        # Tree / LSTM / Ensemble
        if pred.get("tree_prediction"):
            tp = pred["tree_prediction"]
            self.card_tree.set_value(
                f"{tp['direction']} ({tp['confidence']*100:.0f}%)",
                ACCENT_GREEN if "UP" in str(tp['direction']).upper() else ACCENT_RED
            )
        
        if pred.get("lstm_prediction"):
            lp = pred["lstm_prediction"]
            self.card_lstm.set_value(
                f"{lp['direction']} ({lp['confidence']*100:.0f}%)",
                ACCENT_GREEN if "UP" in str(lp['direction']).upper() else ACCENT_RED
            )
        else:
            self.card_lstm.set_value("N/A", TEXT_DIM)
        
        if pred.get("ensemble_prediction"):
            ep = pred["ensemble_prediction"]
            self.card_ensemble.set_value(
                f"{ep['direction']} ({ep['confidence']*100:.0f}%)",
                ACCENT_GREEN if "UP" in str(ep['direction']).upper() else ACCENT_RED
            )
        
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
            self.resistance_table.setRowCount(min(3, len(pred["resistances"])))
            for i, r in enumerate(pred["resistances"][:3]):
                dist = ((r['price'] - spot) / spot) * 100
                self.resistance_table.setItem(i, 0, QTableWidgetItem(f"{r['price']:,.0f}"))
                self.resistance_table.setItem(i, 1, QTableWidgetItem(f"+{dist:.1f}%"))
                self.resistance_table.setItem(i, 2, QTableWidgetItem(f"{r['strength']:.1f}"))
        
        if pred.get("supports"):
            self.support_table.setRowCount(min(3, len(pred["supports"])))
            for i, s in enumerate(pred["supports"][:3]):
                dist = ((spot - s['price']) / spot) * 100
                self.support_table.setItem(i, 0, QTableWidgetItem(f"{s['price']:,.0f}"))
                self.support_table.setItem(i, 1, QTableWidgetItem(f"-{dist:.1f}%"))
                self.support_table.setItem(i, 2, QTableWidgetItem(f"{s['strength']:.1f}"))
    
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
