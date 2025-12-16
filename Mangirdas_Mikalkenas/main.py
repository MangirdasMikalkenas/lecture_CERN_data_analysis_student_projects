from gui import (get_path, init_music_player, QApplication,
                create_main_window)



music_f = "cyberpunk.mp3"
logo_f = "logo.jpg"
BORDER_COLOR = "#1B1C1D"  # Dark grey
# ---- GUI ENTRY (no function wrapper) ----

music_path = get_path(music_f) # Resolve full path
init_music_player(music_path) # Start background music

# ---- CREATE PYQT APPLICATION ----
app = QApplication([])

# Global white text style
app.setStyleSheet("""
    QWidget {
        color: white;
        background-color: black;
    }
    QLineEdit, QComboBox, QSpinBox, QDateEdit {
        color: white;
        background-color: #1B1C1D;
        border: 1px solid #2A2B2C;
    }
    QCheckBox, QRadioButton {
        color: white;
    }
    QScrollArea {
        background-color: black;
        border: none;
    }
""")

window = create_main_window()# Build main GUI from gui.py
window.show()   # Show GUI window

app.exec() # Blocks until GUI closes
