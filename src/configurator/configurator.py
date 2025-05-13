# src/configurator/configurator.py
import sys
import os
import json
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSpinBox, QCheckBox, QPushButton, QHBoxLayout

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


class Configurator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Konfigurator")
        self.setFixedSize(300, 300)

        self.layout = QVBoxLayout()

        self.layout.addWidget(QLabel("Kadrowanie (x, y, szer, wys):"))
        self.crop_x = QSpinBox()
        self.crop_y = QSpinBox()
        self.crop_w = QSpinBox()
        self.crop_h = QSpinBox()
        for box in [self.crop_x, self.crop_y, self.crop_w, self.crop_h]:
            box.setRange(0, 1920)
        row = QHBoxLayout()
        for widget in [self.crop_x, self.crop_y, self.crop_w, self.crop_h]:
            row.addWidget(widget)
        self.layout.addLayout(row)

        self.layout.addWidget(QLabel("Confidence threshold:"))
        self.confidence = QSpinBox()
        self.confidence.setRange(0, 100)
        self.confidence.setValue(60)
        self.layout.addWidget(self.confidence)

        self.rect_cb = QCheckBox("Ramka")
        self.alert_cb = QCheckBox("Alert na ekranie")
        self.sound_cb = QCheckBox("Dźwięk")
        self.layout.addWidget(self.rect_cb)
        self.layout.addWidget(self.alert_cb)
        self.layout.addWidget(self.sound_cb)

        self.save_btn = QPushButton("Zapisz")
        self.save_btn.clicked.connect(self.save_config)
        self.layout.addWidget(self.save_btn)

        self.setLayout(self.layout)

    def save_config(self):
        config = {
            "frame_crop": {
                "x": self.crop_x.value(),
                "y": self.crop_y.value(),
                "width": self.crop_w.value(),
                "height": self.crop_h.value()
            },
            "detection": {
                "confidence_threshold": self.confidence.value() / 100.0
            },
            "alert": {
                "show_rectangle": self.rect_cb.isChecked(),
                "show_alert": self.alert_cb.isChecked(),
                "play_sound": self.sound_cb.isChecked()
            }
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        self.close()

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = Configurator()
    window.show()
    sys.exit(app.exec())