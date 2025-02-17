#!/usr/bin/env python
import sys
import os
import torch
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTextEdit, QFormLayout
)
from PyQt5.QtCore import QThread, pyqtSignal
from torch.utils.data import DataLoader
from train import create_run_folder, get_regression_model, train_and_validate, test_model
from data import PipetteDataModule

class TrainingWorker(QThread):
    update_signal = pyqtSignal(int, float, float, float, float)
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, model_name, train_images_dir, annotations_csv,
                 device, batch_size, learning_rate, num_epochs, threshold,
                 checkpoint_folder=None):
        super().__init__()
        self.model_name = model_name
        self.train_images_dir = train_images_dir
        self.annotations_csv = annotations_csv
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.checkpoint_folder = checkpoint_folder

    def run(self):
        self.log_signal.emit("Setting up data module...")
        data_module = PipetteDataModule(
            self.train_images_dir,
            self.annotations_csv,
            train_split=0.7, val_split=0.2, test_split=0.1, seed=42
        )
        train_dataset, val_dataset, test_dataset = data_module.setup()
        self.log_signal.emit("Datasets prepared.")

        # Build config dictionary and create run folder (which saves config.txt)
        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "threshold": self.threshold,
            "device": self.device
        }
        run_folder = create_run_folder(self.model_name, config=config)
        self.log_signal.emit(f"Run folder created at: {run_folder}")

        device = torch.device(self.device)
        model = get_regression_model(model_name=self.model_name, pretrained=True, output_dim=1)
        model.to(device)
        self.log_signal.emit("Model created and moved to device.")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        def update_callback(epoch, train_loss, val_loss, mae, r2):
            self.update_signal.emit(epoch, train_loss, val_loss, mae, r2)
            self.log_signal.emit(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
            )

        self.log_signal.emit("Starting training...")
        best_checkpoint, epochs_list, train_losses, val_losses, mae_scores, r2_scores = train_and_validate(
            model, train_loader, val_loader, device, run_folder,
            num_epochs=self.num_epochs, update=True, update_callback=update_callback
        )

        self.log_signal.emit("Training complete. Testing the best model...")
        model.load_state_dict(torch.load(best_checkpoint))
        criterion = torch.nn.MSELoss()
        test_results = test_model(model, test_loader, device, criterion, run_folder)
        self.log_signal.emit(
            f"Final Test Results: Loss={test_results['Test Loss']:.4f}, "
            f"MAE={test_results['Test MAE']:.4f}, R²={test_results['Test R²']:.4f}"
        )
        self.finished_signal.emit()

class TrainingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defocus Regression Training GUI")
        self.resize(1000, 700)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        graphs_layout = QHBoxLayout()
        self.loss_plot = pg.PlotWidget(title="Loss")
        self.mae_plot = pg.PlotWidget(title="MAE")
        self.r2_plot = pg.PlotWidget(title="R²")
        graphs_layout.addWidget(self.loss_plot)
        graphs_layout.addWidget(self.mae_plot)
        graphs_layout.addWidget(self.r2_plot)
        main_layout.addLayout(graphs_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        controls_form = QFormLayout()
        self.data_dir_line = QLineEdit()
        browse_data_btn = QPushButton("Browse")
        browse_data_btn.clicked.connect(self.select_data_dir)
        data_hbox = QHBoxLayout()
        data_hbox.addWidget(self.data_dir_line)
        data_hbox.addWidget(browse_data_btn)
        controls_form.addRow("Images Dir:", data_hbox)

        self.annotations_line = QLineEdit()
        browse_ann_btn = QPushButton("Browse")
        browse_ann_btn.clicked.connect(self.select_annotations)
        ann_hbox = QHBoxLayout()
        ann_hbox.addWidget(self.annotations_line)
        ann_hbox.addWidget(browse_ann_btn)
        controls_form.addRow("Annotations CSV:", ann_hbox)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["mobilenetv3_large_100"])
        controls_form.addRow("Model:", self.model_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        controls_form.addRow("Epochs:", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(32)
        controls_form.addRow("Batch Size:", self.batch_size_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setRange(1e-6, 1.0)
        self.learning_rate_spin.setValue(1e-4)
        controls_form.addRow("Learning Rate:", self.learning_rate_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setRange(0.0, 10.0)
        self.threshold_spin.setValue(0.3)
        controls_form.addRow("Threshold:", self.threshold_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        controls_form.addRow("Device:", self.device_combo)

        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        controls_form.addRow(self.start_button)

        main_layout.addLayout(controls_form)

        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.mae_scores = []
        self.r2_scores = []
        self.worker = None

    def select_data_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Training Images Directory", "")
        if directory:
            self.data_dir_line.setText(directory)

    def select_annotations(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Annotations CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.annotations_line.setText(file_path)

    def start_training(self):
        train_dir = self.data_dir_line.text()
        annotations = self.annotations_line.text()
        if not os.path.isdir(train_dir):
            QMessageBox.warning(self, "Input Error", "Please select a valid training images directory.")
            return
        if not os.path.isfile(annotations):
            QMessageBox.warning(self, "Input Error", "Please select a valid annotations CSV file.")
            return

        self.start_button.setEnabled(False)
        self.append_log("Training started...")
        model_name = self.model_combo.currentText()
        num_epochs = self.epochs_spin.value()
        batch_size = self.batch_size_spin.value()
        learning_rate = self.learning_rate_spin.value()
        threshold = self.threshold_spin.value()
        device = self.device_combo.currentText()

        checkpoint_folder = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoint_folder, exist_ok=True)

        self.worker = TrainingWorker(
            model_name, train_dir, annotations, device,
            batch_size, learning_rate, num_epochs, threshold,
            checkpoint_folder
        )
        self.worker.update_signal.connect(self.on_update)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.log_signal.connect(self.append_log)
        self.worker.start()

    def on_update(self, epoch, train_loss, val_loss, mae, r2):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.mae_scores.append(mae)
        self.r2_scores.append(r2)
        self.loss_plot.clear()
        self.loss_plot.plot(self.epochs, self.train_losses, pen='c', name="Train Loss")
        self.loss_plot.plot(self.epochs, self.val_losses, pen='m', name="Val Loss")
        self.mae_plot.clear()
        self.mae_plot.plot(self.epochs, self.mae_scores, pen=pg.mkPen('orange', width=2))
        self.r2_plot.clear()
        self.r2_plot.plot(self.epochs, self.r2_scores, pen=pg.mkPen('lime', width=2))

    def on_finished(self):
        self.append_log("Training finished.")
        self.start_button.setEnabled(True)

    def append_log(self, message):
        self.log_text.append(message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrainingGUI()
    window.show()
    sys.exit(app.exec_())
