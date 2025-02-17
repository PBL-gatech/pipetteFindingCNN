#!/usr/bin/env python
import sys
import os
import torch
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTextEdit, QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal

# Import your training modules (make sure these are in your PYTHONPATH)
from train import Trainer
from data import PipetteDataModule

# =============================================================================
# Worker thread to run training without blocking the GUI
# =============================================================================
class TrainingWorker(QThread):
    # Now emitting: epoch, train_loss, val_loss, accuracy, mae, r2
    update_signal = pyqtSignal(int, float, float, float, float, float)
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

        trainer = Trainer(
            model_name=self.model_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=self.device,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            threshold=self.threshold
        )

        scaler = torch.amp.GradScaler()
        train_loader, val_loader = trainer._get_dataloaders()
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.log_signal.emit(f"Epoch {epoch+1}/{self.num_epochs} started...")
            train_loss = trainer.train_one_epoch(train_loader, scaler)
            # Now validate returns: (val_loss, accuracy, mae, r2)
            val_loss, accuracy, mae, r2 = trainer.validate(val_loader)
            trainer.train_losses.append(train_loss)
            trainer.val_losses.append(val_loss)
            trainer.accuracy_scores.append(accuracy)
            trainer.mae_scores.append(mae)
            trainer.r2_scores.append(r2)

            self.log_signal.emit(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Accuracy={accuracy:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(trainer.run_folder, f"best_model_focus_{epoch+1}.pth")
                torch.save(trainer.model.state_dict(), save_path)
                self.log_signal.emit(f"Saved best model to {save_path}")

            trainer.scheduler.step()
            self.update_signal.emit(epoch+1, train_loss, val_loss, accuracy, mae, r2)
        # After the training loop finishes:
        self.log_signal.emit("Training loop complete. Now testing on the final test set...")

        test_results = trainer.test_model(test_dataset)
        test_loss, test_acc, test_mae, test_r2 = test_results

        self.log_signal.emit(
            f"Final Test Results:\n"
            f"Loss={test_loss:.4f}, Acc={test_acc:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}"
)

        self.log_signal.emit("Saving training metrics and graphs...")

        trainer._save_results()
        self.finished_signal.emit()


# =============================================================================
# Main GUI window
# =============================================================================
class TrainingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defocus Regression Training GUI")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ----- Top: Graphs in a grid layout (2x2) -----
        graphs_layout = QGridLayout()
        self.loss_plot = pg.PlotWidget(title="Training & Validation Loss")
        self.accuracy_plot = pg.PlotWidget(title="Accuracy (± threshold)")
        self.mae_plot = pg.PlotWidget(title="MAE (µm)")
        self.r2_plot = pg.PlotWidget(title="R²")
        self.loss_plot.setFixedSize(400, 400)
        self.accuracy_plot.setFixedSize(400, 400)
        self.mae_plot.setFixedSize(400, 400)
        self.r2_plot.setFixedSize(400, 400)
        graphs_layout.addWidget(self.loss_plot, 0, 0)
        graphs_layout.addWidget(self.accuracy_plot, 0, 1)
        graphs_layout.addWidget(self.mae_plot, 1, 0)
        graphs_layout.addWidget(self.r2_plot, 1, 1)
        main_layout.addLayout(graphs_layout)

        # Create curve objects for plotting
        self.train_loss_curve = self.loss_plot.plot(pen=pg.mkPen('c', width=2))
        self.val_loss_curve = self.loss_plot.plot(pen=pg.mkPen('m', width=2))
        self.accuracy_curve = self.accuracy_plot.plot(pen=pg.mkPen('y', width=2))
        self.mae_curve = self.mae_plot.plot(pen=pg.mkPen('orange', width=2))
        self.r2_curve = self.r2_plot.plot(pen=pg.mkPen('lime', width=2))

        # Lists to hold the metrics for plotting
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.accuracy_scores = []
        self.mae_scores = []
        self.r2_scores = []

        # ----- Middle: Scrollable update text area -----
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(60)
        main_layout.addWidget(self.log_text)

        # ----- Bottom: Editable controls in a single horizontal row -----
        controls_layout = QHBoxLayout()

        # Training Images Directory: label, line edit, and browse button
        self.data_dir_line = QLineEdit()
        self.data_dir_line.setPlaceholderText("Select images directory")
        data_dir_button = QPushButton("Browse")
        data_dir_button.clicked.connect(self.select_data_dir)
        data_dir_container = QWidget()
        data_dir_container_layout = QHBoxLayout(data_dir_container)
        data_dir_container_layout.setContentsMargins(0, 0, 0, 0)
        data_dir_container_layout.addWidget(QLabel("Images Dir:"))
        data_dir_container_layout.addWidget(self.data_dir_line)
        data_dir_container_layout.addWidget(data_dir_button)
        controls_layout.addWidget(data_dir_container)

        # Annotations CSV: label, line edit, and browse button
        self.annotations_line = QLineEdit()
        self.annotations_line.setPlaceholderText("Select annotations CSV")
        annotations_button = QPushButton("Browse")
        annotations_button.clicked.connect(self.select_annotations)
        annotations_container = QWidget()
        annotations_container_layout = QHBoxLayout(annotations_container)
        annotations_container_layout.setContentsMargins(0, 0, 0, 0)
        annotations_container_layout.addWidget(QLabel("Annotations:"))
        annotations_container_layout.addWidget(self.annotations_line)
        annotations_container_layout.addWidget(annotations_button)
        controls_layout.addWidget(annotations_container)

        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["resnet101", "mobilenetv3", "efficientnet_lite", "convnext"])
        model_container = QWidget()
        model_layout = QHBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        controls_layout.addWidget(model_container)

        # Number of epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        epochs_container = QWidget()
        epochs_layout = QHBoxLayout(epochs_container)
        epochs_layout.setContentsMargins(0, 0, 0, 0)
        epochs_layout.addWidget(QLabel("Epochs:"))
        epochs_layout.addWidget(self.epochs_spin)
        controls_layout.addWidget(epochs_container)

        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(32)
        batch_container = QWidget()
        batch_layout = QHBoxLayout(batch_container)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.addWidget(QLabel("Batch:"))
        batch_layout.addWidget(self.batch_size_spin)
        controls_layout.addWidget(batch_container)

        # Learning rate
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-6, 1.0)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setValue(1e-4)
        lr_container = QWidget()
        lr_layout = QHBoxLayout(lr_container)
        lr_layout.setContentsMargins(0, 0, 0, 0)
        lr_layout.addWidget(QLabel("LR:"))
        lr_layout.addWidget(self.learning_rate_spin)
        controls_layout.addWidget(lr_container)

        # Threshold
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 10.0)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setValue(0.3)
        threshold_container = QWidget()
        threshold_layout = QHBoxLayout(threshold_container)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(QLabel("Thresh:"))
        threshold_layout.addWidget(self.threshold_spin)
        controls_layout.addWidget(threshold_container)

        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        device_container = QWidget()
        device_layout = QHBoxLayout(device_container)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.addWidget(QLabel("Device:"))
        device_layout.addWidget(self.device_combo)
        controls_layout.addWidget(device_container)

        # Start training button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        controls_layout.addWidget(self.start_button)

        main_layout.addLayout(controls_layout)

        self.worker = None

    # ---- File selection callbacks ----
    def select_data_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Training Images Directory", "")
        if directory:
            self.data_dir_line.setText(directory)

    def select_annotations(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Annotations CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.annotations_line.setText(file_path)

    # ---- Start training ----
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

    # ---- Slots to update plots and logs ----
    def on_update(self, epoch, train_loss, val_loss, accuracy, mae, r2):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.accuracy_scores.append(accuracy)
        self.mae_scores.append(mae)
        self.r2_scores.append(r2)
        self.train_loss_curve.setData(self.epochs, self.train_losses)
        self.val_loss_curve.setData(self.epochs, self.val_losses)
        self.accuracy_curve.setData(self.epochs, self.accuracy_scores)
        self.mae_curve.setData(self.epochs, self.mae_scores)
        self.r2_curve.setData(self.epochs, self.r2_scores)

    def on_finished(self):
        self.append_log("Training finished.")
        self.start_button.setEnabled(True)

    def append_log(self, message):
        self.log_text.append(message)

# =============================================================================
# Run the application
# =============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrainingGUI()
    window.show()
    sys.exit(app.exec_())
