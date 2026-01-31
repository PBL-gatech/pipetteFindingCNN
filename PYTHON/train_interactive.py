#!/usr/bin/env python
import sys
import os
import torch
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QMessageBox, QTextEdit, QFormLayout
)
from PyQt5.QtCore import QThread, pyqtSignal
from torch.utils.data import DataLoader
from train import create_run_folder, train_and_validate, test_model
from model import build_model
from data import PipetteDataModule

class TrainingWorker(QThread):
    update_signal = pyqtSignal(object)
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, model_name, train_images_dir, annotations_csv,
                 device, batch_size, learning_rate, weight_decay, num_epochs,
                 img_size, heatmap_stride, heatmap_sigma, lambda_z, huber_beta,
                 flip_p, rotate90_p, num_workers, mixed_precision,
                 checkpoint_folder=None):
        super().__init__()
        self.model_name = model_name
        self.train_images_dir = train_images_dir
        self.annotations_csv = annotations_csv
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.img_size = img_size
        self.heatmap_stride = heatmap_stride
        self.heatmap_sigma = heatmap_sigma
        self.lambda_z = lambda_z
        self.huber_beta = huber_beta
        self.flip_p = flip_p
        self.rotate90_p = rotate90_p
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.checkpoint_folder = checkpoint_folder

    def run(self):
        self.log_signal.emit("Setting up data module...")
        data_module = PipetteDataModule(
            self.train_images_dir,
            self.annotations_csv,
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            seed=42,
            img_size=self.img_size,
            flip_p=self.flip_p,
            rotate90_p=self.rotate90_p,
        )
        train_dataset, val_dataset, test_dataset = data_module.setup()
        self.log_signal.emit("Datasets prepared.")

        z_mean, z_std = data_module.get_z_stats()

        # Build config dictionary and create run folder (which saves config.json)
        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "device": self.device,
            "img_size": self.img_size,
            "heatmap_stride": self.heatmap_stride,
            "heatmap_sigma": self.heatmap_sigma,
            "lambda_z": self.lambda_z,
            "huber_beta": self.huber_beta,
            "flip_p": self.flip_p,
            "rotate90_p": self.rotate90_p,
            "num_workers": self.num_workers,
            "mixed_precision": self.mixed_precision,
            "z_mean": z_mean,
            "z_std": z_std,
        }
        run_folder = create_run_folder(self.model_name, config=config)
        self.log_signal.emit(f"Run folder created at: {run_folder}")

        device = torch.device(self.device)
        model = build_model(
            model_name=self.model_name,
            pretrained=True,
            heatmap_sigma=self.heatmap_sigma,
            heatmap_stride=self.heatmap_stride,
            lambda_z=self.lambda_z,
            huber_beta=self.huber_beta,
        )
        model.set_z_stats(z_mean, z_std)
        model.to(device)
        self.log_signal.emit("Model created and moved to device.")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        def update_callback(payload):
            self.update_signal.emit(payload)
            self.log_signal.emit(
                f"Epoch {payload['epoch']}: "
                f"Train Loss={payload['train_loss']:.4f}, Val Loss={payload['val_loss']:.4f}, "
                f"HMLoss train/val=({payload['train_loss_heatmap']:.4f}/{payload['val_loss_heatmap']:.4f}), "
                f"XYLoss train/val=({payload['train_loss_xy']:.4f}/{payload['val_loss_xy']:.4f}), "
                f"ZLoss train/val=({payload['train_loss_z']:.4f}/{payload['val_loss_z']:.4f}), "
                f"Val HM max/mean/std=({payload['val_hm_max']:.3f}/{payload['val_hm_mean']:.3f}/{payload['val_hm_std']:.3f}), "
                f"MAE x/y/z = ({payload['MAE_x_px']:.2f}, {payload['MAE_y_px']:.2f}, {payload['MAE_z_um']:.3f}), "
                f"R2 x/y/z = ({payload['R2_x']:.3f}, {payload['R2_y']:.3f}, {payload['R2_z']:.3f}), "
                f"LR={payload['learning_rate']:.2e}"
            )

        self.log_signal.emit("Starting training...")
        best_checkpoint, history = train_and_validate(
            model,
            train_loader,
            val_loader,
            device,
            run_folder,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            mixed_precision=self.mixed_precision,
            update_callback=update_callback,
        )

        self.log_signal.emit("Training complete. Testing the best model...")
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        test_results = test_model(
            model,
            test_loader,
            device,
            run_folder=run_folder,
        )
        self.log_signal.emit(
            f"Final Test Results: Loss={test_results['Test Loss']:.4f}, "
            f"MAE x/y/z = ({test_results['MAE_x_px']:.2f}, {test_results['MAE_y_px']:.2f}, {test_results['MAE_z_um']:.3f}), "
            f"R2 x/y/z = ({test_results['R2_x']:.3f}, {test_results['R2_y']:.3f}, {test_results['R2_z']:.3f})"
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
        self.mae_plot.addLegend()
        self.r2_plot.addLegend()
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
        self.model_combo.setEditable(True)
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

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setValue(2e-5)
        controls_form.addRow("Weight Decay:", self.weight_decay_spin)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(64, 1024)
        self.img_size_spin.setValue(224)
        controls_form.addRow("Image Size:", self.img_size_spin)

        self.heatmap_stride_spin = QSpinBox()
        self.heatmap_stride_spin.setRange(1, 64)
        self.heatmap_stride_spin.setValue(4)
        controls_form.addRow("Heatmap Stride:", self.heatmap_stride_spin)

        self.heatmap_sigma_spin = QDoubleSpinBox()
        self.heatmap_sigma_spin.setDecimals(3)
        self.heatmap_sigma_spin.setRange(0.1, 20.0)
        self.heatmap_sigma_spin.setValue(2.0)
        controls_form.addRow("Heatmap Sigma:", self.heatmap_sigma_spin)

        self.lambda_z_spin = QDoubleSpinBox()
        self.lambda_z_spin.setDecimals(3)
        self.lambda_z_spin.setRange(0.0, 100.0)
        self.lambda_z_spin.setValue(1.0)
        controls_form.addRow("Lambda Z:", self.lambda_z_spin)

        self.huber_beta_spin = QDoubleSpinBox()
        self.huber_beta_spin.setDecimals(3)
        self.huber_beta_spin.setRange(0.001, 10.0)
        self.huber_beta_spin.setValue(1.0)
        controls_form.addRow("Huber Beta:", self.huber_beta_spin)

        self.flip_prob_spin = QDoubleSpinBox()
        self.flip_prob_spin.setDecimals(2)
        self.flip_prob_spin.setRange(0.0, 1.0)
        self.flip_prob_spin.setValue(0.5)
        controls_form.addRow("Flip Probability:", self.flip_prob_spin)

        self.rotate_prob_spin = QDoubleSpinBox()
        self.rotate_prob_spin.setDecimals(2)
        self.rotate_prob_spin.setRange(0.0, 1.0)
        self.rotate_prob_spin.setValue(0.5)
        controls_form.addRow("Rotate90 Probability:", self.rotate_prob_spin)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 64)
        self.num_workers_spin.setValue(8)
        controls_form.addRow("DataLoader Workers:", self.num_workers_spin)

        self.mixed_precision_check = QCheckBox("Enable Mixed Precision (AMP)")
        self.mixed_precision_check.setChecked(True)
        controls_form.addRow(self.mixed_precision_check)

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
        self.mae_x = []
        self.mae_y = []
        self.mae_z = []
        self.r2_x = []
        self.r2_y = []
        self.r2_z = []
        self.lrs = []
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

        # reset traces for a fresh run
        self.epochs.clear()
        self.train_losses.clear()
        self.val_losses.clear()
        self.mae_x.clear()
        self.mae_y.clear()
        self.mae_z.clear()
        self.r2_x.clear()
        self.r2_y.clear()
        self.r2_z.clear()
        self.lrs.clear()
        self.loss_plot.clear()
        self.mae_plot.clear()
        self.r2_plot.clear()

        self.start_button.setEnabled(False)
        self.append_log("Training started...")
        model_name = self.model_combo.currentText()
        num_epochs = self.epochs_spin.value()
        batch_size = self.batch_size_spin.value()
        learning_rate = self.learning_rate_spin.value()
        weight_decay = self.weight_decay_spin.value()
        img_size = self.img_size_spin.value()
        heatmap_stride = self.heatmap_stride_spin.value()
        heatmap_sigma = self.heatmap_sigma_spin.value()
        lambda_z = self.lambda_z_spin.value()
        huber_beta = self.huber_beta_spin.value()
        flip_p = self.flip_prob_spin.value()
        rotate_p = self.rotate_prob_spin.value()
        num_workers = self.num_workers_spin.value()
        mixed_precision = self.mixed_precision_check.isChecked()
        device = self.device_combo.currentText()

        checkpoint_folder = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoint_folder, exist_ok=True)

        self.worker = TrainingWorker(
            model_name, train_dir, annotations, device,
            batch_size, learning_rate, weight_decay, num_epochs,
            img_size, heatmap_stride, heatmap_sigma, lambda_z, huber_beta,
            flip_p, rotate_p, num_workers, mixed_precision,
            checkpoint_folder
        )
        self.worker.update_signal.connect(self.on_update)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.log_signal.connect(self.append_log)
        self.worker.start()

    def on_update(self, payload):
        self.epochs.append(payload["epoch"])
        self.train_losses.append(payload["train_loss"])
        self.val_losses.append(payload["val_loss"])
        self.mae_x.append(payload["MAE_x_px"])
        self.mae_y.append(payload["MAE_y_px"])
        self.mae_z.append(payload["MAE_z_um"])
        self.r2_x.append(payload["R2_x"])
        self.r2_y.append(payload["R2_y"])
        self.r2_z.append(payload["R2_z"])
        self.lrs.append(payload["learning_rate"])

        self.loss_plot.clear()
        self.loss_plot.plot(self.epochs, self.train_losses, pen='c', name="Train Loss")
        self.loss_plot.plot(self.epochs, self.val_losses, pen='m', name="Val Loss")

        self.mae_plot.clear()
        self.mae_plot.plot(self.epochs, self.mae_x, pen=pg.mkPen('r', width=2), name="MAE X")
        self.mae_plot.plot(self.epochs, self.mae_y, pen=pg.mkPen('g', width=2), name="MAE Y")
        self.mae_plot.plot(self.epochs, self.mae_z, pen=pg.mkPen('b', width=2), name="MAE Z")

        self.r2_plot.clear()
        self.r2_plot.plot(self.epochs, self.r2_x, pen=pg.mkPen('r', width=2), name="R2 X")
        self.r2_plot.plot(self.epochs, self.r2_y, pen=pg.mkPen('g', width=2), name="R2 Y")
        self.r2_plot.plot(self.epochs, self.r2_z, pen=pg.mkPen('b', width=2), name="R2 Z")

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
