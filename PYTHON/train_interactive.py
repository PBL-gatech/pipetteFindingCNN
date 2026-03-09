#!/usr/bin/env python
import sys
import os
import torch
import pyqtgraph as pg
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QMessageBox, QTextEdit, QFormLayout
)
from PyQt5.QtCore import QThread, pyqtSignal
from torch.utils.data import DataLoader
from train import (
    create_run_folder,
    get_regression_model,
    train_and_validate,
    test_model,
    compute_focus_params,
    weighted_huber_loss,
)
from converter2 import convert_checkpoint_to_torchscript
from data import PipetteDataModule

class TrainingWorker(QThread):
    update_signal = pyqtSignal(int, float, float, float, float, float, float)
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, model_name, train_images_dir, annotations_csv,
                 device, batch_size, learning_rate, num_epochs,
                 img_size, huber_beta, checkpoint_folder=None,
                 focus_inner_um: float = 3.0,
                 focus_weight_ratio: float | None = None,
                 negative_weight_ratio: float = 1.0,
                 enable_contrast_stretch: bool = False,
                 enable_aug_flip_rotate: bool = False):
        super().__init__()
        self.model_name = model_name
        self.train_images_dir = train_images_dir
        self.annotations_csv = annotations_csv
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.img_size = img_size
        self.huber_beta = huber_beta
        self.checkpoint_folder = checkpoint_folder
        self.focus_inner_um = focus_inner_um
        self.focus_weight_ratio = focus_weight_ratio
        self.negative_weight_ratio = negative_weight_ratio
        self.enable_contrast_stretch = bool(enable_contrast_stretch)
        self.enable_aug_flip_rotate = bool(enable_aug_flip_rotate)

    def run(self):
        # Enable fast CUDA kernels when available
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Build config dictionary and create run folder (which saves config.txt)
        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "device": self.device,
            "img_size": self.img_size,
            "huber_beta": self.huber_beta,
            "focus_inner_um": self.focus_inner_um,
            "negative_weight_ratio": self.negative_weight_ratio,
            "enable_contrast_stretch": self.enable_contrast_stretch,
            "enable_aug_flip_rotate": self.enable_aug_flip_rotate,
        }
        run_folder = create_run_folder(self.model_name, config=config)
        self.log_signal.emit(f"Run folder created at: {run_folder}")
        self.log_signal.emit(
            "Preprocess toggles -> "
            f"contrast_stretch={self.enable_contrast_stretch}, "
            f"aug_flip_rotate={self.enable_aug_flip_rotate}"
        )

        self.log_signal.emit("Setting up data module...")
        cache_images = True  # enable RAM cache for faster training
        data_module = PipetteDataModule(
            self.train_images_dir,
            self.annotations_csv,
            train_split=0.7, val_split=0.2, test_split=0.1, seed=42,
            default_img_size=self.img_size,
            split_save_path=os.path.join(run_folder, "data_splits.pkl"),
            channel_stats_cache_path=os.path.join(run_folder, "channel_stats.pkl"),
            channel_stats_max_images=2000,
            enable_contrast_stretch=self.enable_contrast_stretch,
            enable_aug_flip_rotate=self.enable_aug_flip_rotate,
            cache_images=cache_images,
            logger=self.log_signal.emit,
        )
        train_dataset, val_dataset, test_dataset = data_module.setup()
        self.log_signal.emit("Datasets prepared.")

        focus_outer_um, focus_weight_ratio = compute_focus_params(
            train_dataset,
            inner_band_um=self.focus_inner_um,
            manual_weight_ratio=self.focus_weight_ratio,
        )
        self.focus_weight_ratio = focus_weight_ratio
        self.log_signal.emit(
            f"Weighted Huber config -> inner |z|<={self.focus_inner_um}µm weight x{self.focus_weight_ratio:.2f}; outer span ~{focus_outer_um:.2f}µm"
        )

        # Persist normalization stats for inference
        z_mean = getattr(train_dataset, "z_mean", None)
        z_std = getattr(train_dataset, "z_std", None)
        if z_mean is not None and z_std is not None:
            stats_path = os.path.join(run_folder, "z_norm.json")
            with open(stats_path, "w") as f:
                json.dump({"z_mean": z_mean, "z_std": z_std}, f, indent=2)
            self.log_signal.emit(f"Saved z normalization stats to: {stats_path}")

        if data_module.channel_mean is not None and data_module.channel_std is not None:
            channel_stats_path = os.path.join(run_folder, "channel_norm.json")
            with open(channel_stats_path, "w") as f:
                json.dump({"mean": data_module.channel_mean, "std": data_module.channel_std}, f, indent=2)
            self.log_signal.emit(f"Saved channel normalization stats to: {channel_stats_path}")

        preprocess_cfg_path = os.path.join(run_folder, "preprocess_config.json")
        with open(preprocess_cfg_path, "w") as f:
            json.dump(
                {
                    "enable_contrast_stretch": self.enable_contrast_stretch,
                    "enable_aug_flip_rotate": self.enable_aug_flip_rotate,
                    "contrast_method": "mu_plus_minus_2sigma_clip_uint8",
                },
                f,
                indent=2,
            )
        self.log_signal.emit(f"Saved preprocessing config to: {preprocess_cfg_path}")

        device = torch.device(self.device)
        self.log_signal.emit(f"CUDA available: {torch.cuda.is_available()} | Selected device: {device}")
        if device.type == "cuda":
            try:
                self.log_signal.emit(f"GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
        else:
            self.log_signal.emit("CUDA not in use; training on CPU.")
        model = get_regression_model(model_name=self.model_name, pretrained=True, output_dim=1)
        model = model.to(device, memory_format=torch.channels_last)
        self.log_signal.emit("Model created and moved to device.")

        def loader_kwargs():
            if cache_images:
                return {"num_workers": 0, "pin_memory": device.type == "cuda"}
            num_workers = os.cpu_count() or 0
            num_workers = max(2, min(num_workers - 2, 16)) if num_workers > 0 else 0
            kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda"}
            if num_workers > 0:
                kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
            return kwargs

        dl_kwargs = loader_kwargs()

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, **dl_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, **dl_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, **dl_kwargs)
        self.log_signal.emit(
            f"Dataloaders ready | train batches: {len(train_loader)} | val batches: {len(val_loader)} | test batches: {len(test_loader)} | workers: {dl_kwargs.get('num_workers',0)}"
        )

        def update_callback(epoch, train_loss, val_loss, mae, mae_pos, mae_neg, r2):
            self.update_signal.emit(epoch, train_loss, val_loss, mae, mae_pos, mae_neg, r2)
            self.log_signal.emit(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"MAE={mae:.4f}, MAE+={mae_pos:.4f}, MAE-={mae_neg:.4f}, R^2={r2:.4f}"
            )

        self.log_signal.emit("Starting training...")
        (
            best_checkpoint,
            epochs_list,
            train_losses,
            val_losses,
            mae_scores,
            r2_scores,
            mae_scores_pos,
            mae_scores_neg,
        ) = train_and_validate(
            model, train_loader, val_loader, device, run_folder,
            num_epochs=self.num_epochs, update=True, update_callback=update_callback,
            huber_beta=self.huber_beta, learning_rate=self.learning_rate,
            logger=self.log_signal.emit, compile_model=False,
            focus_inner_um=self.focus_inner_um, focus_outer_um=focus_outer_um,
            focus_weight_ratio=self.focus_weight_ratio,
            negative_weight_ratio=self.negative_weight_ratio,
        )

        self.log_signal.emit("Training complete. Testing the best model...")
        torchscript_output = os.path.join(run_folder, "PipetteFocuserNet.pt")
        convert_checkpoint_to_torchscript(
            checkpoint_path=best_checkpoint,
            output_path=torchscript_output,
            model_name=self.model_name,
            img_size=self.img_size,
        )
        model.load_state_dict(torch.load(best_checkpoint))
        criterion = lambda outputs, targets: weighted_huber_loss(
            outputs,
            targets,
            self.huber_beta,
            getattr(train_dataset, "z_mean", 0.0),
            getattr(train_dataset, "z_std", 1.0),
            self.focus_inner_um,
            self.focus_weight_ratio,
            self.negative_weight_ratio,
        )
        test_results = test_model(model, test_loader, device, criterion, run_folder)
        self.log_signal.emit(
            f"Final Test Results: Loss={test_results['Test Loss']:.4f}, "
            f"MAE={test_results['Test MAE']:.4f}, R^2={test_results['Test R2']:.4f}"
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
        self.loss_plot = pg.PlotWidget(title="MSE Loss (normalized)")
        self.mae_plot = pg.PlotWidget(title="MAE")
        self.r2_plot = pg.PlotWidget(title="R^2")
        self.loss_plot.addLegend()
        self.mae_plot.addLegend()
        self.r2_plot.addLegend()
        self.loss_plot.setLabel("bottom", "Epoch")
        self.loss_plot.setLabel("left", "Loss")
        self.mae_plot.setLabel("bottom", "Epoch")
        self.mae_plot.setLabel("left", "MAE", units="um")
        self.r2_plot.setLabel("bottom", "Epoch")
        self.r2_plot.setLabel("left", "R^2")
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

        self.enable_contrast_checkbox = QCheckBox("Enable contrast stretching")
        self.enable_contrast_checkbox.setChecked(False)
        self.enable_contrast_checkbox.setToolTip(
            "Apply per-image mu +/- 2 sigma contrast stretch before normalization."
        )
        controls_form.addRow(self.enable_contrast_checkbox)

        self.enable_aug_flip_rotate_checkbox = QCheckBox("Enable flipping augmentation")
        self.enable_aug_flip_rotate_checkbox.setChecked(False)
        self.enable_aug_flip_rotate_checkbox.setToolTip(
            "Apply random flips plus occasional 90-degree turns during training."
        )
        controls_form.addRow(self.enable_aug_flip_rotate_checkbox)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "mobilenetv3_large_100",
            "mobilenetv4_hybrid_medium.ix_e550_r224_in1k",
            "efficientnetv2_s",
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "mobilevitv2_050",
            "mobilevitv2_100",
            "efficientformer_l1.snap_dist_in1k",
            "swin_tiny_patch4_window7_224",
        ])
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

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(64, 512)
        self.img_size_spin.setSingleStep(16)
        self.img_size_spin.setValue(224)
        controls_form.addRow("Image Size:", self.img_size_spin)

        self.huber_beta_spin = QDoubleSpinBox()
        self.huber_beta_spin.setDecimals(3)
        self.huber_beta_spin.setRange(0.01, 10.0)
        self.huber_beta_spin.setValue(1.0)
        controls_form.addRow("Huber beta:", self.huber_beta_spin)

        self.focus_weight_spin = QDoubleSpinBox()
        self.focus_weight_spin.setDecimals(2)
        self.focus_weight_spin.setRange(0.0, 100.0)
        self.focus_weight_spin.setSingleStep(0.1)
        self.focus_weight_spin.setValue(0.0)  # 0 = auto (outer/inner)
        controls_form.addRow("Focus weight (0=auto):", self.focus_weight_spin)

        self.negative_weight_spin = QDoubleSpinBox()
        self.negative_weight_spin.setDecimals(2)
        self.negative_weight_spin.setRange(0.0, 100.0)
        self.negative_weight_spin.setSingleStep(0.1)
        self.negative_weight_spin.setValue(1.0)
        controls_form.addRow("Negative z-weight (neg samples):", self.negative_weight_spin)

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
        self.mae_scores_pos = []
        self.mae_scores_neg = []
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
        # Reset per-run metrics so a new run doesn't append to the previous one
        self.epochs.clear()
        self.train_losses.clear()
        self.val_losses.clear()
        self.mae_scores.clear()
        self.mae_scores_pos.clear()
        self.mae_scores_neg.clear()
        self.r2_scores.clear()
        self.loss_plot.clear()
        self.mae_plot.clear()
        self.r2_plot.clear()

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
        img_size = self.img_size_spin.value()
        huber_beta = self.huber_beta_spin.value()
        device = self.device_combo.currentText()
        focus_weight_ratio = self.focus_weight_spin.value()
        focus_weight_ratio = None if focus_weight_ratio <= 0 else focus_weight_ratio
        negative_weight_ratio = self.negative_weight_spin.value()
        focus_inner_um = 3.0  # fixed focus band per current data balancing
        enable_contrast_stretch = self.enable_contrast_checkbox.isChecked()
        enable_aug_flip_rotate = self.enable_aug_flip_rotate_checkbox.isChecked()

        checkpoint_folder = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoint_folder, exist_ok=True)

        self.worker = TrainingWorker(
            model_name, train_dir, annotations, device,
            batch_size, learning_rate, num_epochs,
            img_size, huber_beta,
            checkpoint_folder,
            focus_inner_um=focus_inner_um,
            focus_weight_ratio=focus_weight_ratio,
            negative_weight_ratio=negative_weight_ratio,
            enable_contrast_stretch=enable_contrast_stretch,
            enable_aug_flip_rotate=enable_aug_flip_rotate,
        )
        self.worker.update_signal.connect(self.on_update)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.log_signal.connect(self.append_log)
        self.worker.start()

    def on_update(self, epoch, train_loss, val_loss, mae, mae_pos, mae_neg, r2):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.mae_scores.append(mae)
        self.mae_scores_pos.append(mae_pos)
        self.mae_scores_neg.append(mae_neg)
        self.r2_scores.append(r2)
        self.loss_plot.clear()
        self.loss_plot.plot(self.epochs, self.train_losses, pen='c', name="Train Loss")
        self.loss_plot.plot(self.epochs, self.val_losses, pen='m', name="Val Loss")
        self.mae_plot.clear()
        self.mae_plot.plot(self.epochs, self.mae_scores, pen=pg.mkPen('orange', width=2), name="MAE")
        self.mae_plot.plot(self.epochs, self.mae_scores_pos, pen=pg.mkPen('cyan', width=1.8), name="MAE +")
        self.mae_plot.plot(self.epochs, self.mae_scores_neg, pen=pg.mkPen('magenta', width=1.8), name="MAE -")
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
