"""
PyQt GUI that combines data preparation and z-balancing.

This tool reuses:
- data_preparer.py for movement parsing and per-session zeroing/alignment
- balancer_z.py for dataset balancing and artifact/statistics output
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import traceback
from typing import Optional

from data_preparer import (
    extract_image_data,
    find_closest_movement_record,
    load_movement_data,
)
from balancer_z import CAP_PER_BIN, CULL_LIMIT, SEED, TARGET_COL, balance_defocus
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListView,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
CAMERA_FRAMES_MANIFEST = "camera_frames_manifest.csv"


def find_image_folder(session_dir: str) -> Optional[str]:
    """Return the image folder path for a session, if present."""
    candidate_folders = (
        "camera_frames",
        "P_DET_IMAGES",
        os.path.join("camera_frames", "P_DET_IMAGES"),
    )
    for folder_name in candidate_folders:
        candidate = os.path.join(session_dir, folder_name)
        if os.path.isdir(candidate):
            return candidate
    return None


def is_valid_session_dir(session_dir: str) -> bool:
    """A valid session must contain movement_recording.csv and an image folder."""
    movement_path = os.path.join(session_dir, "movement_recording.csv")
    return os.path.isfile(movement_path) and find_image_folder(session_dir) is not None


def collect_image_paths(image_root: str) -> list[str]:
    """Recursively collect image files under image_root."""
    image_paths: list[str] = []
    for root, _, files in os.walk(image_root):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, filename))
    return image_paths


def sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def unique_output_name(preferred_name: str, session_tag: str, used_names: set[str]) -> str:
    """Return a unique filename for the combined camera_frames folder."""
    if preferred_name not in used_names:
        return preferred_name

    root, ext = os.path.splitext(preferred_name)
    base_candidate = f"{session_tag}__{preferred_name}"
    if base_candidate not in used_names:
        return base_candidate

    suffix = 2
    while True:
        candidate = f"{session_tag}__{root}_{suffix}{ext}"
        if candidate not in used_names:
            return candidate
        suffix += 1


def load_manifest(manifest_path: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    if not os.path.isfile(manifest_path):
        return entries

    with open(manifest_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_path = row.get("source_path")
            output_name = row.get("output_name")
            if not source_path or not output_name:
                continue
            entries[source_path] = {
                "output_name": output_name,
                "size_bytes": row.get("size_bytes", ""),
                "mtime_ns": row.get("mtime_ns", ""),
            }
    return entries


def save_manifest(manifest_path: str, entries: dict[str, dict[str, str]]) -> None:
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_path", "output_name", "size_bytes", "mtime_ns"])
        for source_path in sorted(entries.keys()):
            row = entries[source_path]
            writer.writerow(
                [
                    source_path,
                    row.get("output_name", ""),
                    row.get("size_bytes", ""),
                    row.get("mtime_ns", ""),
                ]
            )


class PreprocessWorker(QObject):
    log = pyqtSignal(str)
    success = pyqtSignal(str, str)
    failure = pyqtSignal(str)
    done = pyqtSignal()

    def __init__(
        self,
        session_dirs: list[str],
        output_dir: str,
        cull_limit: Optional[float],
        cap_per_bin: int,
        seed: int,
        show_plots: bool,
    ):
        super().__init__()
        self.session_dirs = session_dirs
        self.output_dir = output_dir
        self.cull_limit = cull_limit
        self.cap_per_bin = cap_per_bin
        self.seed = seed
        self.show_plots = show_plots

    def run(self) -> None:
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            combined_camera_frames_dir = os.path.join(self.output_dir, "camera_frames")
            os.makedirs(combined_camera_frames_dir, exist_ok=True)
            manifest_path = os.path.join(self.output_dir, CAMERA_FRAMES_MANIFEST)
            manifest = load_manifest(manifest_path)

            all_rows: list[tuple[str, float, float, float, float]] = []
            used_output_filenames: set[str] = {
                name
                for name in os.listdir(combined_camera_frames_dir)
                if os.path.isfile(os.path.join(combined_camera_frames_dir, name))
            }

            for idx, session_dir in enumerate(self.session_dirs, start=1):
                self.log.emit(f"[{idx}/{len(self.session_dirs)}] Processing: {session_dir}")

                movement_path = os.path.join(session_dir, "movement_recording.csv")
                image_dir = find_image_folder(session_dir)
                if image_dir is None:
                    raise FileNotFoundError(
                        "Missing image folder in "
                        f"{session_dir} (expected camera_frames, P_DET_IMAGES, "
                        "or camera_frames/P_DET_IMAGES)."
                    )
                if not os.path.isfile(movement_path):
                    raise FileNotFoundError(f"Missing movement file: {movement_path}")

                movement_data = load_movement_data(movement_path)
                if not movement_data:
                    raise ValueError(f"No movement data loaded from {movement_path}")

                image_paths = collect_image_paths(image_dir)
                if not image_paths:
                    raise ValueError(f"No image files found in {image_dir}")

                image_files_with_timestamp: list[tuple[str, float, str]] = []
                for image_path in image_paths:
                    image_name = os.path.basename(image_path)
                    _, timestamp = extract_image_data(image_name)
                    image_files_with_timestamp.append((image_name, timestamp, image_path))
                image_files_with_timestamp.sort(key=lambda item: item[1])

                session_tag = sanitize_name(os.path.basename(os.path.normpath(session_dir)))
                session_rows: list[tuple[str, float, float, float, float]] = []
                session_copy_count = 0
                session_skip_count = 0
                for image_name, timestamp, image_path in image_files_with_timestamp:
                    movement_record = find_closest_movement_record(timestamp, movement_data)
                    pipette_x = float(movement_record["pipette"][0])
                    pipette_y = float(movement_record["pipette"][1])
                    pipette_z = float(movement_record["pipette"][2])
                    session_rows.append(
                        (
                            image_name,
                            pipette_z,
                            pipette_x,
                            pipette_y,
                            pipette_z,
                        )
                    )

                    source_path = os.path.normpath(os.path.abspath(image_path))
                    source_stat = os.stat(image_path)
                    source_size = str(source_stat.st_size)
                    source_mtime_ns = str(source_stat.st_mtime_ns)

                    manifest_entry = manifest.get(source_path)
                    reused_existing_copy = False
                    if manifest_entry:
                        candidate_output_name = manifest_entry.get("output_name", "")
                        candidate_output_path = os.path.join(
                            combined_camera_frames_dir, candidate_output_name
                        )
                        if (
                            candidate_output_name
                            and os.path.isfile(candidate_output_path)
                            and manifest_entry.get("size_bytes") == source_size
                            and manifest_entry.get("mtime_ns") == source_mtime_ns
                        ):
                            output_name = candidate_output_name
                            reused_existing_copy = True
                        else:
                            output_name = unique_output_name(
                                preferred_name=image_name,
                                session_tag=session_tag,
                                used_names=used_output_filenames,
                            )
                    else:
                        output_name = unique_output_name(
                            preferred_name=image_name,
                            session_tag=session_tag,
                            used_names=used_output_filenames,
                        )

                    output_image_path = os.path.join(combined_camera_frames_dir, output_name)
                    if not reused_existing_copy:
                        shutil.copy2(image_path, output_image_path)
                        session_copy_count += 1
                    else:
                        session_skip_count += 1

                    all_rows.append(
                        (
                            output_name,
                            pipette_z,
                            pipette_x,
                            pipette_y,
                            pipette_z,
                        )
                    )
                    used_output_filenames.add(output_name)
                    manifest[source_path] = {
                        "output_name": output_name,
                        "size_bytes": source_size,
                        "mtime_ns": source_mtime_ns,
                    }

                # Keep data_preparer.py behavior: save per-session CSV near source images.
                session_csv_path = os.path.join(image_dir, "pipette_z_data.csv")
                with open(session_csv_path, "w", newline="") as session_csv:
                    writer = csv.writer(session_csv)
                    writer.writerow(
                        [
                            "filename",
                            "defocus_microns",
                            "pipette_x_microns",
                            "pipette_y_microns",
                            "pipette_z_microns",
                        ]
                    )
                    writer.writerows(session_rows)

                self.log.emit(
                    f"Saved {len(session_rows)} rows to session CSV: {session_csv_path}"
                )
                self.log.emit(
                    f"Copied {session_copy_count} image(s) to: {combined_camera_frames_dir}"
                )
                self.log.emit(
                    f"Skipped {session_skip_count} already-copied image(s)."
                )

            if not all_rows:
                raise ValueError("No rows were generated from selected sessions.")

            save_manifest(manifest_path, manifest)

            combined_csv_path = os.path.join(self.output_dir, "pipette_z_data.csv")
            with open(combined_csv_path, "w", newline="") as combined_csv:
                writer = csv.writer(combined_csv)
                writer.writerow(
                    [
                        "filename",
                        "defocus_microns",
                        "pipette_x_microns",
                        "pipette_y_microns",
                        "pipette_z_microns",
                    ]
                )
                writer.writerows(all_rows)

            self.log.emit(
                f"Saved combined CSV with {len(all_rows)} rows: {combined_csv_path}"
            )
            self.log.emit(
                f"Combined camera_frames folder: {combined_camera_frames_dir}"
            )
            self.log.emit("Balancing combined dataset...")
            # balancer_z uses matplotlib; force a non-GUI backend in worker threads.
            import matplotlib.pyplot as plt

            plt.switch_backend("Agg")

            balanced_csv_path = balance_defocus(
                input_path=combined_csv_path,
                target_col=TARGET_COL,
                cull_limit=self.cull_limit,
                cap_per_bin=self.cap_per_bin,
                seed=self.seed,
                show_plots=False,
            )

            artifacts_dir = os.path.dirname(balanced_csv_path)
            self.log.emit(f"Balancing complete. Artifacts: {artifacts_dir}")
            self.success.emit(combined_csv_path, balanced_csv_path)
        except Exception:
            self.failure.emit(traceback.format_exc())
        finally:
            self.done.emit()


class DataPreprocesserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread: Optional[QThread] = None
        self.worker: Optional[PreprocessWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("Data Preprocesser")
        self.resize(980, 700)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        folders_label = QLabel("Selected date_time session folders:")
        root_layout.addWidget(folders_label)

        self.session_list = QListWidget()
        self.session_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        root_layout.addWidget(self.session_list)

        folder_buttons_layout = QHBoxLayout()
        self.add_session_button = QPushButton("Add Session Folder(s)")
        self.add_parent_button = QPushButton("Add From Parent Folder")
        self.remove_selected_button = QPushButton("Remove Selected")
        self.clear_button = QPushButton("Clear")

        folder_buttons_layout.addWidget(self.add_session_button)
        folder_buttons_layout.addWidget(self.add_parent_button)
        folder_buttons_layout.addWidget(self.remove_selected_button)
        folder_buttons_layout.addWidget(self.clear_button)
        root_layout.addLayout(folder_buttons_layout)

        options_layout = QGridLayout()
        root_layout.addLayout(options_layout)

        output_label = QLabel("Combined output folder:")
        self.output_dir_edit = QLineEdit()
        self.output_browse_button = QPushButton("Browse")

        options_layout.addWidget(output_label, 0, 0)
        options_layout.addWidget(self.output_dir_edit, 0, 1)
        options_layout.addWidget(self.output_browse_button, 0, 2)

        self.enable_cull_checkbox = QCheckBox("Enable cull limit (+/- microns)")
        self.enable_cull_checkbox.setChecked(CULL_LIMIT is not None)
        self.cull_spin = QDoubleSpinBox()
        self.cull_spin.setDecimals(3)
        self.cull_spin.setRange(0.0, 1000000.0)
        self.cull_spin.setValue(float(CULL_LIMIT or 0.0))
        self.cull_spin.setEnabled(self.enable_cull_checkbox.isChecked())

        options_layout.addWidget(self.enable_cull_checkbox, 1, 0)
        options_layout.addWidget(self.cull_spin, 1, 1)

        cap_label = QLabel("Cap per bin:")
        self.cap_spin = QSpinBox()
        self.cap_spin.setRange(1, 1000000)
        self.cap_spin.setValue(int(CAP_PER_BIN))
        options_layout.addWidget(cap_label, 2, 0)
        options_layout.addWidget(self.cap_spin, 2, 1)

        seed_label = QLabel("Random seed:")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(int(SEED))
        options_layout.addWidget(seed_label, 3, 0)
        options_layout.addWidget(self.seed_spin, 3, 1)

        self.show_plots_checkbox = QCheckBox(
            "Interactive plot windows (disabled during background run)"
        )
        self.show_plots_checkbox.setChecked(False)
        self.show_plots_checkbox.setEnabled(False)
        options_layout.addWidget(self.show_plots_checkbox, 4, 0)

        self.run_button = QPushButton("Run Preprocess + Balance")
        root_layout.addWidget(self.run_button)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        root_layout.addWidget(self.log_text)

        self.add_session_button.clicked.connect(self.add_multiple_sessions)
        self.add_parent_button.clicked.connect(self.add_from_parent_folder)
        self.remove_selected_button.clicked.connect(self.remove_selected_sessions)
        self.clear_button.clicked.connect(self.session_list.clear)
        self.output_browse_button.clicked.connect(self.browse_output_dir)
        self.enable_cull_checkbox.toggled.connect(self.cull_spin.setEnabled)
        self.run_button.clicked.connect(self.run_pipeline)

    def append_log(self, message: str) -> None:
        self.log_text.append(message)

    def add_multiple_sessions(self) -> None:
        dialog = QFileDialog(self, "Select one or more date_time session folders")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        # Native Windows folder picker does not support multi-select directories.
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)

        for view in dialog.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dialog.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if not dialog.exec_():
            return

        selected_dirs = dialog.selectedFiles()
        if not selected_dirs:
            return

        added_count = 0
        invalid_dirs = []
        for selected_dir in selected_dirs:
            if not is_valid_session_dir(selected_dir):
                invalid_dirs.append(selected_dir)
                continue
            if self._add_session_if_missing(selected_dir):
                added_count += 1

        if added_count > 0:
            self.append_log(f"Added {added_count} session folder(s).")

        if invalid_dirs:
            invalid_preview = "\n".join(invalid_dirs[:8])
            suffix = "\n..." if len(invalid_dirs) > 8 else ""
            QMessageBox.warning(
                self,
                "Some Folders Skipped",
                "These folders are invalid (need movement_recording.csv and image folder):\n\n"
                f"{invalid_preview}{suffix}",
            )

    def add_from_parent_folder(self) -> None:
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Select parent folder containing date_time sessions"
        )
        if not parent_dir:
            return

        added_count = 0
        for child_name in sorted(os.listdir(parent_dir)):
            child_path = os.path.join(parent_dir, child_name)
            if os.path.isdir(child_path) and is_valid_session_dir(child_path):
                if self._add_session_if_missing(child_path):
                    added_count += 1

        if added_count == 0:
            QMessageBox.information(
                self,
                "No Sessions Added",
                "No valid session folders were found (or all were already listed).",
            )
        else:
            self.append_log(f"Added {added_count} session folder(s) from: {parent_dir}")

    def _add_session_if_missing(self, session_dir: str) -> bool:
        normalized = os.path.normpath(os.path.abspath(session_dir))
        existing = {
            os.path.normpath(os.path.abspath(self.session_list.item(i).text()))
            for i in range(self.session_list.count())
        }
        if normalized in existing:
            return False

        self.session_list.addItem(normalized)
        self._set_default_output_dir_if_empty()
        return True

    def _set_default_output_dir_if_empty(self) -> None:
        if self.output_dir_edit.text().strip():
            return
        if self.session_list.count() == 0:
            return
        first_session = self.session_list.item(0).text()
        parent = os.path.dirname(first_session)
        default_output = os.path.join(parent, "combined")
        self.output_dir_edit.setText(default_output)

    def browse_output_dir(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select output folder for combined CSV and balancing artifacts"
        )
        if selected_dir:
            self.output_dir_edit.setText(os.path.normpath(selected_dir))

    def remove_selected_sessions(self) -> None:
        selected_items = self.session_list.selectedItems()
        for item in selected_items:
            row = self.session_list.row(item)
            self.session_list.takeItem(row)

    def _session_dirs(self) -> list[str]:
        return [self.session_list.item(i).text() for i in range(self.session_list.count())]

    def _set_controls_enabled(self, enabled: bool) -> None:
        controls = [
            self.add_session_button,
            self.add_parent_button,
            self.remove_selected_button,
            self.clear_button,
            self.output_dir_edit,
            self.output_browse_button,
            self.enable_cull_checkbox,
            self.cull_spin,
            self.cap_spin,
            self.seed_spin,
            self.show_plots_checkbox,
            self.run_button,
            self.session_list,
        ]
        for control in controls:
            control.setEnabled(enabled)
        if enabled:
            self.cull_spin.setEnabled(self.enable_cull_checkbox.isChecked())

    def run_pipeline(self) -> None:
        session_dirs = self._session_dirs()
        if not session_dirs:
            QMessageBox.warning(self, "No Sessions", "Add at least one session folder.")
            return

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            self._set_default_output_dir_if_empty()
            output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "No Output Folder", "Select an output folder.")
            return

        cull_limit = self.cull_spin.value() if self.enable_cull_checkbox.isChecked() else None
        cap_per_bin = self.cap_spin.value()
        seed = self.seed_spin.value()
        show_plots = False

        self.log_text.clear()
        self.append_log("Starting preprocessing + balancing...")
        self.append_log(f"Sessions selected: {len(session_dirs)}")
        self.append_log(f"Combined output folder: {output_dir}")

        self.thread = QThread(self)
        self.worker = PreprocessWorker(
            session_dirs=session_dirs,
            output_dir=output_dir,
            cull_limit=cull_limit,
            cap_per_bin=cap_per_bin,
            seed=seed,
            show_plots=show_plots,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.success.connect(self.on_success)
        self.worker.failure.connect(self.on_failure)
        self.worker.done.connect(self.thread.quit)
        self.worker.done.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)

        self._set_controls_enabled(False)
        self.thread.start()

    def on_success(self, combined_csv: str, balanced_csv: str) -> None:
        artifacts_dir = os.path.dirname(balanced_csv)
        combined_camera_frames_dir = os.path.join(os.path.dirname(combined_csv), "camera_frames")
        self.append_log(f"Combined CSV: {combined_csv}")
        self.append_log(f"Combined images: {combined_camera_frames_dir}")
        self.append_log(f"Balanced CSV: {balanced_csv}")
        self.append_log(f"Artifacts folder: {artifacts_dir}")
        QMessageBox.information(
            self,
            "Completed",
            "Preprocessing and balancing finished.\n\n"
            f"Combined CSV:\n{combined_csv}\n\n"
            f"Combined camera_frames:\n{combined_camera_frames_dir}\n\n"
            f"Balanced CSV:\n{balanced_csv}\n\n"
            f"Artifacts folder:\n{artifacts_dir}",
        )

    def on_failure(self, error_trace: str) -> None:
        self.append_log("ERROR")
        self.append_log(error_trace)
        QMessageBox.critical(
            self,
            "Failed",
            "Processing failed. See log output for details.",
        )

    def on_thread_finished(self) -> None:
        self._set_controls_enabled(True)
        self.thread = None
        self.worker = None


def run_app() -> int:
    app = QApplication(sys.argv)
    window = DataPreprocesserWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(run_app())
