import sys
import os
import threading
import signal
import psutil
from PySide6.QtWidgets import QListWidgetItem, QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QCheckBox, QComboBox, QLabel, QListWidget, QTextEdit, QRadioButton, QButtonGroup, QTabWidget, QAbstractItemView
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QUrl, Qt, QThread, Signal
from jpeg_to_audio import Book, EasyOCROCR, TesseractOCR, GTTSStrategy, ParlerTTSStrategy, KokoroTTSStrategy, XAITextPostProcessStrategy, FileSystemRepo

# Global cancellation flag
_cancellation_flag = threading.Event()

class ProcessingThread(QThread):
    """Background thread for processing jobs"""
    finished = Signal()
    error = Signal(str)
    cancelled = Signal()
    
    def __init__(self, processing_function, *args, **kwargs):
        super().__init__()
        self.processing_function = processing_function
        self.args = args
        self.kwargs = kwargs
        self._cancelled = False
        self._child_processes = []
    
    def cancel(self):
        """Cancel the processing and kill child processes"""
        global _cancellation_flag
        _cancellation_flag.set()
        self._cancelled = True
        
        # Kill all child processes
        self._kill_child_processes()
    
    def _kill_child_processes(self):
        """Kill all child processes spawned by this thread"""
        try:
            # Get current process
            current_process = psutil.Process()
            
            # Get all child processes
            children = current_process.children(recursive=True)
            
            for child in children:
                try:
                    # Try to terminate gracefully first
                    child.terminate()
                    child.wait(timeout=2)  # Wait up to 2 seconds
                except psutil.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    child.kill()
                except psutil.NoSuchProcess:
                    # Process already dead
                    pass
                    
        except Exception as e:
            print(f"Error killing child processes: {e}")
    
    def run(self):
        try:
            # Reset cancellation flag
            global _cancellation_flag
            _cancellation_flag.clear()
            
            # Run the processing function
            self.processing_function(*self.args, **self.kwargs)
            
            if not _cancellation_flag.is_set():
                self.finished.emit()
            else:
                self.cancelled.emit()
        except Exception as e:
            if not _cancellation_flag.is_set():
                self.error.emit(str(e))
            else:
                self.cancelled.emit()
        finally:
            # Always clean up child processes
            if self._cancelled:
                self._kill_child_processes()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ebook to Audio Converter")

        # Status bar for messages
        self.statusBar().showMessage("Ready")
        
        # Processing thread
        self.processing_thread = None

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.folder_label = QLabel("Select folder with book pages:")
        left_layout.addWidget(self.folder_label)

        self.select_folder_button = QPushButton("Select Folder")
        self.select_folder_button.setToolTip("Choose the root folder containing book pages (JPEG/JPG).")
        self.select_folder_button.clicked.connect(self.select_folder)
        left_layout.addWidget(self.select_folder_button)

        # Radio buttons for text extraction options
        self.text_extraction_group = QButtonGroup(self)
        self.extract_text_radio = QRadioButton("Extract Text")
        self.rerun_post_process_radio = QRadioButton("Rerun Post-Processing")
        self.skip_text_extraction = QRadioButton("Skip Text Extraction")
        self.text_extraction_group.addButton(self.extract_text_radio)
        self.text_extraction_group.addButton(self.rerun_post_process_radio)
        self.text_extraction_group.addButton(self.skip_text_extraction)
        left_layout.addWidget(self.extract_text_radio)
        left_layout.addWidget(self.rerun_post_process_radio)
        left_layout.addWidget(self.skip_text_extraction)
        self.extract_text_radio.setChecked(True)  # Default selection
        self.extract_text_radio.setToolTip("Extract text from images using the selected OCR strategy.")
        self.rerun_post_process_radio.setToolTip("Rerun post-processing on existing text files.")
        self.skip_text_extraction.setToolTip("Skip text extraction step.")

        self.generate_audio_checkbox = QCheckBox("Generate Audio")
        left_layout.addWidget(self.generate_audio_checkbox)
        self.generate_audio_checkbox.setToolTip("Generate audio from extracted text using the selected TTS strategy.")

        self.ocr_strategy_label = QLabel("Select OCR Strategy:")
        left_layout.addWidget(self.ocr_strategy_label)
        self.ocr_strategy_label.setToolTip("Choose the OCR engine for text extraction.")

        self.ocr_strategy_combobox = QComboBox()
        self.ocr_strategy_combobox.addItem("EasyOCR", EasyOCROCR)
        self.ocr_strategy_combobox.addItem("TesseractOCR", TesseractOCR)
        left_layout.addWidget(self.ocr_strategy_combobox)
        self.ocr_strategy_combobox.setToolTip("OCR engine selection.")

        self.tts_strategy_label = QLabel("Select TTS Strategy:")
        left_layout.addWidget(self.tts_strategy_label)
        self.tts_strategy_label.setToolTip("Choose the TTS engine for audio generation.")

        self.tts_strategy_combobox = QComboBox()
        self.tts_strategy_combobox.addItem("KokoroTTS", KokoroTTSStrategy)
        self.tts_strategy_combobox.addItem("GTTS", GTTSStrategy)
        self.tts_strategy_combobox.addItem("ParlerTTS", ParlerTTSStrategy)
        left_layout.addWidget(self.tts_strategy_combobox)
        self.tts_strategy_combobox.setToolTip("TTS engine selection.")

         # Dynamic Voice Selection Combobox
        self.voice_combobox = QComboBox()
        self.voice_combobox.setHidden(False)  # Initially hidden
        left_layout.addWidget(self.voice_combobox)
        self.update_voice_selection(0)  # Default to british George
        # Connect the signal for index change
        self.tts_strategy_combobox.currentIndexChanged.connect(self.update_voice_selection)
        self.voice_combobox.setToolTip("Select a voice for Kokoro TTS.")


        self.post_process_strategy_label = QLabel("Select Post-Process Strategy:")
        left_layout.addWidget(self.post_process_strategy_label)
        self.post_process_strategy_label.setToolTip("Choose the post-processing strategy for OCR text.")

        self.post_process_strategy_combobox = QComboBox()
        self.post_process_strategy_combobox.addItem("XAITextPostProcess", XAITextPostProcessStrategy)
        left_layout.addWidget(self.post_process_strategy_combobox)
        self.post_process_strategy_combobox.setToolTip("Post-processing strategy selection.")

        self.chapter_label = QLabel("Select Chapter(s):")
        left_layout.addWidget(self.chapter_label)
        self.chapter_label.setToolTip("Select one or more chapters to process or export.")

        # --- Change from QComboBox to QListWidget for multi-select ---
        self.chapter_listwidget = QListWidget()
        self.chapter_listwidget.setSelectionMode(QAbstractItemView.MultiSelection)
        left_layout.addWidget(self.chapter_listwidget)
        self.chapter_listwidget.setToolTip("Select one or more chapters. Hold Ctrl/Cmd or Shift to select multiple.")
        # ------------------------------------------------------------

        # Create execute/cancel button
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute)
        self.execute_button.setToolTip("Run the selected actions (text extraction, post-processing, audio generation).")
        left_layout.addWidget(self.execute_button)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export)
        self.export_button.setToolTip("Export processed chapters to the repository.")
        left_layout.addWidget(self.export_button)

        main_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.display_file_content)
        right_layout.addWidget(self.file_list)

        # --- Tabbed region for file content and logs ---
        self.tabs = QTabWidget()
        self.file_content = QTextEdit()
        self.file_content.setReadOnly(True)
        self.logs_content = QTextEdit()
        self.logs_content.setReadOnly(True)
        self.tabs.addTab(self.file_content, "File Content")
        self.tabs.addTab(self.logs_content, "Logs")
        right_layout.addWidget(self.tabs)
        # ------------------------------------------------

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.play_button = QPushButton("Play Audio")
        self.play_button.clicked.connect(self.media_player.play)
        self.play_button.setVisible(False)
        right_layout.addWidget(self.play_button)
        self.stop_button = QPushButton("Stop Audio")
        self.stop_button.clicked.connect(self.media_player.stop)
        self.stop_button.setVisible(False)
        right_layout.addWidget(self.stop_button)

        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def update_voice_selection(self, index):
        selected_strategy = self.tts_strategy_combobox.itemData(index)
        
        if selected_strategy == KokoroTTSStrategy:
            # Populate the voice combobox with voices from KokoroTTSStrategy
            self.voice_combobox.clear()
            for voice in KokoroTTSStrategy.voices:
                self.voice_combobox.addItem(voice)
            self.voice_combobox.setHidden(False)
            self.voice_combobox.setCurrentIndex(7)  # Default to british George
        else:
            # Hide the voice combobox for other strategies
            self.voice_combobox.setHidden(True)

    @staticmethod
    def truncate_text(text, length=50):
        if len(text) >= length:
            text = '...' + text[max(0, len(text) - length):]
        return text

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_label.setText(f"Selected folder:\n{self.truncate_text(folder)}")
            self.selected_folder = folder
            self.load_files()
            self.load_chapters()
            self.statusBar().showMessage("Folder loaded.")
        else:
            self.statusBar().showMessage("No folder selected.")

    def load_files(self):
        self.file_list.clear()
        found_files = []
        for root, _, files in os.walk(self.selected_folder):
            for file in files:
                # Exclude chunk txt files
                if file.endswith(('.txt', '.mp3', '.wav')) and not ("_chunk_" in file and file.endswith('.txt')):
                    found_files.append(os.path.join(root, file))
        sorted_files = sorted(found_files)
        for file in sorted_files:
            item = QListWidgetItem(self.truncate_text(file))  # Display only last 50 characters
            item.setData(Qt.UserRole, file)  # Store full path
            self.file_list.addItem(item)

    def load_chapters(self):
        self.chapter_listwidget.clear()
        # Add "All Chapters" as a special item
        all_item = QListWidgetItem("All Chapters")
        self.chapter_listwidget.addItem(all_item)
        folders = []
        for root, dirs, _ in os.walk(self.selected_folder):
            for dir in dirs:
                folders.append(dir)
        for dir in sorted(folders):
            self.chapter_listwidget.addItem(QListWidgetItem(dir))
        # Select "All Chapters" by default
        self.chapter_listwidget.setCurrentRow(0)

    def get_selected_chapters(self):
        selected_items = self.chapter_listwidget.selectedItems()
        chapters = [item.text() for item in selected_items]
        if "All Chapters" in chapters or not chapters:
            return ["All Chapters"]
        return chapters

    def display_file_content(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                self.file_content.setPlainText(file.read())
            self.file_content.setVisible(True)
            self.play_button.setVisible(False)
            self.stop_button.setVisible(False)
            self.tabs.setCurrentWidget(self.file_content)  # Switch to file content tab
        elif file_path.endswith(('.mp3', '.wav')):
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.file_content.setVisible(False)
            self.play_button.setVisible(True)
            self.stop_button.setVisible(True)
            self.tabs.setCurrentWidget(self.file_content)  # Still switch to file content tab

    def set_processing_state(self, processing=True):
        """Enable/disable UI elements during processing"""
        self.select_folder_button.setEnabled(not processing)
        self.export_button.setEnabled(not processing)
        
        if processing:
            # Change execute button to cancel button
            self.execute_button.setText("Cancel")
            self.execute_button.clicked.disconnect()
            self.execute_button.clicked.connect(self.cancel_processing)
            self.execute_button.setToolTip("Cancel the current processing job.")
            self.execute_button.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; }")
            
            self.statusBar().showMessage("Processing... Please wait.")
            self.logs_content.clear()
            self.logs_content.append("=== Processing Started ===\n")
            self.tabs.setCurrentWidget(self.logs_content)  # Switch to logs tab
        else:
            # Change cancel button back to execute button
            self.execute_button.setText("Execute")
            self.execute_button.clicked.disconnect()
            self.execute_button.clicked.connect(self.execute)
            self.execute_button.setToolTip("Run the selected actions (text extraction, post-processing, audio generation).")
            self.execute_button.setStyleSheet("")  # Reset to default style
            
            self.statusBar().showMessage("Ready")

    def cancel_processing(self):
        """Cancel the current processing job and kill all child processes"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.cancel()
            self.statusBar().showMessage("Cancelling and killing processes...")

    def execute(self):
        if not hasattr(self, 'selected_folder'):
            self.folder_label.setText("Please select a folder first.")
            self.statusBar().showMessage("Please select a folder first.")
            return
        
        self.set_processing_state(True)

        def process_job():
            global _cancellation_flag

            ocr_strategy_class = self.ocr_strategy_combobox.currentData()
            tts_strategy_class = self.tts_strategy_combobox.currentData()
            post_process_strategy_class = self.post_process_strategy_combobox.currentData()

            ocr_strategy = ocr_strategy_class(post_process_strategy=post_process_strategy_class())
            tts_strategy = tts_strategy_class()

            if isinstance(tts_strategy, KokoroTTSStrategy) and not self.voice_combobox.isHidden():
                selected_voice = self.voice_combobox.currentText()
                tts_strategy.voice_name = selected_voice

            book = Book(self.selected_folder, ocr_strategy, tts_strategy)
            selected_chapters = self.get_selected_chapters()

            def log(msg):
                QThread.msleep(10)
                self.append_log(msg)

            def refresh_files():
                QThread.msleep(10)
                self.load_files()

            if "All Chapters" in selected_chapters:
                chapters = list(book.chapters.keys())
            else:
                chapters = selected_chapters

            for chapter in chapters:
                if self.extract_text_radio.isChecked():
                    log(f"Started text extraction for chapter: {chapter}")
                    book.extract_text(chapter, overwrite=True)
                    if _cancellation_flag.is_set():
                        log(f"Cancelled during text extraction for chapter: {chapter}")
                        return
                    book.save_chapter_text(chapter)
                    log(f"Finished text extraction for chapter: {chapter}")
                    refresh_files()

                if self.rerun_post_process_radio.isChecked():
                    log(f"Started post-processing for chapter: {chapter}")
                    book.rerun_post_processing(chapter)
                    if _cancellation_flag.is_set():
                        log(f"Cancelled during post-processing for chapter: {chapter}")
                        return
                    log(f"Finished post-processing for chapter: {chapter}")
                    refresh_files()

                if self.generate_audio_checkbox.isChecked():
                    log(f"Started audio generation for chapter: {chapter}")
                    book.convert_text_to_audio(chapter, overwrite=True)
                    if _cancellation_flag.is_set():
                        log(f"Cancelled during audio generation for chapter: {chapter}")
                        return
                    log(f"Finished audio generation for chapter: {chapter}")
                    refresh_files()

        self.processing_thread = ProcessingThread(process_job)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.cancelled.connect(self.on_processing_cancelled)
        self.processing_thread.start()

    def on_processing_finished(self):
        """Called when processing is complete"""
        self.load_files()
        self.set_processing_state(False)
        self.statusBar().showMessage("Processing complete.")

    def on_processing_cancelled(self):
        """Called when processing is cancelled"""
        self.set_processing_state(False)
        self.statusBar().showMessage("Processing cancelled.")

    def on_processing_error(self, error_message):
        """Called when processing encounters an error"""
        self.set_processing_state(False)
        self.statusBar().showMessage(f"Error: {error_message}")

    def export(self):
        if not hasattr(self, 'selected_folder'):
            self.folder_label.setText("Please select a folder first.")
            self.statusBar().showMessage("Please select a folder first.")
            return
        
        self.set_processing_state(True)

        def export_job():
            global _cancellation_flag

            ocr_strategy_class = self.ocr_strategy_combobox.currentData()
            tts_strategy_class = self.tts_strategy_combobox.currentData()
            post_process_strategy_class = self.post_process_strategy_combobox.currentData()

            ocr_strategy = ocr_strategy_class(post_process_strategy=post_process_strategy_class())
            tts_strategy = tts_strategy_class()

            book = Book(self.selected_folder, ocr_strategy, tts_strategy)
            repo = FileSystemRepo(os.path.join(os.path.dirname(os.path.dirname(self.selected_folder)), 'book_chapters'))

            selected_chapters = self.get_selected_chapters()

            def log(msg):
                QThread.msleep(10)
                self.append_log(msg)

            def refresh_files():
                QThread.msleep(10)
                self.load_files()

            if "All Chapters" in selected_chapters:
                chapters = list(book.chapters.keys())
            else:
                chapters = selected_chapters

            for chapter in chapters:
                log(f"Started export for chapter: {chapter}")
                book.export_chapter(chapter, repo)
                if _cancellation_flag.is_set():
                    log(f"Cancelled during export for chapter: {chapter}")
                    return
                log(f"Finished export for chapter: {chapter}")
                refresh_files()

        self.processing_thread = ProcessingThread(export_job)
        self.processing_thread.finished.connect(self.on_export_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.cancelled.connect(self.on_processing_cancelled)
        self.processing_thread.start()

    def on_export_finished(self):
        """Called when export is complete"""
        self.load_files()
        self.set_processing_state(False)
        self.statusBar().showMessage("Export complete.")

    def append_log(self, message):
        self.logs_content.append(message)
        scrollbar = self.logs_content.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
