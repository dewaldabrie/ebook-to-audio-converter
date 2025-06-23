import sys
import os
from PySide6.QtWidgets import QListWidgetItem, QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QCheckBox, QComboBox, QLabel, QListWidget, QTextEdit, QRadioButton, QButtonGroup
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QUrl, Qt, QThread, Signal
from jpeg_to_audio import Book, EasyOCROCR, TesseractOCR, GTTSStrategy, ParlerTTSStrategy, KokoroTTSStrategy, XAITextPostProcessStrategy, FileSystemRepo

class ProcessingThread(QThread):
    """Background thread for processing jobs"""
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, processing_function, *args, **kwargs):
        super().__init__()
        self.processing_function = processing_function
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.processing_function(*self.args, **self.kwargs)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

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
        self.update_voice_selection(0)
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

        self.chapter_label = QLabel("Select Chapter:")
        left_layout.addWidget(self.chapter_label)
        self.chapter_label.setToolTip("Select a chapter to process or export.")

        self.chapter_combobox = QComboBox()
        self.chapter_combobox.addItem("All Chapters")
        left_layout.addWidget(self.chapter_combobox)
        self.chapter_combobox.setToolTip("Chapter selection.")

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

        self.file_content = QTextEdit()
        self.file_content.setReadOnly(True)
        right_layout.addWidget(self.file_content)

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
                if file.endswith(('.txt', '.mp3', '.wav')):
                    found_files.append(os.path.join(root, file))
        sorted_files = sorted(found_files)
        for file in sorted_files:
            item = QListWidgetItem(self.truncate_text(file))  # Display only last 50 characters
            item.setData(Qt.UserRole, file)  # Store full path
            self.file_list.addItem(item)

    def load_chapters(self):
        self.chapter_combobox.clear()
        self.chapter_combobox.addItem("All Chapters")
        folders = []
        for root, dirs, _ in os.walk(self.selected_folder):
            for dir in dirs:
                folders.append(dir)
        for dir in sorted(folders):
            self.chapter_combobox.addItem(dir)

    def display_file_content(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                self.file_content.setPlainText(file.read())
            self.file_content.setVisible(True)
            self.play_button.setVisible(False)
            self.stop_button.setVisible(False)
        elif file_path.endswith(('.mp3', '.wav')):
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.file_content.setVisible(False)
            self.play_button.setVisible(True)
            self.stop_button.setVisible(True)

    def set_processing_state(self, processing=True):
        """Enable/disable UI elements during processing"""
        self.select_folder_button.setEnabled(not processing)
        self.execute_button.setEnabled(not processing)
        self.export_button.setEnabled(not processing)
        
        if processing:
            self.statusBar().showMessage("Processing... Please wait.")
        else:
            self.statusBar().showMessage("Ready")

    def execute(self):
        if not hasattr(self, 'selected_folder'):
            self.folder_label.setText("Please select a folder first.")
            self.statusBar().showMessage("Please select a folder first.")
            return
        
        # Disable UI elements
        self.set_processing_state(True)
        
        # Create processing function
        def process_job():
            ocr_strategy_class = self.ocr_strategy_combobox.currentData()
            tts_strategy_class = self.tts_strategy_combobox.currentData()
            post_process_strategy_class = self.post_process_strategy_combobox.currentData()

            ocr_strategy = ocr_strategy_class(post_process_strategy=post_process_strategy_class())
            tts_strategy = tts_strategy_class()

            # If KokoroTTS is selected, set the voice
            if isinstance(tts_strategy, KokoroTTSStrategy) and not self.voice_combobox.isHidden():
                selected_voice = self.voice_combobox.currentText()
                tts_strategy.voice_name = selected_voice

            book = Book(self.selected_folder, ocr_strategy, tts_strategy)
            selected_chapter = self.chapter_combobox.currentText()

            if self.extract_text_radio.isChecked():
                if selected_chapter == "All Chapters":
                    book.extract_all_texts(overwrite=True)
                    book.save_all_texts()
                else:
                    book.extract_text(selected_chapter, overwrite=True)
                    book.save_chapter_text(selected_chapter)

            if self.rerun_post_process_radio.isChecked():
                if selected_chapter == "All Chapters":
                    book.rerun_post_processing()
                else:
                    book.rerun_post_processing(selected_chapter)

            if self.generate_audio_checkbox.isChecked():
                if selected_chapter == "All Chapters":
                    book.convert_all_texts_to_audio(overwrite=True)
                else:
                    book.convert_text_to_audio(selected_chapter, overwrite=True)

        # Start processing in background thread
        self.processing_thread = ProcessingThread(process_job)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()

    def on_processing_finished(self):
        """Called when processing is complete"""
        self.load_files()
        self.set_processing_state(False)
        self.statusBar().showMessage("Processing complete.")

    def on_processing_error(self, error_message):
        """Called when processing encounters an error"""
        self.set_processing_state(False)
        self.statusBar().showMessage(f"Error: {error_message}")

    def export(self):
        if not hasattr(self, 'selected_folder'):
            self.folder_label.setText("Please select a folder first.")
            self.statusBar().showMessage("Please select a folder first.")
            return
        
        # Disable UI elements
        self.set_processing_state(True)
        
        # Create export function
        def export_job():
            ocr_strategy_class = self.ocr_strategy_combobox.currentData()
            tts_strategy_class = self.tts_strategy_combobox.currentData()
            post_process_strategy_class = self.post_process_strategy_combobox.currentData()

            ocr_strategy = ocr_strategy_class(post_process_strategy=post_process_strategy_class())
            tts_strategy = tts_strategy_class()

            book = Book(self.selected_folder, ocr_strategy, tts_strategy)
            repo = FileSystemRepo(os.path.join(os.path.dirname(os.path.dirname(self.selected_folder)), 'book_chapters'))

            selected_chapter = self.chapter_combobox.currentText()

            if selected_chapter == "All Chapters":
                book.export_all_chapters(repo)
            else:
                book.export_chapter(selected_chapter, repo)

        # Start export in background thread
        self.processing_thread = ProcessingThread(export_job)
        self.processing_thread.finished.connect(self.on_export_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()

    def on_export_finished(self):
        """Called when export is complete"""
        self.load_files()
        self.set_processing_state(False)
        self.statusBar().showMessage("Export complete.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
