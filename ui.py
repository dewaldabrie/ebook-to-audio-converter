import sys
import os
import threading
import signal
import psutil
from PySide6.QtWidgets import QListWidgetItem, QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QCheckBox, QComboBox, QLabel, QListWidget, QTextEdit, QRadioButton, QButtonGroup, QTabWidget, QAbstractItemView, QLineEdit, QMessageBox
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QUrl, Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIntValidator
from jpeg_to_audio import Book, EasyOCROCR, TesseractOCR, GTTSStrategy, ParlerTTSStrategy, KokoroTTSStrategy, XAITextPostProcessStrategy, FileSystemRepo, NPagePerChapterSummaryStrategy, NPagePerBookSummaryStrategy
import traceback
import json
import os.path

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
            global _cancellation_flag
            _cancellation_flag.clear()
            self.processing_function(*self.args, **self.kwargs)
            if not _cancellation_flag.is_set():
                self.finished.emit()
            else:
                self.cancelled.emit()
        except Exception as e:
            tb = traceback.format_exc()
            if not _cancellation_flag.is_set():
                self.error.emit(f"{e}\n{tb}")
            else:
                self.cancelled.emit()
        finally:
            if self._cancelled:
                self._kill_child_processes()

class SafeTextEdit(QTextEdit):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window

    def keyPressEvent(self, event):
        if self.isReadOnly() and self.main_window and self.main_window.processing_thread and self.main_window.processing_thread.isRunning():
            QMessageBox.warning(self, "Editing Disabled", "Editing is disabled while processing is running.")
            return
        super().keyPressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ebook to Audio Converter")
        
        # Load last used folder and settings
        self.config_file = os.path.join(os.path.expanduser("~"), ".ebook_to_audio_config.json")
        self.selected_folder = self.load_last_folder()

        # Status bar for messages
        self.statusBar().showMessage("Ready")
        
        # Processing thread
        self.processing_thread = None

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.folder_label = QLabel("Select folder with scanned book:")
        left_layout.addWidget(self.folder_label)

        self.select_folder_button = QPushButton("Select Folder")
        self.select_folder_button.setToolTip("Choose the root folder containing subfolders for each chapter of the book. Each subfolder should have the scanned pages as images (JPEG/JPG).")
        self.select_folder_button.clicked.connect(self.select_folder)
        left_layout.addWidget(self.select_folder_button)

        # Remove highlighting and set focus to "Select Folder" button
        self.select_folder_button.setStyleSheet("")  # Remove any custom highlight

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

        # --- OCR Strategy ---
        self.ocr_strategy_label = QLabel("Select OCR Strategy:")
        left_layout.addWidget(self.ocr_strategy_label)
        self.ocr_strategy_label.setToolTip("Choose the OCR engine for text extraction.")

        self.ocr_strategy_combobox = QComboBox()
        self.ocr_strategy_combobox.addItem("EasyOCR", EasyOCROCR)
        self.ocr_strategy_combobox.addItem("TesseractOCR", TesseractOCR)
        left_layout.addWidget(self.ocr_strategy_combobox)
        self.ocr_strategy_combobox.setToolTip("OCR engine selection.")

        # --- Post-Process Strategy ---
        self.post_process_strategy_label = QLabel("Select Post-Process Strategy:")
        left_layout.addWidget(self.post_process_strategy_label)
        self.post_process_strategy_label.setToolTip("Choose the post-processing strategy for OCR text.")

        self.post_process_strategy_combobox = QComboBox()
        self.post_process_strategy_combobox.addItem("XAITextPostProcess", XAITextPostProcessStrategy)
        left_layout.addWidget(self.post_process_strategy_combobox)
        self.post_process_strategy_combobox.setToolTip("Post-processing strategy selection.")
        
        # --- Add summary checkbox ---
        self.create_summary_checkbox = QCheckBox("Create Summary Version of Chapter(s)")
        left_layout.addWidget(self.create_summary_checkbox)
        self.create_summary_checkbox.setToolTip("Generate a summarized version of each chapter or book using the Grok API.")

        # --- Summary strategy selection ---
        self.summary_label = QLabel("Select Summarisation Strategy:")
        left_layout.addWidget(self.summary_label)
        self.summary_label.setToolTip("Choose how to best summarise this work.")
        self.summary_strategy_combobox = QComboBox()
        self.summary_strategy_combobox.addItem("N pages per chapter", NPagePerChapterSummaryStrategy)
        self.summary_strategy_combobox.addItem("N pages per book", NPagePerBookSummaryStrategy)
        left_layout.addWidget(self.summary_strategy_combobox)
        self.summary_strategy_combobox.setToolTip("Choose summary strategy.")

        self.n_pages_chapter_input = QLineEdit("1")
        self.n_pages_chapter_input.setValidator(QIntValidator(1, 20))
        self.n_pages_chapter_input.setToolTip("Number of pages per chapter for summary.")
        left_layout.addWidget(self.n_pages_chapter_input)

        self.n_pages_book_input = QLineEdit("2")
        self.n_pages_book_input.setValidator(QIntValidator(1, 20))
        self.n_pages_book_input.setToolTip("Number of pages for book summary.")
        left_layout.addWidget(self.n_pages_book_input)

        self.generate_audio_checkbox = QCheckBox("Generate Audio")
        left_layout.addWidget(self.generate_audio_checkbox)
        self.generate_audio_checkbox.setToolTip("Generate audio from extracted text using the selected TTS strategy.")
        
        # --- TTS Strategy ---
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
        self.tts_strategy_combobox.currentIndexChanged.connect(self.update_voice_selection)
        self.voice_combobox.setToolTip("Select a voice for Kokoro TTS.")

        # --- Add style translation checkbox and dropdown ---
        self.style_translation_checkbox = QCheckBox("Style Translation")
        left_layout.addWidget(self.style_translation_checkbox)
        self.style_translation_checkbox.setToolTip("Translate the style of summaries and post-processing.")

        self.style_translation_combobox = QComboBox()
        self.style_translation_combobox.addItems([
            "Shakespearean",
            "Hippy 90s",
            "Flowerly, descriptive with vivid imagery",
            "Standup Comedy",
            "Everything Rhymes",
            "Academic, cold and sanitised of any personality",
            "Custom"
        ])
        self.style_translation_combobox.setVisible(False)
        left_layout.addWidget(self.style_translation_combobox)

        self.custom_style_input = QLineEdit()
        self.custom_style_input.setPlaceholderText("Enter custom style/theme...")
        self.custom_style_input.setVisible(False)
        left_layout.addWidget(self.custom_style_input)

        # --- Only apply style translation to summary option ---
        self.only_style_summary_checkbox = QCheckBox("Only apply style translation to summary")
        self.only_style_summary_checkbox.setVisible(False)
        left_layout.addWidget(self.only_style_summary_checkbox)

        def should_show_only_style_summary():
            # Show if both summary and style translation are checked
            if self.create_summary_checkbox.isChecked() and self.style_translation_checkbox.isChecked():
                return True
            # Or if style translation is checked and any *_summary.txt files exist
            if self.style_translation_checkbox.isChecked():
                if hasattr(self, 'selected_folder'):
                    for root, _, files in os.walk(self.selected_folder):
                        if any(f.endswith('_summary.txt') for f in files):
                            return True
            return False

        def update_style_inputs():
            show = self.style_translation_checkbox.isChecked()
            self.style_translation_combobox.setVisible(show)
            self.custom_style_input.setVisible(show and self.style_translation_combobox.currentText() == "Custom")
            # Dynamically show the "only style summary" checkbox
            self.only_style_summary_checkbox.setVisible(should_show_only_style_summary())
        self.style_translation_checkbox.stateChanged.connect(update_style_inputs)
        self.style_translation_combobox.currentIndexChanged.connect(update_style_inputs)
        self.create_summary_checkbox.stateChanged.connect(update_style_inputs)
        self.select_folder_button.clicked.connect(update_style_inputs)
        update_style_inputs()

        # --- UI visibility logic ---
        def update_dynamic_visibility():
            # 1. OCR strategy
            show_ocr = self.extract_text_radio.isChecked()
            self.ocr_strategy_label.setVisible(show_ocr)
            self.ocr_strategy_combobox.setVisible(show_ocr)

            # 2. TTS strategy
            show_tts = self.generate_audio_checkbox.isChecked()
            self.tts_strategy_label.setVisible(show_tts)
            self.tts_strategy_combobox.setVisible(show_tts)
            self.voice_combobox.setVisible(show_tts and self.tts_strategy_combobox.currentData() == KokoroTTSStrategy)

            # 3. Post-process strategy
            show_post = self.extract_text_radio.isChecked() or self.rerun_post_process_radio.isChecked()
            self.post_process_strategy_label.setVisible(show_post)
            self.post_process_strategy_combobox.setVisible(show_post)

            # 4. Summary strategy
            show_summary = self.create_summary_checkbox.isChecked()
            self.summary_label.setVisible(show_summary)
            self.summary_strategy_combobox.setVisible(show_summary)
            self.n_pages_chapter_input.setVisible(show_summary and self.summary_strategy_combobox.currentIndex() in [0, 1])
            self.n_pages_book_input.setVisible(show_summary and self.summary_strategy_combobox.currentIndex() == 1)

        # Connect all relevant signals
        self.extract_text_radio.toggled.connect(update_dynamic_visibility)
        self.rerun_post_process_radio.toggled.connect(update_dynamic_visibility)
        self.skip_text_extraction.toggled.connect(update_dynamic_visibility)
        self.generate_audio_checkbox.toggled.connect(update_dynamic_visibility)
        self.tts_strategy_combobox.currentIndexChanged.connect(update_dynamic_visibility)
        self.create_summary_checkbox.toggled.connect(update_dynamic_visibility)
        self.summary_strategy_combobox.currentIndexChanged.connect(update_dynamic_visibility)
        update_dynamic_visibility()

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
        self.file_content = SafeTextEdit(main_window=self)
        self.file_content.setReadOnly(False)
        self.logs_content = QTextEdit()
        self.logs_content.setReadOnly(True)
        self.tabs.addTab(self.file_content, "File Content")
        self.tabs.addTab(self.logs_content, "Logs")
        right_layout.addWidget(self.tabs)
        # ------------------------------------------------

        # --- Add Save button for file content ---
        self.save_file_button = QPushButton("Save")
        self.save_file_button.setVisible(False)
        self.save_file_button.setToolTip("Save changes to the current text file.")
        self.save_file_button.clicked.connect(self.save_current_file)
        right_layout.addWidget(self.save_file_button)
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

        # Now that all UI elements are created, restore the last settings
        self.restore_last_settings()

        self.select_folder_button.setFocus()

    def load_last_folder(self):
        """Load the last used folder from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('last_folder', None)
        except Exception as e:
            print(f"Error loading config: {e}")
        return None

    def load_last_settings(self):
        """Load all last used settings from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                    # Restore text extraction option
                    text_extraction_option = config.get('text_extraction_option', 'extract_text')
                    if text_extraction_option == 'extract_text':
                        self.extract_text_radio.setChecked(True)
                    elif text_extraction_option == 'rerun_post_process':
                        self.rerun_post_process_radio.setChecked(True)
                    elif text_extraction_option == 'skip_text_extraction':
                        self.skip_text_extraction.setChecked(True)
                    
                    # Restore OCR strategy
                    ocr_strategy = config.get('ocr_strategy', 'EasyOCR')
                    for i in range(self.ocr_strategy_combobox.count()):
                        if self.ocr_strategy_combobox.itemText(i) == ocr_strategy:
                            self.ocr_strategy_combobox.setCurrentIndex(i)
                            break
                    
                    # Restore TTS strategy
                    tts_strategy = config.get('tts_strategy', 'KokoroTTS')
                    for i in range(self.tts_strategy_combobox.count()):
                        if self.tts_strategy_combobox.itemText(i) == tts_strategy:
                            self.tts_strategy_combobox.setCurrentIndex(i)
                            break
                    
                    # Restore voice selection
                    voice = config.get('voice', 'bm_george')
                    for i in range(self.voice_combobox.count()):
                        if self.voice_combobox.itemText(i) == voice:
                            self.voice_combobox.setCurrentIndex(i)
                            break
                    
                    # Restore summary settings
                    self.create_summary_checkbox.setChecked(config.get('create_summary', False))
                    
                    summary_strategy = config.get('summary_strategy', 'N pages per chapter')
                    for i in range(self.summary_strategy_combobox.count()):
                        if self.summary_strategy_combobox.itemText(i) == summary_strategy:
                            self.summary_strategy_combobox.setCurrentIndex(i)
                            break
                    
                    self.n_pages_chapter_input.setText(str(config.get('n_pages_chapter', 1)))
                    self.n_pages_book_input.setText(str(config.get('n_pages_book', 2)))
                    
                    # Restore audio generation setting
                    self.generate_audio_checkbox.setChecked(config.get('generate_audio', False))
                    
                    # Restore style translation settings
                    self.style_translation_checkbox.setChecked(config.get('style_translation_enabled', False))
                    
                    style_translation_type = config.get('style_translation_type', 'Shakespearean')
                    for i in range(self.style_translation_combobox.count()):
                        if self.style_translation_combobox.itemText(i) == style_translation_type:
                            self.style_translation_combobox.setCurrentIndex(i)
                            break
                    
                    self.custom_style_input.setText(config.get('custom_style', ''))
                    self.only_style_summary_checkbox.setChecked(config.get('only_style_summary', False))
                    
                    # Restore folder and load files if folder exists
                    if self.selected_folder and os.path.exists(self.selected_folder):
                        self.folder_label.setText(f"Selected folder:\n{self.truncate_text(self.selected_folder)}")
                        self.load_files()
                        self.load_chapters()
                        self.statusBar().showMessage("Last settings restored.")
                    
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save_last_settings(self):
        """Save all current settings to config file"""
        try:
            config = {
                'last_folder': self.selected_folder,
                'text_extraction_option': 'extract_text' if self.extract_text_radio.isChecked() else 
                                        'rerun_post_process' if self.rerun_post_process_radio.isChecked() else 
                                        'skip_text_extraction',
                'ocr_strategy': self.ocr_strategy_combobox.currentText(),
                'tts_strategy': self.tts_strategy_combobox.currentText(),
                'voice': self.voice_combobox.currentText() if not self.voice_combobox.isHidden() else 'bm_george',
                'create_summary': self.create_summary_checkbox.isChecked(),
                'summary_strategy': self.summary_strategy_combobox.currentText(),
                'n_pages_chapter': int(self.n_pages_chapter_input.text()),
                'n_pages_book': int(self.n_pages_book_input.text()),
                'generate_audio': self.generate_audio_checkbox.isChecked(),
                'style_translation_enabled': self.style_translation_checkbox.isChecked(),
                'style_translation_type': self.style_translation_combobox.currentText(),
                'custom_style': self.custom_style_input.text(),
                'only_style_summary': self.only_style_summary_checkbox.isChecked()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def save_last_folder(self, folder_path):
        """Save the selected folder to config file"""
        self.selected_folder = folder_path
        self.save_last_settings()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.selected_folder = folder
            self.folder_label.setText(f"Selected folder:\n{self.truncate_text(folder)}")
            self.load_files()
            self.load_chapters()
            self.save_last_settings()  # Save all settings including the new folder
            self.statusBar().showMessage("Folder loaded.")
        else:
            self.statusBar().showMessage("No folder selected.")

    def restore_last_settings(self):
        """Restore the last used settings after all UI elements are initialized"""
        self.load_last_settings()

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
        
        # Save settings when voice selection changes
        self.save_last_settings()

    def closeEvent(self, event):
        """Save settings when the application is closed"""
        self.save_last_settings()
        event.accept()

    # Connect all setting changes to save function
    def connect_setting_changes(self):
        """Connect all setting changes to automatically save"""
        self.extract_text_radio.toggled.connect(self.save_last_settings)
        self.rerun_post_process_radio.toggled.connect(self.save_last_settings)
        self.skip_text_extraction.toggled.connect(self.save_last_settings)
        self.ocr_strategy_combobox.currentIndexChanged.connect(self.save_last_settings)
        self.tts_strategy_combobox.currentIndexChanged.connect(self.save_last_settings)
        self.create_summary_checkbox.stateChanged.connect(self.save_last_settings)
        self.summary_strategy_combobox.currentIndexChanged.connect(self.save_last_settings)
        self.n_pages_chapter_input.textChanged.connect(self.save_last_settings)
        self.n_pages_book_input.textChanged.connect(self.save_last_settings)
        self.generate_audio_checkbox.stateChanged.connect(self.save_last_settings)
        self.style_translation_checkbox.stateChanged.connect(self.save_last_settings)
        self.style_translation_combobox.currentIndexChanged.connect(self.save_last_settings)
        self.custom_style_input.textChanged.connect(self.save_last_settings)
        self.only_style_summary_checkbox.stateChanged.connect(self.save_last_settings)

    @staticmethod
    def truncate_text(text, length=50):
        if len(text) >= length:
            text = '...' + text[max(0, len(text) - length):]
        return text

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
        self.current_file_path = file_path  # Track the currently displayed file
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                self.file_content.setPlainText(file.read())
            self.file_content.setVisible(True)
            self.save_file_button.setVisible(True)  # Show Save button for text files
            self.play_button.setVisible(False)
            self.stop_button.setVisible(False)
            self.tabs.setCurrentWidget(self.file_content)  # Switch to file content tab
        elif file_path.endswith(('.mp3', '.wav')):
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.file_content.setVisible(False)
            self.save_file_button.setVisible(False)  # Hide Save button for audio files
            self.play_button.setVisible(True)
            self.stop_button.setVisible(True)
            self.tabs.setCurrentWidget(self.file_content)  # Still switch to file content tab

    def save_current_file(self):
        """Save the contents of the file_content pane to the current file."""
        if hasattr(self, 'current_file_path') and self.current_file_path.endswith('.txt'):
            try:
                with open(self.current_file_path, 'w') as file:
                    file.write(self.file_content.toPlainText())
                self.statusBar().showMessage(f"Saved: {self.current_file_path}", 3000)
            except Exception as e:
                self.statusBar().showMessage(f"Error saving file: {e}", 5000)
        else:
            self.statusBar().showMessage("No text file selected to save.", 3000)

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
            self.file_content.setReadOnly(processing)
            self.save_file_button.setEnabled(not processing)
        else:
            # Change cancel button back to execute button
            self.execute_button.setText("Execute")
            self.execute_button.clicked.disconnect()
            self.execute_button.clicked.connect(self.execute)
            self.execute_button.setToolTip("Run the selected actions (text extraction, post-processing, audio generation).")
            self.execute_button.setStyleSheet("")  # Reset to default style
            
            self.statusBar().showMessage("Ready")
            self.file_content.setReadOnly(processing)
            self.save_file_button.setEnabled(not processing)

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

            # Always ensure a valid TTS strategy is selected if audio generation is requested
            if self.generate_audio_checkbox.isChecked() and tts_strategy_class is None:
                # Fallback to the first available TTS strategy
                self.tts_strategy_combobox.setCurrentIndex(0)
                tts_strategy_class = self.tts_strategy_combobox.currentData()
            tts_strategy = tts_strategy_class() if tts_strategy_class else None

            if isinstance(tts_strategy, KokoroTTSStrategy) and not self.voice_combobox.isHidden():
                selected_voice = self.voice_combobox.currentText()
                tts_strategy.voice_name = selected_voice

            summary_strategy_class = self.summary_strategy_combobox.currentData()
            n_pages_chapter = int(self.n_pages_chapter_input.text())
            n_pages_book = int(self.n_pages_book_input.text())

            # Gather style translation info
            style_translation = None
            only_style_summary = False
            if self.style_translation_checkbox.isChecked():
                style_translation = self.style_translation_combobox.currentText()
                if style_translation == "Custom":
                    style_translation = self.custom_style_input.text().strip()
                    if not style_translation:
                        style_translation = None
                only_style_summary = self.only_style_summary_checkbox.isChecked()

            book = Book(
                self.selected_folder, ocr_strategy, tts_strategy,
                summary_strategy=summary_strategy_class(),
                summary_n_pages=n_pages_chapter,
                book_summary_n_pages=n_pages_book,
                style_translation=style_translation,
                only_style_summary=only_style_summary
            )

            selected_chapters = self.get_selected_chapters()

            def log(msg):
                QThread.msleep(10)
                self.append_log(msg)

            def refresh_files():
                QTimer.singleShot(0, self.load_files)

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

                # --- Summary generation ---
                if self.create_summary_checkbox.isChecked():
                    log(f"Started summary generation for chapter: {chapter}")
                    try:
                        summary_path = book.summarize_chapter(chapter)
                        log(f"Summary saved to {summary_path}")
                        refresh_files()
                    except Exception as e:
                        log(f"Error during summary generation for chapter {chapter}: {e}")
                        continue

                if self.generate_audio_checkbox.isChecked():
                    # Robustly find the chapter object
                    chapter_obj = None
                    for key in book.chapters:
                        if key.strip().lower() == chapter.strip().lower():
                            chapter_obj = book.chapters[key]
                            break
                    if not chapter_obj:
                        log(f"Cannot generate audio for chapter: '{chapter}' (not found in book.chapters: {list(book.chapters.keys())})")
                        continue

                    text_path = os.path.join(chapter_obj.folder_path, f"{chapter_obj.title}.txt")
                    log(f"Checking for text file: {text_path}")
                    if not os.path.exists(text_path):
                        log(f"Cannot generate audio for chapter: {chapter} because text file does not exist at {text_path}. Run text extraction first.")
                        continue

                    log(f"Started audio generation for chapter: {chapter}")
                    book.convert_text_to_audio(chapter, overwrite=True)
                    if _cancellation_flag.is_set():
                        log(f"Cancelled during audio generation for chapter: {chapter}")
                        return
                    log(f"Finished audio generation for chapter: {chapter}")
                    refresh_files()
                    # --- Generate audio for summary if present ---
                    if self.create_summary_checkbox.isChecked():
                        log(f"Started audio generation for summary of chapter: {chapter}")
                        try:
                            summary_audio_path = book.convert_summary_to_audio(chapter, overwrite=True)
                            log(f"Summary audio saved to {summary_audio_path}")
                            refresh_files()
                        except Exception as e:
                            log(f"Error during summary audio generation for chapter {chapter}: {e}")
                            continue

        self.processing_thread = ProcessingThread(process_job)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.cancelled.connect(self.on_processing_cancelled)
        self.processing_thread.start()

    def on_processing_finished(self):
        """Called when processing is complete"""
        self.load_files()
        self.reload_current_file()
        self.set_processing_state(False)
        self.statusBar().showMessage("Processing complete.")

    def on_processing_cancelled(self):
        """Called when processing is cancelled"""
        self.set_processing_state(False)
        self.statusBar().showMessage("Processing cancelled.")

    def on_processing_error(self, error_message):
        """Called when processing encounters an error"""
        self.set_processing_state(False)
        self.append_log(f"Error: {error_message}")
        self.statusBar().showMessage(f"Error: {error_message}", 5000)
        self.statusBar().repaint()

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

            # Always ensure a valid TTS strategy is selected if audio generation is requested
            if self.generate_audio_checkbox.isChecked() and tts_strategy_class is None:
                # Fallback to the first available TTS strategy
                self.tts_strategy_combobox.setCurrentIndex(0)
                tts_strategy_class = self.tts_strategy_combobox.currentData()
            tts_strategy = tts_strategy_class() if tts_strategy_class else None

            summary_strategy_class = self.summary_strategy_combobox.currentData()
            n_pages_chapter = int(self.n_pages_chapter_input.text())
            n_pages_book = int(self.n_pages_book_input.text())

            # Gather style translation info
            style_translation = None
            only_style_summary = False
            if self.style_translation_checkbox.isChecked():
                style_translation = self.style_translation_combobox.currentText()
                if style_translation == "Custom":
                    style_translation = self.custom_style_input.text().strip()
                    if not style_translation:
                        style_translation = None
                only_style_summary = self.only_style_summary_checkbox.isChecked()

            book = Book(
                self.selected_folder, ocr_strategy, tts_strategy,
                summary_strategy=summary_strategy_class(),
                summary_n_pages=n_pages_chapter,
                book_summary_n_pages=n_pages_book,
                style_translation=style_translation,
                only_style_summary=only_style_summary
            )
            repo = FileSystemRepo(os.path.join(os.path.dirname(os.path.dirname(self.selected_folder)), 'audio_books'))

            selected_chapters = self.get_selected_chapters()

            def log(msg):
                QThread.msleep(10)
                self.append_log(msg)

            def refresh_files():
                QTimer.singleShot(0, self.load_files)

            if "All Chapters" in selected_chapters:
                chapters = list(book.chapters.keys())
            else:
                chapters = selected_chapters

            for chapter in chapters:
                log(f"Started export for chapter: {chapter}")
                book.export_chapter(chapter, repo, include_summary=True)
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
        self.reload_current_file()
        self.set_processing_state(False)
        self.statusBar().showMessage("Export complete.")

    def append_log(self, message):
        self.logs_content.append(message)
        scrollbar = self.logs_content.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def reload_current_file(self):
        """Reload the file currently open in the editor, if it still exists and is a text file."""
        if hasattr(self, 'current_file_path') and self.current_file_path and self.current_file_path.endswith('.txt'):
            try:
                with open(self.current_file_path, 'r') as file:
                    self.file_content.setPlainText(file.read())
            except Exception as e:
                self.statusBar().showMessage(f"Error reloading file: {e}", 5000)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
