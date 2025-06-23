# Ebook to Audio Converter

> **Looking to install? See [INSTALLATION.md](./INSTALLATION.md) for step-by-step setup instructions.**



## Aim of the Project

The Ebook to Audio Converter is a Python application designed to transform scanned book pages into audible content. By leveraging Optical Character Recognition (OCR) to extract text from images and Text-to-Speech (TTS) technologies to convert that text into audio, this project aims to enhance accessibility to literature. It is particularly beneficial for individuals who prefer audio formats, such as those with visual impairments or those who enjoy audiobooks during commutes or multitasking.


## Installation

For detailed installation instructions, please see [INSTALLATION.md](./INSTALLATION.md).

## Main Components

The application is built with a modular architecture, comprising the following key components:

- **OCR Strategies**:

  - `EasyOCROCR`: Uses the EasyOCR library for efficient text extraction from images.
  - `TesseractOCR`: Employs the Tesseract OCR engine, enhanced with image preprocessing for improved accuracy.

- **TTS Strategies**:

  - `GTTSStrategy`: Utilizes Google Text-to-Speech for reliable and natural-sounding audio output.
  - `ParlerTTSStrategy`: Integrates the Parler TTS model, offering customizable voice synthesis.
  - `KokoroTTSStrategy`: Implements the Kokoro TTS model with multiple voice options for varied narration styles.

- **Post-Processing**:

  - `XAITextPostProcessStrategy`: Refines OCR-extracted text by correcting errors (e.g., joined words, spelling mistakes) and formatting it for better TTS compatibility, using an AI-driven API.

- **Repository**:

  - `FileSystemRepo`: Manages the storage of extracted text and generated audio files in a structured filesystem.

- **User Interface**:

  - A PySide6-based graphical user interface (GUI) that provides an intuitive way to configure and control the conversion process.

## Summary of Features

- **Folder Selection**: Select a directory containing scanned book pages (JPEG/JPG files).
- **Strategy Customization**: Choose from available OCR and TTS strategies, with voice selection for Kokoro TTS.
- **Chapter Processing**: Process all chapters or a specific chapter within the selected folder.
- **Flexible Actions**: Extract text, generate audio, rerun post-processing, or skip text extraction as needed.
- **Export Capabilities**: Save processed text and audio files to a designated repository.
- **Interactive UI**: View extracted text, play generated audio, and manage files directly from the interface.

## How Everything Comes Together in the UI

The PySide6-based UI ties all components into a cohesive and user-friendly experience:

1. **Folder Selection**:

   - Users begin by clicking "Select Folder" to choose a directory with scanned book pages. The folder path is displayed with truncation for readability.

2. **Configuration Options**:

   - **Text Extraction Options**: Radio buttons allow users to choose "Extract Text," "Rerun Post-Processing," or "Skip Text Extraction."
   - **Audio Generation**: A checkbox enables audio generation.
   - **Strategy Selection**: Dropdown menus offer choices for OCR (EasyOCR, Tesseract), TTS (Kokoro, GTTS, Parler), and post-processing (XAI) strategies. For Kokoro TTS, a dynamic voice selection dropdown appears.
   - **Chapter Selection**: A dropdown lists "All Chapters" or specific chapter folders detected in the directory.

3. **Execution**:

   - The "Execute" button triggers the selected actions (text extraction, post-processing, audio generation) based on the chosen chapter and strategies. Multiprocessing ensures efficient handling of multiple pages.

4. **File Interaction**:

   - A list widget displays processed files (text and audio). Clicking a text file shows its content in a read-only text area, while selecting an audio file enables "Play" and "Stop" buttons for playback using QMediaPlayer.

5. **Export**:

   - The "Export" button saves processed chapters to a `FileSystemRepo`, organizing text and audio in a structured directory.

The UI abstracts the complexity of OCR, TTS, and file management, providing real-time feedback and control over the conversion process.

## Rationale for the Implementation

This implementation stands out for several reasons:

- **Modular Design**: The use of strategy patterns for OCR, TTS, and post-processing ensures components are interchangeable and extensible. New strategies can be added without altering the core application.
- **User-Centric UI**: The PySide6 GUI simplifies interaction, making the tool accessible to non-technical users while offering flexibility for advanced customization.
- **Performance Efficiency**: Multiprocessing accelerates the processing of multiple pages or chapters, leveraging CPU cores for faster execution.
- **Quality Focus**: Post-processing with XAI corrects OCR errors and optimizes text for TTS, ensuring high-quality audio output.

This balance of flexibility, usability, and performance makes the application both practical and scalable.

## Future Work

- **Abbreviated Chapters**: Add functionality to generate concise summaries or abbreviated versions of chapters, ideal for quick listens or previews.
- **Expanded TTS Options**: Incorporate additional voices and languages for greater personalization and accessibility.
- **Cloud Storage Integration**: Enable saving and retrieving files from cloud services like Google Drive or Dropbox for seamless access across devices.