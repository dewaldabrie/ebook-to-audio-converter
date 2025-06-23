# Installation Guide

This guide provides step-by-step instructions to set up the project, including the UI and the Kokoro TTS model (included via `sys.path.append('Kokoro-82M')`), as well as configuring the xAI API key for post-processing.

## Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.8 or higher**: Download from [python.org](https://www.python.org/downloads/).
- **Git**: Download from [git-scm.com](https://git-scm.com/downloads).
- **Virtual Environment Tool**: Use `venv` (included with Python) or `conda` (available with Anaconda/Miniconda).

## Step 1: Clone the Project Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/dewaldabrie/ebook-to-audio-converter.git
cd ebook-to-audio-converter
```


## Step 2: Set Up a Virtual Environment

Create and activate a virtual environment to isolate dependencies:

### Using `venv`
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Using `conda`
```bash
conda create --name myenv python=3.8
conda activate myenv
```

## Step 3: Install Dependencies

Install the required Python packages. If a `requirements.txt` file is provided, run:

```bash
pip install -r requirements.txt
```

**Note**: The project requires NumPy version <2.0 due to compatibility issues.

## Step 4: Set Up Kokoro TTS

The Kokoro TTS model is included in the project via `sys.path.append('Kokoro-82M')`. Set it up as follows:

1. **Clone the Kokoro Repository**: Inside the project directory, clone the Kokoro repository and name it `Kokoro-82M`:

   ```bash
   git clone https://github.com/kokoro-tts/kokoro.git Kokoro-82M
   ```

2. **Verify Directory Structure**: Ensure the `Kokoro-82M` directory is in the project root alongside your main scripts.

## Step 5: Configure xAI API Key

The project uses an xAI API key for text post-processing:

1. **Obtain an xAI API Key**: Sign up at the [xAI website](https://x.ai) or follow their documentation to get an API key.

2. **Integrate the API Key**: Open the relevant script (e.g., `jpeg_to_audio.py`) and locate the API key placeholder in the `XAITextPostProcessStrategy` class:

   ```python
   "Authorization": "Bearer xai-YOUR-xAI-API-key"
   ```

   Replace the placeholder with your actual xAI API key.

   **Security Tip**: Avoid hardcoding the key. Instead, set it as an environment variable (e.g., `export XAI_API_KEY="your-key"`) and modify the code to read it:

   ```python
   import os
   "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
   ```

## Step 6: Run the UI

Launch the application UI with:

```bash
python ui.py
```

The UI should start, allowing you to use the project features.

## Troubleshooting

- **NumPy Issues**: If errors occur, ensure NumPy is <2.0:
  ```bash
  pip install "numpy<2.0"
  ```
- **Kokoro Not Found**: Verify the `Kokoro-82M` directory exists and is correctly named.
- **API Key Errors**: Check that the xAI API key is valid and correctly set.
- **Missing Packages**: Re-run the `pip install` commands if imports fail.

For additional help, refer to the project documentation or repository issues page.