import os
import shutil
import ssl
import certifi
from pprint import pformat
from typing import Dict, List

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import wave
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
import logging
import cv2
import numpy as np
from gtts import gTTS
import requests
import soundfile as sf
import time

# Kokoro
import sys
import subprocess
import threading
import signal
import psutil

# Add this global variable
_cancellation_flag = threading.Event()

def ensure_kokoro_installed():
    """
    Ensure the Kokoro TTS model is present in the project.
    If not, clone it from the official repository.
    """
    kokoro_dir = 'Kokoro-82M'
    if not os.path.isdir(kokoro_dir):
        print(f"[INFO] '{kokoro_dir}' not found. Cloning Kokoro repository...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/kokoro-tts/kokoro.git",
                    kokoro_dir
                ],
                check=True
            )
            print(f"[INFO] Kokoro repository cloned successfully into '{kokoro_dir}'.")
        except Exception as e:
            print(f"[ERROR] Failed to clone Kokoro repository: {e}")
            raise

# Check NumPy version
if int(np.__version__.split('.')[0]) >= 2:
    raise ImportError("This module is not compatible with NumPy 2.x. Please downgrade to NumPy 1.x.")

from typing import Dict

logging.basicConfig(level=logging.DEBUG)
from multiprocessing import Pool, cpu_count, get_context

# Add this function to handle graceful shutdown
def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    global _cancellation_flag
    _cancellation_flag.set()
    # Exit the process
    os._exit(0)

# Set up signal handlers for graceful termination
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

class PostProcessStrategy:
    def post_process_text(self, text, meta: Dict = None):
        raise NotImplementedError("Post-process strategy must implement the post_process_text method")

class XAITextPostProcessStrategy(PostProcessStrategy):
    def post_process_text(self, text, meta=None):
        """
        Post-process OCR text using the xAI API.
        The API key is loaded from the environment variable XAI_API_KEY.
        """
        if not text:
            return text
        
        # Check for cancellation before making API call
        if _cancellation_flag.is_set():
            return text
            
        logging.info("Post-processing text")
        url = "https://api.x.ai/v1/chat/completions"
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            logging.error("XAI_API_KEY not set in environment variables.")
            return text
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a book digitization assistant."
                },
                {
                    "role": "user",
                    "content": f"""
                    Fix any errors in the OCR result for a scanned chapter below. 
                    Correct common OCR errors and also issues like 
                      1) page header being included on the start of every page 
                         (which corresponds to a line in the text below). 
                         These headers are typically all caps and not part of the main text.
                         They break the flow of the sentence and is typically repeated multiple times
                         throughout the chapter.
                      2) adjacent words joined because spaces are missed 
                      3) spelling mistakes
                      4) scripture reference can be terse like "Is. 1:1" which doesn't vocalise well. Should rather be "Isaiah chapter one, verse one"
                    and any other likely mistakes. The input is the result of a EasyOCR model and the result will be used 
                    directly in a TTS model, so fix up anything that a human wouuldn't want to hear in a narrated book. 
                    Reply with nothing other than the corrected text.

                    Avoid accentuating chapter headings with asterisk, since the TTS model vocalised the asterisk symbols.
                    Just put the headings on their own line with a double line break before and after.
                    
                    Metadata for the text that folllows: 
                    {pformat(meta)}
                    """
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "model": "grok-3-mini",
            "stream": False,
            "temperature": 0
        }
        
        # Check for cancellation before making the request
        if _cancellation_flag.is_set():
            return text
            
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                corrected = response.json().get('choices', [{}])[0].get('message', {}).get('content', text)
                return corrected
            else:
                logging.error(f"Failed to post-process text: {response.status_code} {response.text}")
                return text
        except requests.exceptions.Timeout:
            logging.error("xAI API request timed out. Try increasing the timeout or check your network/API status.")
            return text
        except requests.exceptions.RequestException as e:
            if _cancellation_flag.is_set():
                return text
            logging.error(f"Request failed: {e}")
            return text

class OCRStrategy:
    def __init__(self, post_process_strategy: PostProcessStrategy = XAITextPostProcessStrategy()):
        self.post_process_strategy = post_process_strategy

    def preprocess_image(self, image_path):
        raise NotImplementedError("OCR strategy must implement the preprocess_image method")

    def extract_text(self, image_path):
        raise NotImplementedError("OCR strategy must implement the extract_text method")

    def post_process_text(self, text, meta=None):
        return self.post_process_strategy.post_process_text(text, meta=meta)

class TesseractOCR(OCRStrategy):
    def preprocess_image(self, image_path):
        from PIL import Image, ImageEnhance, ImageOps  # Local import
        import cv2
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        # Convert to grayscale
        image = image.convert('L')
        # Increase contrast
        image = ImageEnhance.Contrast(image).enhance(2)
        enhanced_path = image_path.replace('.jpeg', '_enhanced.jpeg')
        image.save(enhanced_path)
        # Convert PIL image to numpy array
        img = cv2.imread(enhanced_path)
        gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        opn = cv2.morphologyEx(gry, cv2.MORPH_OPEN, None)
        # Save the processed image
        from PIL import Image as PILImage
        PILImage.fromarray(opn).save(image_path)
        return

    def extract_text(self, image_path):
        from PIL import Image  # Local import
        import pytesseract
        image = Image.open(image_path)
        return pytesseract.image_to_string(image, lang='eng')

class EasyOCROCR(OCRStrategy):
    def __init__(self, post_process_strategy: PostProcessStrategy = XAITextPostProcessStrategy()):
        super().__init__(post_process_strategy)
        self.reader = None  # Lazy initialization

    def preprocess_image(self, image_path):
        from PIL import Image, ImageOps  # Local import
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image.save(image_path)

    def extract_text(self, image_path):
        if self.reader is None:
            import easyocr  # Lazy import
            self.reader = easyocr.Reader(['en'], gpu=False)
        result = self.reader.readtext(image_path, detail=0)
        return ' '.join(result)

import pydub
import nltk

class TTSStrategy:
    def convert_text_to_audio(self, text, output_path):
        raise NotImplementedError("TTS strategy must implement the convert_text_to_audio method")

    @staticmethod
    def split_text_into_chunks(text, max_length=500):
        try:
            nltk.data.find('tokenizers/punkt_tah')
        except LookupError:
            nltk.download('punkt')
        sentences = nltk.tokenize.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += sentence + ' '
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ' '
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    @staticmethod
    def combine_audio_chunks(audio_chunks: List[str], output_path):
        # Read all files to gather data
        all_audio_data = []
        total_frames = 0
        params = None

        for chunk in audio_chunks:
            with wave.open(chunk, 'rb') as wf:
                if params is None:
                    params = list(wf.getparams())
                    logging.debug(f"Initial params: {wf.getparams()}")
                    params[3] = 0  # Set nframes to 0 initially
                else:
                    assert params[:3] == list(wf.getparams())[:3], "All files must have the same audio parameters except nframes"
                
                frames = wf.readframes(wf.getnframes())
                all_audio_data.append(np.frombuffer(frames, dtype=np.int16))
                total_frames += wf.getnframes()

        # Write the concatenated file
        params[3] = total_frames  # Update nframes
        logging.debug(f"Writing params: {params}")
        logging.debug(f"Writing {len(all_audio_data)} frames to output:")
        output_path = output_path + '.wav'
        with wave.open(output_path, 'wb') as output:
            output.setparams(tuple(params))
            logging.debug(f"Resulting params: {output.getparams()}")
            for data in all_audio_data:
                output.writeframes(data.tobytes())
        logging.debug(f"Concatenated audio written to {output_path}")

        # Convert the concatenated .wav file to .mp3
        sound = pydub.AudioSegment.from_wav(output_path)
        mp3_output_path = output_path.replace('.wav', '.mp3')
        sound.export(mp3_output_path, format="mp3")
        # Explicitly delete the AudioSegment object and wait before removing the file
        del sound
        time.sleep(0.2)  # Give the OS time to release the file handle
        try:
            os.remove(output_path)
        except Exception as e:
            logging.warning(f"Could not delete wav file {output_path}: {e}")
        return mp3_output_path
        

class GTTSStrategy(TTSStrategy):
    def convert_text_to_audio(self, text, output_path):
        chunks = self.split_text_into_chunks(text)
        with Pool(cpu_count()) as pool:
            audio_chunks = list(pool.imap(self._process_chunk, [(chunk, output_path, i, len(chunks)) for i, chunk in enumerate(chunks)]))
        self.combine_audio_chunks(audio_chunks, output_path)

    def _process_chunk(self, params):
        chunk, output_path, i, total_chunks = params
        chunk_path = f"{output_path}_chunk_{i}.mp3"
        with open(f"{output_path}_chunk_{i}.txt", 'w') as f:
            f.write(chunk)
        logging.info(f"Saving chunk {i+1}/{total_chunks} to {chunk_path}")
        tts = gTTS(chunk)
        tts.save(chunk_path)
        return chunk_path


class ParlerTTSStrategy(TTSStrategy):
    def __init__(self):
        self.model = None

    def init_worker(self, device):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        global tts_model
        tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
        global tts_tokenizer
        tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        description = "Carol has a slow, measured, and intentional delivery, with a very close recording that almost has no background noise."
        global tts_input_ids
        tts_input_ids = tts_tokenizer(description, return_tensors="pt").input_ids.to(device)
        
    def convert_text_to_audio(self, text, output_path):
        import torch
        from multiprocessing import get_context, cpu_count
        chunks = self.split_text_into_chunks(text)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with get_context("spawn").Pool(
            cpu_count(),
            initializer=self.init_worker,
            initargs=(device,)
        ) as pool:
            audio_chunks = list(pool.imap(self._process_chunk, [(device, chunk, output_path, i, len(chunks)) for i, chunk in enumerate(chunks)]))
        self.combine_audio_chunks(audio_chunks, output_path)

    def _process_chunk(self, args):
        import torch
        import soundfile as sf
        device, chunk, output_path, i, total_chunks = args
        chunk_path = f"{output_path}_chunk_{i}.wav"
        with open(f"{output_path}_chunk_{i}.txt", 'w') as f:
            f.write(chunk)
        logging.info(f"Saving chunk {i+1}/{total_chunks} to {chunk_path}")
        global tts_model
        global tts_tokenizer
        global tts_input_ids
        if not tts_model:
            raise ValueError("Model not initialized")
        prompt_input_ids = tts_tokenizer(chunk, return_tensors="pt").input_ids.to(device)
        generation = tts_model.generate(input_ids=tts_input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(chunk_path, audio_arr, tts_model.config.sampling_rate)
        return chunk_path

class KokoroTTSStrategy(TTSStrategy):
    voices = [
        'af', # Default voice is a 50-50 mix of Bella & Sarah
        'af_bella', 'af_sarah', 'am_adam', 'am_michael',
        'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
        'af_nicole', 'af_sky',
    ]
    voice_name = 'af'

    def __init__(self):
        ensure_kokoro_installed()
        self.device = "cpu"
        self.voice_name = 'af'
        # Do NOT load model or voicepack here!

    @staticmethod
    def init_worker(device, voice_name):
        import sys
        sys.path.append('Kokoro-82M')
        from models import build_model
        from kokoro import generate
        import torch
        global tts_model
        tts_model = build_model("Kokoro-82M/kokoro-v0_19.pth", device)
        global tts_voicepack
        tts_voicepack = torch.load(f"Kokoro-82M/voices/{voice_name}.pt", weights_only=True).to(device)
        global kokoro_generate
        kokoro_generate = generate  # Make generate available as a global
        import logging
        logging.info(f"Loaded Kokoro voice: {voice_name}")

    def convert_text_to_audio(self, text, output_path):
        from multiprocessing import get_context, cpu_count
        chunks = self.split_text_into_chunks(text)
        batch_params = [(chunk, output_path, i, len(chunks)) for i, chunk in enumerate(chunks) if chunk]

        if batch_params:
            try:
                with get_context("spawn").Pool(
                    processes=cpu_count(), 
                    initializer=KokoroTTSStrategy.init_worker,
                    initargs=(self.device, self.voice_name)
                ) as pool:
                    audio_chunks = list(pool.imap(
                        KokoroTTSStrategy._process_chunk, 
                        batch_params
                    ))
                if not audio_chunks:
                    logging.error("No audio chunks were generated.")
                self.combine_audio_chunks(audio_chunks, output_path)
                for file in audio_chunks:
                    os.remove(file)
            except Exception as e:
                logging.error(f"Failed to convert text to audio: {e}")
        else:
            logging.info("No text to convert to audio.")

    @classmethod
    def _process_chunk(cls, params):
        import soundfile as sf
        chunk, output_path, i, total_chunks = params
        chunk_path = f"{output_path}_chunk_{i}.wav"
        with open(f"{output_path}_chunk_{i}.txt", 'w') as f:
            f.write(chunk)
        logging.info(f"Saving chunk {i+1}/{total_chunks} to {chunk_path}")

        audio = []
        global tts_model
        global tts_voicepack
        global kokoro_generate
        if not tts_model:
            raise ValueError("Model not initialized")
        snippet, _ = kokoro_generate(tts_model, chunk.strip(), tts_voicepack, lang=cls.voice_name[0])
        audio.extend(snippet)
        sf.write(chunk_path, audio, 24000)
        return chunk_path

from abc import ABC, abstractmethod
import nltk

class Repository(ABC):
    @abstractmethod
    def save_text(self, book_title: str, chapter_title: str, text: str):
        pass

    @abstractmethod
    def save_audio(self, book_title: str, chapter_title: str, audio_path: str):
        pass

class FileSystemRepo(Repository):
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def save_text(self, book_title: str, chapter_title: str, text: str):
        text_folder = os.path.join(self.root_folder, book_title, 'text')
        os.makedirs(text_folder, exist_ok=True)
        text_path = os.path.join(text_folder, f"{chapter_title}.txt")
        with open(text_path, 'w') as f:
            f.write(text)

    def save_audio(self, book_title: str, chapter_title: str, audio_path: str):
        audio_folder = os.path.join(self.root_folder, book_title, 'audio')
        os.makedirs(audio_folder, exist_ok=True)
        audio_dest_path = os.path.join(audio_folder, f"{chapter_title}.mp3")
        shutil.copyfile(audio_path, audio_dest_path)

class Chapter:
    def __init__(self, folder_path, ocr_strategy: OCRStrategy, tts_strategy: TTSStrategy):
        self.folder_path = folder_path
        self.title = os.path.basename(folder_path)
        self.pages = self.load_pages()
        self.ocr_strategy = ocr_strategy
        self.tts_strategy = tts_strategy

    def load_pages(self) -> Dict[str, None]:
        pages = {}
        for file_name in sorted(os.listdir(self.folder_path)):
            if (file_name.lower().endswith('.jpeg') and not file_name.lower().endswith('_processed.jpeg') and not file_name.lower().endswith('_enhanced.jpeg')) or \
                (file_name.lower().endswith('.jpg') and not file_name.lower().endswith('_processed.jpg') and not file_name.lower().endswith('_enhanced.jpg')):
                pages[os.path.join(self.folder_path, file_name)] = None
        return pages

    def extract_text_from_pages(self, overwrite=False):
        text_path = os.path.join(self.folder_path, f"{self.title}.txt")
        if not overwrite and os.path.exists(text_path):
            return
        with get_context("spawn").Pool(cpu_count()) as pool:
            results = pool.starmap(self._process_page, [(page, overwrite) for page in self.pages.keys()])
        for page, text in results:
            self.pages[page] = text

    def _process_page(self, page, overwrite):
        ext = os.path.splitext(page)[1]
        try:
            first_pass_file = page.replace(ext, '_ocr.txt')
            if not overwrite and os.path.exists(first_pass_file):
                with open(first_pass_file, 'r') as f:
                    text = f.read()
                    logging.debug(f"Skipping page {page} as text already exists.")
            else:
                image = self.ocr_strategy.preprocess_image(page)
                text = self.ocr_strategy.extract_text(page)
                with open(first_pass_file, 'w') as f:
                    f.write(text)
                    f.flush()
            os.remove(first_pass_file)
            return page, text
        except Exception as e:
            logging.error(f"Failed to process {page}: {e}")
            return page, None

    def post_process_from_pages(self, meta=None):
        chunked_pages = [list(self.pages.keys())[i:i+5] for i in range(0, len(self.pages), 5)]
        
        if _cancellation_flag.is_set():
            return
            
        try:
            with get_context("spawn").Pool(cpu_count()) as pool:
                for i, result in enumerate(pool.imap_unordered(self._post_process_chunk, [(pages, meta) for pages in chunked_pages])):
                    if _cancellation_flag.is_set():
                        pool.terminate()
                        pool.join()
                        return
        except Exception as e:
            if _cancellation_flag.is_set():
                return
            logging.error(f"Failed to post-process pages: {e}")

    def _post_process_chunk(self, params):
        # Check for cancellation at the start of each worker process
        if _cancellation_flag.is_set():
            return ""
            
        pages, meta = params
        pre_processed_text = '\n'.join(pages)
        return self.ocr_strategy.post_process_text(pre_processed_text, meta=meta)

    def post_process_from_file(self, meta=None):
        with open(os.path.join(self.folder_path, f"{self.title}.txt"), 'r') as f:
            lines = f.readlines()
        chunked_lines = [lines[i:i+50] for i in range(0, len(lines), 50)]
        with get_context("spawn").Pool(cpu_count()) as pool:
            processed_content = list(pool.imap(self._post_process_chunk, [(chunk, meta) for chunk in chunked_lines]))
        self.save_to_text_file('\n'.join(processed_content))

    @property
    def text(self):
        if any(self.pages.values()):
            return '\n'.join(self.pages.values())
        else:
            return ""
    
    @property
    def text_from_file(self):
        text_path = os.path.join(self.folder_path, f"{self.title}.txt")
        if os.path.exists(text_path):
            with open(text_path, 'r') as f:
                return f.read()
        else:
            return
    
    def save_to_text_file(self, content):
        text_path = os.path.join(self.folder_path, f"{self.title}.txt")
        with open(text_path, 'w') as f:
            f.write(content)
    
    def save_pages_to_chapter_text_file(self):
        if self.text:
            text_path = os.path.join(self.folder_path, f"{self.title}.txt")
            logging.info(f"Saving chapter {self.title} to {text_path}")
            with open(text_path, 'w') as f:
                f.write(self.text)
        else:
            logging.info(f"No text found for chapter {self.title}. Skipping save.")

    def convert_text_to_audio(self, overwrite=False):
        """
        Convert the chapter's text to audio using the selected TTS strategy.
        """
        audio_output_path = os.path.join(self.folder_path, f"{self.title}")
        if not overwrite and (os.path.exists(audio_output_path + '.wav') or os.path.exists(audio_output_path + '.mp3')):
            logging.info(f"Audio for chapter {self.title} already exists at {audio_output_path}.(wav/mp3). Skipping audio conversion.")
            return
        text_path = os.path.join(self.folder_path, f"{self.title}.txt")
        if not os.path.exists(text_path):
            logging.info(f"Text for chapter {self.title} does not exist at {text_path}. Skipping audio conversion.")
            return
        with open(text_path, 'r') as f:
            text = f.read()
        self.tts_strategy.convert_text_to_audio(text, audio_output_path)

    def export(self, repo: Repository, book_title=None):
        text_path = os.path.join(self.folder_path, f"{self.title}.txt")
        if os.path.exists(text_path):
            with open(text_path, 'r') as f:
                text = f.read()
            repo.save_text(book_title, self.title, text)

        audio_path = os.path.join(self.folder_path, f"{self.title}.mp3")
        if os.path.exists(audio_path):
            repo.save_audio(book_title, self.title, audio_path)

class Book:
    def __init__(self, root_folder, ocr_strategy: OCRStrategy, tts_strategy: TTSStrategy):
        self.root_folder = root_folder
        self.tts_strategy = tts_strategy
        self.chapters = self.load_chapters(ocr_strategy)
    
    @property
    def title(self):
        return os.path.basename(self.root_folder).replace('_', ' ')

    def load_chapters(self, ocr_strategy):
        chapters = {}
        for folder_name in sorted(os.listdir(self.root_folder)):
            folder_path = os.path.join(self.root_folder, folder_name)
            if os.path.isdir(folder_path):
                chapter =  Chapter(folder_path, ocr_strategy, self.tts_strategy)
                chapters[chapter.title] = chapter
        return chapters

    def extract_all_texts(self, overwrite=False) -> None:
        for chapter in self.chapters.values():
            logging.info(f"Extracting text from chapter: {chapter.title}")
            chapter.extract_text_from_pages(overwrite=overwrite)
            chapter.post_process_from_pages(meta={'book_title': self.title, 'chapter_title': chapter.title})
    
    def save_all_texts(self):
        for chapter in self.chapters.values():
            chapter.save_pages_to_chapter_text_file()

    def save_chapter_text(self, chapter_title):
        if chapter_title in self.chapters:
            self.chapters[chapter_title].save_pages_to_chapter_text_file()

    def convert_all_texts_to_audio(self, overwrite=False):
        for chapter in self.chapters.values():
            logging.info(f"Converting text to audio for chapter: {chapter.title}")
            chapter.convert_text_to_audio(overwrite=overwrite)

    def extract_text(self, chapter_title, overwrite=False):
        if chapter_title in self.chapters:
            chapter = self.chapters[chapter_title]
            logging.info(f"Extracting text from chapter: {chapter.title}")
            chapter.extract_text_from_pages(overwrite=overwrite)
            chapter.post_process_from_pages(meta={'book_title': self.title, 'chapter_title': chapter.title})

    def convert_text_to_audio(self, chapter_title, overwrite=False):
        if chapter_title in self.chapters:
            chapter = self.chapters[chapter_title]
            logging.info(f"Converting text to audio for chapter: {chapter.title}")
            chapter.convert_text_to_audio(overwrite=overwrite)
    
    def rerun_post_processing(self, chapter_title=None):
        for chapter in self.chapters.values():
            if not chapter_title or chapter_title == chapter.title:
                chapter.post_process_from_file(meta={'book_title': self.title, 'chapter_title': chapter.title})

    def export_all_chapters(self, repo: Repository):
        for chapter in self.chapters.values():
            logging.info(f"Exporting chapter: {chapter.title}")
            chapter.export(repo, book_title=self.title)

    def export_chapter(self, chapter_title, repo: Repository):
        if chapter_title in self.chapters:
            chapter = self.chapters[chapter_title]
            logging.info(f"Exporting chapter: {chapter.title}")
            chapter.export(repo, book_title=self.title)

if __name__ == '__main__':
    ROOT_FOLDER = os.path.join(os.path.dirname(__file__), 'book_pages', 'sandford__healing_for_a_womans_emotions')
    ocr_strategy = EasyOCROCR(post_process_strategy=XAITextPostProcessStrategy())  # Change to EasyOCROCR() to use easyocr
    tts_strategy = KokoroTTSStrategy()  # Change to KokoroTTSStrategy() to use Kokoro
    # tts_strategy = None
    book = Book(ROOT_FOLDER, ocr_strategy, tts_strategy)
    print("Chapters: ", book.chapters.keys())
    book.extract_all_texts(overwrite=False)
    book.save_all_texts()
    book.convert_all_texts_to_audio(overwrite=False)