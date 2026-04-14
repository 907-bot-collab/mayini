import sys
import os
import numpy as np
import traceback

# Ensure we are using the local source
sys.path.insert(0, os.path.abspath("src"))

from mayini.preprocessing.widget import PreprocessorWidget

def safe_print(msg):
    print(str(msg).encode('ascii', 'ignore').decode('ascii'))

def test_widget():
    widget = PreprocessorWidget()
    
    # 1. Text test
    safe_print("Testing Text Processing...")
    try:
        text, summary, hist = widget.process_text_data(
            text_input="Hello world! This is a test http://google.com email@test.com",
            operations=["Clean", "Normalize", "Tokenize", "Stem", "Vectorize"],
            remove_urls=True, remove_emails=True, expand_contractions=True,
            remove_stopwords=True, tokenize_type="word", stemmer_type="porter",
            vectorize_type="tfidf", max_features=5000
        )
        safe_print(f"Text output summary: {summary}")
        safe_print(f"Text output history: {hist}")
    except Exception as e:
        safe_print(f"Text Processing Failed with exception: {traceback.format_exc()}")

    # 2. Image test
    safe_print("\nTesting Image Processing...")
    try:
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img, summary, hist = widget.process_image_data(
            image_input=dummy_image,
            operations=["Resize", "Rotate", "Augment", "Edge Detection", "Features"],
            resize_enabled=True, resize_height=64, resize_width=64, resize_method="bilinear",
            rotate_angle=45, augment_noise=True, augment_brightness=True, brightness_factor=1.2,
            edge_method="sobel", feature_type="hog"
        )
        safe_print(f"Image output summary: {summary}")
    except Exception as e:
        safe_print(f"Image Processing Failed with exception: {traceback.format_exc()}")

    # 3. Audio test
    safe_print("\nTesting Audio Processing...")
    try:
        sr = 22050
        dummy_audio = np.random.randn(sr).astype(np.float32)
        info, summary, hist = widget.process_audio_data(
            audio_input=(sr, dummy_audio),
            operations=["MFCC", "Spectrogram", "Pitch Shift", "Effects", "Analysis"],
            mfcc_n_coeff=13, mfcc_n_fft=2048, spec_n_fft=2048, spec_hop_length=512,
            pitch_semitones=2, effect_type="reverb"
        )
        safe_print(f"Audio output info: {info}")
        safe_print(f"Audio output summary: {summary}")
    except Exception as e:
        safe_print(f"Audio Processing Failed with exception: {traceback.format_exc()}")

if __name__ == "__main__":
    test_widget()
