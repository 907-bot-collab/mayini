
import gradio as gr
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json

from .preprocess import AutomatedPreprocessor


class PreprocessorWidget:
    """
    Interactive widget interface for the AutomatedPreprocessor
    Provides Gradio-based web UI with 4 modality tabs
    """
    
    def __init__(self):
        """Initialize preprocessor widget"""
        self.preprocessor = AutomatedPreprocessor()
        self.last_result = None
    
    # ========================================================================
    # TEXT PROCESSING
    # ========================================================================
    
    def process_text_data(self, text_input: str, operations: List[str],
                         remove_urls: bool = True, remove_emails: bool = True,
                         expand_contractions: bool = True,
                         remove_stopwords: bool = False,
                         tokenize_type: str = 'word',
                         stemmer_type: str = 'porter',
                         vectorize_type: str = 'tfidf',
                         max_features: int = 5000) -> Tuple[str, str, str]:
        """
        Process text data through selected pipeline
        
        Returns:
        --------
        tuple : (processed_text, summary, history)
        """
        try:
            if not text_input.strip():
                return "", "❌ No text provided", ""
            
            # Create fresh preprocessor
            self.preprocessor = AutomatedPreprocessor()
            self.preprocessor.data = text_input
            self.preprocessor.modality = 'text'
            
            # Build pipeline
            pipeline = []
            
            if 'Clean' in operations:
                pipeline.append({
                    'operation': 'clean',
                    'params': {
                        'remove_urls': remove_urls,
                        'remove_emails': remove_emails,
                        'expand_contractions': expand_contractions,
                        'remove_stopwords': remove_stopwords
                    }
                })
            
            if 'Normalize' in operations:
                pipeline.append({
                    'operation': 'normalize',
                    'params': {'lowercase': True}
                })
            
            if 'Tokenize' in operations:
                pipeline.append({
                    'operation': 'tokenize',
                    'params': {'type': tokenize_type}
                })
            
            if 'Stem' in operations:
                pipeline.append({
                    'operation': 'stem',
                    'params': {'stemmer': stemmer_type}
                })
            
            if 'Vectorize' in operations:
                pipeline.append({
                    'operation': 'vectorize',
                    'params': {
                        'type': vectorize_type,
                        'max_features': max_features
                    }
                })
            
            # Execute pipeline
            result = self.preprocessor.preprocess(pipeline)
            self.last_result = result
            
            # Format result for display
            if isinstance(result, list):
                result_str = ' '.join(str(r) for r in result[:100])
                if len(result) > 100:
                    result_str += f"\n... ({len(result)} total tokens)"
            else:
                result_str = str(result)[:1000]
                if len(str(result)) > 1000:
                    result_str += "\n... (truncated)"
            
            summary = self.preprocessor.summary()
            history = '\n'.join([f"• {h}" for h in self.preprocessor.get_history()])
            
            return result_str, summary, history
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return "", error_msg, ""
    
    # ========================================================================
    # IMAGE PROCESSING
    # ========================================================================
    
    def process_image_data(self, image_input, operations: List[str],
                          resize_enabled: bool = False,
                          resize_height: int = 224, resize_width: int = 224,
                          resize_method: str = 'bilinear',
                          rotate_angle: float = 0,
                          augment_noise: bool = False,
                          augment_brightness: bool = False,
                          brightness_factor: float = 1.2,
                          edge_method: str = 'sobel',
                          feature_type: str = 'hog') -> Tuple[Any, str, str]:
        """
        Process image data through selected pipeline
        
        Returns:
        --------
        tuple : (processed_image, summary, history)
        """
        try:
            if image_input is None:
                return None, "❌ No image provided", ""
            
            # Create fresh preprocessor
            self.preprocessor = AutomatedPreprocessor()
            self.preprocessor.data = np.array(image_input, dtype=np.float32)
            self.preprocessor.modality = 'image'
            
            # Build pipeline
            pipeline = []
            
            if 'Resize' in operations and resize_enabled:
                pipeline.append({
                    'operation': 'resize',
                    'params': {
                        'size': (resize_height, resize_width),
                        'method': resize_method
                    }
                })
            
            if 'Rotate' in operations and rotate_angle != 0:
                pipeline.append({
                    'operation': 'rotate',
                    'params': {'angle': rotate_angle}
                })
            
            if 'Augment' in operations:
                pipeline.append({
                    'operation': 'augment',
                    'params': {
                        'noise': augment_noise,
                        'brightness': augment_brightness,
                        'brightness_factor': brightness_factor
                    }
                })
            
            if 'Edge Detection' in operations:
                pipeline.append({
                    'operation': 'edge_detection',
                    'params': {'method': edge_method}
                })
            
            if 'Features' in operations:
                pipeline.append({
                    'operation': 'features',
                    'params': {'type': feature_type}
                })
            
            # Execute pipeline
            if not pipeline:
                result = self.preprocessor.data
            else:
                result = self.preprocessor.preprocess(pipeline)
            
            self.last_result = result
            
            # Convert to uint8 for display if needed
            if isinstance(result, np.ndarray):
                if result.dtype == np.float32 or result.dtype == np.float64:
                    result = np.clip(result * 255, 0, 255).astype(np.uint8)
                
                if len(result.shape) == 2:
                    # Grayscale - convert to RGB for display
                    result = np.stack([result] * 3, axis=2)
            
            summary = self.preprocessor.summary()
            history = '\n'.join([f"• {h}" for h in self.preprocessor.get_history()])
            
            return result, summary, history
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return None, error_msg, ""
    
    # ========================================================================
    # AUDIO PROCESSING
    # ========================================================================
    
    def process_audio_data(self, audio_input: Tuple, operations: List[str],
                          mfcc_n_coeff: int = 13,
                          mfcc_n_fft: int = 2048,
                          spec_n_fft: int = 2048,
                          spec_hop_length: int = 512,
                          pitch_semitones: float = 2,
                          effect_type: str = 'reverb') -> Tuple[str, str, str]:
        """
        Process audio data through selected pipeline
        
        Returns:
        --------
        tuple : (processing_info, summary, history)
        """
        try:
            if audio_input is None:
                return "", "❌ No audio file uploaded", ""
            
            # Extract audio data and sample rate
            sr, audio_data = audio_input
            audio_data = np.array(audio_data, dtype=np.float32)
            
            # Normalize
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
            
            # Create fresh preprocessor
            self.preprocessor = AutomatedPreprocessor()
            self.preprocessor.data = audio_data
            self.preprocessor.metadata['sample_rate'] = sr
            self.preprocessor.modality = 'audio'
            
            # Build pipeline
            pipeline = []
            
            if 'MFCC' in operations:
                pipeline.append({
                    'operation': 'mfcc',
                    'params': {'n_mfcc': mfcc_n_coeff}
                })
            
            if 'Spectrogram' in operations:
                pipeline.append({
                    'operation': 'spectrogram',
                    'params': {
                        'n_fft': spec_n_fft,
                        'hop_length': spec_hop_length
                    }
                })
            
            if 'Pitch Shift' in operations:
                pipeline.append({
                    'operation': 'pitch_shift',
                    'params': {'semitones': pitch_semitones}
                })
            
            if 'Effects' in operations:
                pipeline.append({
                    'operation': 'effects',
                    'params': {'effect': effect_type}
                })
            
            if 'Analysis' in operations:
                pipeline.append({
                    'operation': 'analysis',
                    'params': {'type': 'tempo'}
                })
            
            # Execute pipeline
            if not pipeline:
                result = self.preprocessor.data
                info_text = f"Audio loaded: {len(audio_data)} samples @ {sr}Hz"
            else:
                result = self.preprocessor.preprocess(pipeline)
                info_text = f"Processed audio: {len(audio_data)} samples @ {sr}Hz"
            
            self.last_result = result
            
            # Add result shape information
            if isinstance(result, np.ndarray):
                info_text += f"\nResult shape: {result.shape}"
            
            summary = self.preprocessor.summary()
            history = '\n'.join([f"• {h}" for h in self.preprocessor.get_history()])
            
            return info_text, summary, history
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return "", error_msg, ""
    
    # ========================================================================
    # VIDEO PROCESSING
    # ========================================================================
    
    def process_video_data(self, video_input, operations: List[str],
                          optical_flow_method: str = 'lucas_kanade',
                          scene_threshold: float = 0.5,
                          scene_method: str = 'histogram') -> Tuple[str, str, str]:
        """
        Process video data through selected pipeline
        
        Returns:
        --------
        tuple : (processing_info, summary, history)
        """
        try:
            if video_input is None:
                return "", "❌ No video file uploaded", ""
            
            # Note: Full video processing requires loading frames
            # For demo, show configuration
            
            config_text = f"""Video Processing Configuration:
─────────────────────────────────
Method: {optical_flow_method}
Threshold: {scene_threshold}
Scene Method: {scene_method}

Operations selected:
"""
            for op in operations:
                config_text += f"  • {op}\n"
            
            config_text += """
Note: Full video processing requires sufficient memory.
For large videos, process frames incrementally."""
            
            # Create fresh preprocessor for metadata
            self.preprocessor = AutomatedPreprocessor()
            self.preprocessor.modality = 'video'
            self.preprocessor.history.append("Video processing configured")
            
            summary = self.preprocessor.summary()
            history = '\n'.join([f"• {h}" for h in self.preprocessor.get_history()])
            
            return config_text, summary, history
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return "", error_msg, ""
    
    # ========================================================================
    # GRADIO INTERFACE CREATION
    # ========================================================================
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface with all tabs and controls
        
        Returns:
        --------
        gr.Blocks : Gradio interface
        """
        
        with gr.Blocks(
            title="Mayini Automated Preprocessor",
            theme=gr.themes.Soft(),
            css="""
            .header { text-align: center; }
            .tab-content { padding: 20px; }
            .operation-group { background: #f0f0f0; padding: 10px; border-radius: 5px; }
            """
        ) as demo:
            
            # Header
            gr.Markdown("# 🎯 Mayini Automated Preprocessor")
            gr.Markdown(
                "**Unified multimodal preprocessing** for text, image, audio, and video data\n\n"
                "Select a modality, upload your data, choose operations, and process instantly!"
            )
            
            with gr.Tabs():
                
                # ================================================================
                # TEXT TAB
                # ================================================================
                
                with gr.TabItem("📝 Text Processing", id="text_tab"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Input & Operations")
                            
                            text_input = gr.Textbox(
                                label="📄 Text Input",
                                placeholder="Paste your text here...",
                                lines=6,
                                max_lines=20
                            )
                            
                            text_ops = gr.CheckboxGroup(
                                choices=["Clean", "Normalize", "Tokenize", "Stem", "Vectorize"],
                                label="🔧 Operations",
                                value=["Clean", "Tokenize"]
                            )
                            
                            gr.Markdown("### Clean Options")
                            remove_urls = gr.Checkbox(label="Remove URLs", value=True)
                            remove_emails = gr.Checkbox(label="Remove Emails", value=True)
                            expand_contractions = gr.Checkbox(label="Expand Contractions", value=True)
                            remove_stopwords = gr.Checkbox(label="Remove Stopwords", value=False)
                            
                            gr.Markdown("### Processing Options")
                            tokenize_type = gr.Dropdown(
                                choices=["word", "char", "wordpiece"],
                                value="word",
                                label="Tokenize Type"
                            )
                            stemmer_type = gr.Dropdown(
                                choices=["porter", "simple", "lancaster"],
                                value="porter",
                                label="Stemmer Type"
                            )
                            vectorize_type = gr.Dropdown(
                                choices=["tfidf", "count", "binary"],
                                value="tfidf",
                                label="Vectorizer Type"
                            )
                            max_features = gr.Slider(100, 10000, 5000, step=100, label="Max Features")
                            
                            text_process_btn = gr.Button("🚀 Process Text", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Results")
                            text_output = gr.Textbox(label="📤 Processed Output", lines=8)
                            text_summary = gr.Textbox(label="📊 Summary", lines=6)
                            text_history = gr.Textbox(label="📋 History", lines=5)
                    
                    # Connect button
                    text_process_btn.click(
                        self.process_text_data,
                        inputs=[
                            text_input, text_ops, remove_urls, remove_emails,
                            expand_contractions, remove_stopwords,
                            tokenize_type, stemmer_type, vectorize_type, max_features
                        ],
                        outputs=[text_output, text_summary, text_history]
                    )
                
                # ================================================================
                # IMAGE TAB
                # ================================================================
                
                with gr.TabItem("🖼️ Image Processing", id="image_tab"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Input & Operations")
                            
                            image_input = gr.Image(label="🖼️ Image Input", type="numpy")
                            
                            image_ops = gr.CheckboxGroup(
                                choices=["Resize", "Rotate", "Augment", "Edge Detection", "Features"],
                                label="🔧 Operations",
                                value=["Resize"]
                            )
                            
                            gr.Markdown("### Resize Options")
                            resize_enabled = gr.Checkbox(label="Enable Resize", value=True)
                            resize_height = gr.Slider(32, 512, 224, step=32, label="Height")
                            resize_width = gr.Slider(32, 512, 224, step=32, label="Width")
                            resize_method = gr.Dropdown(
                                choices=["bilinear", "nearest"],
                                value="bilinear",
                                label="Method"
                            )
                            
                            gr.Markdown("### Rotation")
                            rotate_angle = gr.Slider(-180, 180, 0, step=15, label="Angle (degrees)")
                            
                            gr.Markdown("### Augmentation")
                            augment_noise = gr.Checkbox(label="Add Noise", value=False)
                            augment_brightness = gr.Checkbox(label="Adjust Brightness", value=False)
                            brightness_factor = gr.Slider(0.5, 2.0, 1.2, step=0.1, label="Brightness Factor")
                            
                            gr.Markdown("### Feature Extraction")
                            edge_method = gr.Dropdown(
                                choices=["sobel", "canny", "laplacian"],
                                value="sobel",
                                label="Edge Detection Method"
                            )
                            feature_type = gr.Dropdown(
                                choices=["hog", "histogram"],
                                value="hog",
                                label="Feature Type"
                            )
                            
                            image_process_btn = gr.Button("🚀 Process Image", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Results")
                            image_output = gr.Image(label="📤 Processed Image")
                            image_summary = gr.Textbox(label="📊 Summary", lines=6)
                            image_history = gr.Textbox(label="📋 History", lines=5)
                    
                    # Connect button
                    image_process_btn.click(
                        self.process_image_data,
                        inputs=[
                            image_input, image_ops, resize_enabled,
                            resize_height, resize_width, resize_method,
                            rotate_angle, augment_noise, augment_brightness,
                            brightness_factor, edge_method, feature_type
                        ],
                        outputs=[image_output, image_summary, image_history]
                    )
                
                # ================================================================
                # AUDIO TAB
                # ================================================================
                
                with gr.TabItem("🎵 Audio Processing", id="audio_tab"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Input & Operations")
                            
                            audio_input = gr.Audio(label="🎵 Audio Input", type="numpy")
                            
                            audio_ops = gr.CheckboxGroup(
                                choices=["MFCC", "Spectrogram", "Pitch Shift", "Effects", "Analysis"],
                                label="🔧 Operations",
                                value=["MFCC"]
                            )
                            
                            gr.Markdown("### MFCC Options")
                            mfcc_n_coeff = gr.Slider(10, 40, 13, step=1, label="Number of Coefficients")
                            mfcc_n_fft = gr.Slider(512, 4096, 2048, step=512, label="FFT Size")
                            
                            gr.Markdown("### Spectrogram Options")
                            spec_n_fft = gr.Slider(512, 4096, 2048, step=512, label="FFT Size")
                            spec_hop_length = gr.Slider(128, 1024, 512, step=128, label="Hop Length")
                            
                            gr.Markdown("### Effects")
                            pitch_semitones = gr.Slider(-12, 12, 2, step=1, label="Pitch Shift (semitones)")
                            effect_type = gr.Dropdown(
                                choices=["reverb", "echo", "chorus"],
                                value="reverb",
                                label="Effect Type"
                            )
                            
                            audio_process_btn = gr.Button("🚀 Process Audio", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Results")
                            audio_output = gr.Textbox(label="📤 Processing Info", lines=8)
                            audio_summary = gr.Textbox(label="📊 Summary", lines=6)
                            audio_history = gr.Textbox(label="📋 History", lines=5)
                    
                    # Connect button
                    audio_process_btn.click(
                        self.process_audio_data,
                        inputs=[
                            audio_input, audio_ops, mfcc_n_coeff,
                            mfcc_n_fft, spec_n_fft, spec_hop_length,
                            pitch_semitones, effect_type
                        ],
                        outputs=[audio_output, audio_summary, audio_history]
                    )
                
                # ================================================================
                # VIDEO TAB
                # ================================================================
                
                with gr.TabItem("🎬 Video Processing", id="video_tab"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Input & Operations")
                            
                            video_input = gr.Video(label="🎬 Video Input")
                            
                            video_ops = gr.CheckboxGroup(
                                choices=["Optical Flow", "Scene Detection", "Temporal Features"],
                                label="🔧 Operations",
                                value=["Scene Detection"]
                            )
                            
                            gr.Markdown("### Motion Detection")
                            optical_flow_method = gr.Dropdown(
                                choices=["lucas_kanade", "horn_schunck"],
                                value="lucas_kanade",
                                label="Optical Flow Method"
                            )
                            
                            gr.Markdown("### Scene Detection")
                            scene_threshold = gr.Slider(0.1, 1.0, 0.5, step=0.1, label="Threshold")
                            scene_method = gr.Dropdown(
                                choices=["histogram", "pixel", "ssim"],
                                value="histogram",
                                label="Detection Method"
                            )
                            
                            video_process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Results")
                            video_output = gr.Textbox(label="📤 Processing Info", lines=8)
                            video_summary = gr.Textbox(label="📊 Summary", lines=6)
                            video_history = gr.Textbox(label="📋 History", lines=5)
                    
                    # Connect button
                    video_process_btn.click(
                        self.process_video_data,
                        inputs=[
                            video_input, video_ops,
                            optical_flow_method, scene_threshold, scene_method
                        ],
                        outputs=[video_output, video_summary, video_history]
                    )
                
                # ================================================================
                # HELP TAB
                # ================================================================
                
                with gr.TabItem("❓ Help & Documentation", id="help_tab"):
                    gr.Markdown("""
                    # 📖 Mayini Automated Preprocessor Documentation
                    
                    ## 🎯 Overview
                    
                    The **Mayini Automated Preprocessor** provides a unified interface for preprocessing
                    multimodal data: text, images, audio, and video.
                    
                    ### ✨ Key Features
                    
                    - **Unified API** - Single interface for all data types
                    - **Auto-Detection** - Automatically detects input format
                    - **Pipeline Processing** - Chain multiple operations
                    - **Real-time Preview** - See results instantly
                    - **Metadata Tracking** - Automatic logging of all operations
                    
                    ---
                    
                    ## 📝 TEXT PROCESSING
                    
                    **Available Operations:**
                    - **Clean** - Remove URLs, emails, normalize text
                    - **Normalize** - Lowercase, remove accents
                    - **Tokenize** - Word, character, or subword tokenization
                    - **Stem** - Porter, Simple, or Lancaster stemming
                    - **Vectorize** - TF-IDF, Count, or Binary vectorization
                    
                    **Example Pipeline:**
                    1. Clean → Remove URLs and emails
                    2. Tokenize → Split into words
                    3. Stem → Apply Porter stemming
                    
                    ---
                    
                    ## 🖼️ IMAGE PROCESSING
                    
                    **Available Operations:**
                    - **Resize** - Change image dimensions
                    - **Rotate** - Rotate by any angle
                    - **Augment** - Add noise, adjust brightness
                    - **Edge Detection** - Sobel, Canny, Laplacian
                    - **Features** - Extract HOG or histogram features
                    
                    **Example Pipeline:**
                    1. Resize → 224x224
                    2. Normalize → Standardize values
                    3. Edge Detection → Sobel edges
                    
                    ---
                    
                    ## 🎵 AUDIO PROCESSING
                    
                    **Available Operations:**
                    - **MFCC** - Extract Mel-Frequency Cepstral Coefficients
                    - **Spectrogram** - Generate spectrogram
                    - **Pitch Shift** - Change pitch without changing speed
                    - **Effects** - Reverb, echo, chorus
                    - **Analysis** - Estimate tempo and pitch
                    
                    **Example Pipeline:**
                    1. MFCC → Extract 13 coefficients
                    2. Spectrogram → 2048 FFT size
                    3. Analysis → Estimate tempo
                    
                    ---
                    
                    ## 🎬 VIDEO PROCESSING
                    
                    **Available Operations:**
                    - **Optical Flow** - Lucas-Kanade or Horn-Schunck
                    - **Scene Detection** - Detect scene changes
                    - **Temporal Features** - Extract motion energy
                    - **Augment** - Temporal shift, crop
                    
                    **Example Pipeline:**
                    1. Optical Flow → Compute motion
                    2. Scene Detection → Find cuts
                    3. Temporal Features → Motion analysis
                    
                    ---
                    
                    ## 🚀 QUICK START
                    
                    1. **Select a Tab** - Choose your data modality
                    2. **Upload Data** - Upload your file or paste text
                    3. **Choose Operations** - Check the operations you want
                    4. **Adjust Parameters** - Fine-tune as needed
                    5. **Process** - Click the Process button
                    6. **View Results** - See processed data and metadata
                    
                    ---
                    
                    ## 💡 Tips
                    
                    - **Start Simple** - Begin with basic operations
                    - **Chain Operations** - Combine multiple for better results
                    - **Track History** - View all operations applied
                    - **Adjust Parameters** - Experiment with different settings
                    - **Export Results** - Use the processed data in your pipeline
                    
                    ---
                    
                    ## 📞 Support
                    
                    For issues or feature requests, visit: [Mayini GitHub](https://github.com/mayini-framework)
                    
                    **Version:** 1.0.0  
                    **Last Updated:** 2025-11-18
                    """)
        
        return demo


def launch_widget(share: bool = False, debug: bool = False, server_name: str = "0.0.0.0",
                 server_port: int = 7860, inbrowser: bool = True) -> None:
    """
    Launch the preprocessor widget in a web browser
    
    Parameters:
    -----------
    share : bool
        Create a public shareable link
    debug : bool
        Enable debug mode
    server_name : str
        Server address to bind to
    server_port : int
        Port number
    inbrowser : bool
        Automatically open in browser
    
    Example:
    --------
    from mayini.preprocessing import launch_widget
    
    launch_widget(share=False, debug=True)
    # Opens at http://localhost:7860
    """
    try:
        widget = PreprocessorWidget()
        interface = widget.create_interface()
        
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║        Mayini Automated Preprocessor Widget Launching         ║
╚═══════════════════════════════════════════════════════════════╝

🚀 Starting web interface...
📍 Local URL: http://{server_name}:{server_port}
""")
        
        if share:
            print("🌐 Public URL will be generated below...")
        
        print("📡 Waiting for connections...\n")
        
        interface.launch(
            share=share,
            debug=debug,
            server_name=server_name,
            server_port=server_port,
            inbrowser=inbrowser,
            show_error=True
        )
    
    except Exception as e:
        print(f"""
❌ Error launching widget: {str(e)}

Troubleshooting:
1. Ensure Gradio is installed: pip install gradio
2. Check port {server_port} is not in use
3. Verify all preprocessing modules are installed
4. Check internet connection for Gradio assets
        """)
        raise
