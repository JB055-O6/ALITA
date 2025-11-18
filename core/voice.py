import numpy as np
import threading
from queue import Queue
from pathlib import Path
from typing import Optional, Any, List
import logging
import time
import json
import io
import wave

# Optional heavy dependencies — import lazily and handle absence gracefully
try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional runtime dependency
    sd = None

try:
    import whisper
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig
except Exception:  # pragma: no cover
    whisper = None
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    BitsAndBytesConfig = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from openwakeword.model import Model as WakeWordModel
except Exception:
    WakeWordModel = None

try:
    import silero_vad
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
except Exception:
    silero_vad = None
    load_silero_vad = None
    read_audio = None
    get_speech_timestamps = None

try:
    from vosk import Model, KaldiRecognizer
except Exception:  # pragma: no cover
    Model = None
    KaldiRecognizer = None

try:
    import piper
    from piper.voice import PiperVoice
except Exception:
    piper = None
    PiperVoice = None

try:
    import speechbrain
    from speechbrain.pretrained import EncoderClassifier
except Exception:
    speechbrain = None
    EncoderClassifier = None

from ..config import VoiceConfig
from .safety import get_safety_manager


class VoiceInterface:
    """Voice interface using completely free and offline components with 4-bit quantization."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.command_queue = Queue()
        self.running = False
        self.is_listening = False
        self.audio_buffer = []
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 512
        
        # Wake word detection state
        self.wake_word_detected = False
        self.last_wake_word_time = 0
        
        # Safety system integration
        self.safety_manager = get_safety_manager()
        self.safety_manager.emergency_stop.register_callback(self._on_emergency_stop)
        
        # Initialize wake word detection (OpenWakeWord)
        self.wakeword_model = None
        self._init_wake_word()

        # Initialize VAD (Silero)
        self.vad_model = None
        self._init_vad()

        # Initialize main STT (Whisper with 4-bit quantization)
        self.whisper_model = None
        self.whisper_processor = None
        self._models_loaded = False

        # Initialize backup STT (Vosk) if available
        self.vosk_model = None
        self.vosk_recognizer = None
        self._init_vosk()

        # Initialize TTS (Piper)
        self.tts_engine = None
        self.tts_loaded = False
        
        # Initialize emotion analysis
        self.emotion_classifier = None
        self._init_emotion_analysis()
        
        logging.info("Voice Interface initialized")
    
    def _init_wake_word(self):
        """Initialize wake word detection with custom model."""
        if WakeWordModel is None:
            logging.warning("OpenWakeWord not available")
            return
        
        try:
            # Initialize with default models
            self.wakeword_model = WakeWordModel()
            
            # Try to load custom "Hey ALITA" model if available
            custom_model_path = Path("models/wake_word/hey_alita.tflite")
            if custom_model_path.exists():
                self.wakeword_model.load_model(str(custom_model_path))
                logging.info("Custom wake word model loaded")
            else:
                logging.info("Using default wake word models")
                
        except Exception as e:
            logging.error(f"Wake word initialization failed: {str(e)}")
            self.wakeword_model = None
    
    def _init_vad(self):
        """Initialize Silero VAD for voice activity detection."""
        if load_silero_vad is None:
            logging.warning("Silero VAD not available")
            return
        
        try:
            self.vad_model = load_silero_vad()
            logging.info("Silero VAD initialized")
        except Exception as e:
            logging.error(f"VAD initialization failed: {str(e)}")
            self.vad_model = None
    
    def _init_vosk(self):
        """Initialize Vosk as backup STT."""
        if Model is None or KaldiRecognizer is None:
            logging.warning("Vosk not available")
            return
        
        try:
            model_path = Path("models/vosk-model-small-en-us")
            if model_path.exists():
                self.vosk_model = Model(str(model_path))
                self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
                logging.info("Vosk backup STT initialized")
        except Exception as e:
            logging.error(f"Vosk initialization failed: {str(e)}")
    
    def _init_emotion_analysis(self):
        """Initialize emotion analysis using speechbrain."""
        if EncoderClassifier is None:
            logging.warning("Speechbrain not available for emotion analysis")
            return
        
        try:
            self.emotion_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="models/emotion_recognition"
            )
            logging.info("Emotion analysis initialized")
        except Exception as e:
            logging.error(f"Emotion analysis initialization failed: {str(e)}")

    def load_whisper_model(self):
        """Load Whisper model with 4-bit quantization for fast transcription."""
        if self._models_loaded:
            return
        
        if AutoModelForSpeechSeq2Seq is None or torch is None:
            logging.error("Transformers or torch not available for Whisper")
            return
        
        try:
            logging.info("Loading Whisper model with 4-bit quantization...")
            
            model_id = "openai/whisper-large-v3-turbo"
            
            if torch.cuda.is_available():
                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                
                # Load model with quantization
                self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            else:
                # CPU fallback - use smaller model
                model_id = "openai/whisper-base"
                self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
            
            # Load processor
            self.whisper_processor = AutoProcessor.from_pretrained(model_id)
            
            self._models_loaded = True
            logging.info("Whisper model loaded successfully")
            
        except Exception as e:
            logging.error(f"Whisper model loading failed: {str(e)}")
            # Fallback to basic whisper if available
            if whisper is not None:
                try:
                    self.whisper_model = whisper.load_model("base")
                    self._models_loaded = True
                    logging.info("Fallback to basic Whisper model")
                except:
                    pass
    
    def _setup_tts(self):
        """Setup offline TTS using Piper with multiple voice options."""
        if PiperVoice is None:
            logging.warning("Piper TTS not available")
            return None

        try:
            # Try to load configured voice
            voice_name = getattr(self.config, 'tts_voice', 'en_US-lessac-medium')
            voice_path = Path(f"models/voices/{voice_name}.onnx")
            
            if not voice_path.exists():
                # Try default voice
                voice_path = Path("models/voices/en_US-lessac-medium.onnx")
            
            if voice_path.exists():
                self.tts_engine = PiperVoice.load(str(voice_path))
                self.tts_loaded = True
                logging.info(f"Piper TTS loaded: {voice_path.name}")
                return self.tts_engine
            else:
                logging.warning(f"TTS voice file not found: {voice_path}")
                return None
                
        except Exception as e:
            logging.error(f"TTS setup failed: {str(e)}")
            return None
    
    def start_listening(self):
        """Start background listening for wake word and commands."""
        self.running = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.start()
    
    def stop(self):
        """Stop listening and cleanup."""
        self.running = False
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join()
    

    def _listen_loop(self):
        """Main listening loop using OpenWakeWord and Silero VAD."""
        if sd is None:
            logging.error("sounddevice not available")
            return
        
        logging.info("Starting voice listening loop...")
        
        # Audio buffer for wake word detection
        wake_word_buffer = np.zeros(self.sample_rate * 2, dtype=np.float32)  # 2 second buffer
        buffer_index = 0
        
        def audio_callback(indata, frames, time_info, status):
            nonlocal buffer_index, wake_word_buffer
            
            if status:
                logging.warning(f"Audio callback status: {status}")
            
            if not self.running:
                return
            
            # Convert to mono if needed
            audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
            audio_chunk = audio_chunk.flatten().astype(np.float32)
            
            # Update wake word buffer (rolling window)
            chunk_len = len(audio_chunk)
            if buffer_index + chunk_len > len(wake_word_buffer):
                # Roll buffer
                wake_word_buffer = np.roll(wake_word_buffer, -chunk_len)
                buffer_index = len(wake_word_buffer) - chunk_len
            
            wake_word_buffer[buffer_index:buffer_index + chunk_len] = audio_chunk
            buffer_index += chunk_len
            
            # Check for voice activity using VAD
            if self.vad_model is not None:
                try:
                    # Convert to tensor for VAD
                    audio_tensor = torch.from_numpy(audio_chunk)
                    speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
                    
                    # If speech detected, check for wake word
                    if speech_prob > 0.5:
                        if self._detect_wake_word(wake_word_buffer):
                            logging.info("Wake word detected!")
                            self.wake_word_detected = True
                            self.last_wake_word_time = time.time()
                            
                            # Capture command
                            threading.Thread(
                                target=self._capture_and_process_command,
                                daemon=True
                            ).start()
                            
                except Exception as e:
                    logging.error(f"VAD processing error: {str(e)}")
        
        try:
            # Start audio stream
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32
            ):
                logging.info("Audio stream started")
                while self.running:
                    sd.sleep(100)
                    
        except Exception as e:
            logging.error(f"Audio stream error: {str(e)}")
        finally:
            logging.info("Audio stream stopped")
    
    def _detect_wake_word(self, audio_buffer: np.ndarray) -> bool:
        """Detect wake word in audio buffer."""
        if self.wakeword_model is None:
            return False
        
        try:
            # Get predictions from wake word model
            predictions = self.wakeword_model.predict(audio_buffer)
            
            # Check if any wake word exceeds threshold
            for wake_word, score in predictions.items():
                if score > self.config.wake_word_threshold:
                    logging.info(f"Wake word '{wake_word}' detected with score {score:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Wake word detection error: {str(e)}")
            return False
    
    def _capture_and_process_command(self):
        """Capture audio command and process it."""
        try:
            # Record audio
            audio = self._record_audio(max_seconds=10)
            
            if audio is None or len(audio) == 0:
                logging.warning("No audio captured")
                return
            
            # Transcribe
            command = self._transcribe_audio(audio)
            
            if command:
                logging.info(f"Command transcribed: {command}")
                
                # Check for emergency stop command (Requirement 8.1: <200ms response)
                if self._is_emergency_stop_command(command):
                    self.safety_manager.emergency_stop.trigger_emergency_stop(
                        "Voice command: ALITA stop immediately"
                    )
                    self.speak("Emergency stop activated", blocking=False)
                    return
                
                # Analyze emotion if available
                emotion = self._analyze_emotion(audio)
                
                # Put command in queue with metadata
                self.command_queue.put({
                    "text": command,
                    "emotion": emotion,
                    "timestamp": time.time()
                })
            
        except Exception as e:
            logging.error(f"Command capture error: {str(e)}")
    
    def _is_emergency_stop_command(self, command: str) -> bool:
        """Check if command is an emergency stop command.
        
        Recognizes variations like:
        - "ALITA stop immediately"
        - "stop immediately"
        - "emergency stop"
        - "stop everything"
        """
        command_lower = command.lower().strip()
        
        emergency_phrases = [
            "alita stop immediately",
            "stop immediately",
            "emergency stop",
            "stop everything",
            "stop now",
            "halt",
            "abort"
        ]
        
        for phrase in emergency_phrases:
            if phrase in command_lower:
                return True
        
        return False
    
    def _on_emergency_stop(self):
        """Handle emergency stop event."""
        logging.critical("🚨 Voice Interface: Emergency stop - halting all operations")
        
        # Stop listening
        self.running = False
        
        # Clear command queue
        self.clear_queue()
        
        # Stop any ongoing speech
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
    
    def _record_audio(self, max_seconds: int = 10) -> Optional[np.ndarray]:
        """Record audio for command capture with VAD-based silence detection."""
        if sd is None:
            logging.error("sounddevice not available")
            return None
        
        logging.info(f"Recording audio for up to {max_seconds} seconds...")
        
        recorded_audio = []
        silence_threshold = 0.3  # VAD probability threshold
        silence_duration = 0
        max_silence = 2.0  # Stop after 2 seconds of silence
        
        def callback(indata, frames, time_info, status):
            nonlocal silence_duration
            
            if status:
                logging.warning(f"Recording status: {status}")
            
            # Convert to mono
            audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
            audio_chunk = audio_chunk.flatten().astype(np.float32)
            recorded_audio.append(audio_chunk.copy())
            
            # Check for silence using VAD
            if self.vad_model is not None:
                try:
                    audio_tensor = torch.from_numpy(audio_chunk)
                    speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
                    
                    if speech_prob < silence_threshold:
                        silence_duration += len(audio_chunk) / self.sample_rate
                    else:
                        silence_duration = 0
                        
                except Exception:
                    pass
        
        try:
            # Record with callback
            with sd.InputStream(
                callback=callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.float32
            ):
                start_time = time.time()
                while time.time() - start_time < max_seconds:
                    sd.sleep(100)
                    
                    # Stop if silence detected
                    if silence_duration >= max_silence and len(recorded_audio) > 10:
                        logging.info("Silence detected, stopping recording")
                        break
            
            # Concatenate all chunks
            if recorded_audio:
                audio_array = np.concatenate(recorded_audio)
                logging.info(f"Recorded {len(audio_array) / self.sample_rate:.2f} seconds of audio")
                return audio_array
            else:
                return None
                
        except Exception as e:
            logging.error(f"Audio recording error: {str(e)}")
            return None
    
    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio using Whisper with fallback to Vosk."""
        # Ensure Whisper is loaded
        if not self._models_loaded:
            self.load_whisper_model()
        
        # Try Whisper first
        if self.whisper_model is not None:
            try:
                logging.info("Transcribing with Whisper...")
                
                if self.whisper_processor is not None:
                    # Using transformers Whisper
                    inputs = self.whisper_processor(
                        audio,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    )
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = self.whisper_model.generate(**inputs)
                    
                    transcription = self.whisper_processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]
                else:
                    # Using basic whisper
                    result = self.whisper_model.transcribe(audio)
                    transcription = result["text"]
                
                return transcription.strip()
                
            except Exception as e:
                logging.error(f"Whisper transcription failed: {str(e)}")
        
        # Fallback to Vosk
        if self.vosk_recognizer is not None:
            try:
                logging.info("Falling back to Vosk...")
                
                # Convert to 16-bit PCM
                audio_int16 = (audio * 32767).astype(np.int16)
                
                # Process audio
                self.vosk_recognizer.AcceptWaveform(audio_int16.tobytes())
                result = json.loads(self.vosk_recognizer.Result())
                
                return result.get("text", "").strip()
                
            except Exception as e:
                logging.error(f"Vosk transcription failed: {str(e)}")
        
        return ""
    
    def _analyze_emotion(self, audio: np.ndarray) -> Optional[str]:
        """Analyze emotion from audio using speechbrain."""
        if self.emotion_classifier is None:
            return None
        
        try:
            # Save audio to temporary file (speechbrain requires file input)
            temp_path = Path("temp_audio.wav")
            
            # Write WAV file
            with wave.open(str(temp_path), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                audio_int16 = (audio * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            # Classify emotion
            out_prob, score, index, text_lab = self.emotion_classifier.classify_file(str(temp_path))
            
            # Clean up
            temp_path.unlink()
            
            return text_lab[0] if text_lab else None
            
        except Exception as e:
            logging.error(f"Emotion analysis failed: {str(e)}")
            return None
    
    def speak(self, text: str, blocking: bool = True):
        """Convert text to speech using Piper TTS.
        
        Args:
            text: Text to synthesize
            blocking: If True, wait for speech to complete
        """
        if sd is None:
            logging.error("sounddevice not available for TTS")
            return
        
        # Ensure TTS is loaded
        if not self.tts_loaded:
            self._setup_tts()
        
        if self.tts_engine is None:
            logging.warning("TTS engine not available")
            return
        
        try:
            logging.info(f"Speaking: {text[:50]}...")
            
            # Synthesize audio
            audio_stream = self.tts_engine.synthesize_stream_raw(text)
            
            # Collect audio chunks
            audio_chunks = []
            for audio_bytes in audio_stream:
                audio_chunks.append(np.frombuffer(audio_bytes, dtype=np.int16))
            
            if audio_chunks:
                audio_array = np.concatenate(audio_chunks)
                
                # Convert to float32 for playback
                audio_float = audio_array.astype(np.float32) / 32767.0
                
                # Play audio
                sd.play(audio_float, self.tts_engine.config.sample_rate)
                
                if blocking:
                    sd.wait()
                    
        except Exception as e:
            logging.error(f"TTS failed: {str(e)}")
    
    def get_command(self) -> Optional[dict]:
        """Get next command from queue.
        
        Returns:
            Dictionary with command text, emotion, and timestamp, or None
        """
        try:
            return self.command_queue.get_nowait()
        except:
            return None
    
    def clear_queue(self):
        """Clear all pending commands from queue."""
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                break