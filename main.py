from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from scipy.signal import butter, filtfilt
import requests
from typing import List, Dict, Optional
import io
from fastapi.middleware.cors import CORSMiddleware  

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="EEG Text Processor",
    description="API for processing EEG signals and converting them to text",
    version="1.0.0"
)

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response validation
class ProcessingConfig(BaseModel):
    fs: int = 256
    window_size: Optional[int] = None
    lowcut: float = 0.1
    highcut: float = 20
    sample_text: str = "The quick brown fox jumps over the lazy dog"

class TextConversionRequest(BaseModel):
    text: str

class ConversionResult(BaseModel):
    original_text: str
    decoded_text: str
    accuracy: float

class ProcessingResult(BaseModel):
    message: str
    data_shape: Dict[str, int]
    features_shape: Dict[str, int]
    vocabulary_size: int
    raw_data_sample: List[Dict]
    filtered_data_sample: List[Dict]
    segment_sample: List[List[float]]
    feature_sample: List[Dict]

# Global variables to store trained models and processors
eeg_processor = None
text_mapper = None
rf_model = None

class EEGProcessor:
    def __init__(self, fs=256):
        self.fs = fs
        self.scaler = RobustScaler()
        self.eeg_columns = None
        self.DEFAULT_DATA_URL = "https://drive.google.com/uc?id=1cPnqkPTiAtYvwo7Y90AitbxJy8NT_CN_"
        
    def load_data(self) -> pd.DataFrame:
        """Load default EEG dataset from Google Drive"""
        try:
            response = requests.get(self.DEFAULT_DATA_URL)
            response.raise_for_status()
            
            df = pd.read_csv(io.BytesIO(response.content))
            
            # Handle Missing Values
            initial_cols = len(df.columns)
            df.dropna(axis=1, how='all', inplace=True)
            df.fillna(df.mean(), inplace=True)
            
            # Select EEG Channels
            self.eeg_columns = [col for col in df.columns if col not in ["Unnamed: 32"]]
            df = df[self.eeg_columns]
            
            return df
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

    def apply_bandpass_filter(self, data, lowcut=0.1, highcut=20):
        """Apply bandpass filter to EEG data"""
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a
        
        b, a = butter_bandpass(lowcut, highcut, self.fs)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    
    def segment_data(self, df: pd.DataFrame, window_size: int):
        """Segment data into windows"""
        segments = []
        for i in range(0, len(df) - window_size, window_size):
            segment = df.iloc[i:i + window_size].values
            segments.append(segment)
        return np.array(segments)
    
    def extract_features(self, segment):
        """Extract statistical and frequency domain features"""
        feature_vector = {}
        
        for i, col in enumerate(self.eeg_columns):
            # Time domain features
            feature_vector[f"{col}_mean"] = float(np.mean(segment[:, i]))
            feature_vector[f"{col}_variance"] = float(np.var(segment[:, i]))
            feature_vector[f"{col}_kurtosis"] = float(kurtosis(segment[:, i]))
            feature_vector[f"{col}_skewness"] = float(skew(segment[:, i]))
            
            # Frequency domain features
            fft_values = np.abs(fft(segment[:, i]))
            feature_vector[f"{col}_fft_mean"] = float(np.mean(fft_values))
            feature_vector[f"{col}_fft_max"] = float(np.max(fft_values))
        
        return feature_vector

class TextEEGMapper:
    def __init__(self, feature_df, sample_text):
        self.feature_df = feature_df
        self.sample_text = sample_text.lower()
        self.char_to_features = {}
        self.label_encoder = LabelEncoder()
        
        # Create comprehensive vocabulary
        base_vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 ")
        base_vocab.extend(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{};':\",.<>/?\\|`~"))
        sample_chars = list(set(sample_text.lower()))
        
        self.vocabulary = list(set(base_vocab + sample_chars))
        self.vocabulary.sort()
        self.vocabulary.append("UNK")
        
        self.label_encoder.fit(self.vocabulary)
        self.create_character_mapping()
    
    def create_character_mapping(self):
        letters_per_segment = len(self.feature_df) // len(self.sample_text)
        
        for i, char in enumerate(self.sample_text):
            if char not in self.char_to_features:
                start_idx = i * letters_per_segment
                end_idx = (i + 1) * letters_per_segment
                if end_idx > len(self.feature_df):
                    end_idx = len(self.feature_df)
                char_features = self.feature_df.iloc[start_idx:end_idx].mean()
                self.char_to_features[char] = char_features.values
        
        self.char_to_features["UNK"] = np.mean(
            [features for features in self.char_to_features.values()], 
            axis=0
        )

    def encode_text_to_eeg(self, input_text):
        text_eeg_features = []
        
        for char in input_text.lower():
            if char in self.char_to_features:
                features = self.char_to_features[char]
            else:
                features = self.char_to_features["UNK"]
            text_eeg_features.append(features)
        
        return np.array(text_eeg_features)

    def decode_eeg_to_text(self, eeg_signals, model):
        predicted_labels = model.predict(eeg_signals)
        decoded_text = "".join(self.label_encoder.inverse_transform(predicted_labels))
        return decoded_text

@app.post("/initialize", response_model=Dict[str, str])
async def initialize_processor(config: ProcessingConfig):
    """Initialize the EEG processor with configuration settings"""
    global eeg_processor
    eeg_processor = EEGProcessor(fs=config.fs)
    return {"message": "EEG processor initialized successfully"}

@app.post("/upload-and-train", response_model=ProcessingResult)
async def upload_and_train(config: ProcessingConfig = ProcessingConfig()):
    """Train model using default dataset and return preprocessing results"""
    global eeg_processor, text_mapper, rf_model
    
    if not eeg_processor:
        raise HTTPException(status_code=400, detail="EEG processor not initialized")
    
    try:
        # Load and show raw data sample
        df = eeg_processor.load_data()
        raw_sample = df.head(3).to_dict(orient='records')
        
        # Signal Processing
        raw_values = df[eeg_processor.eeg_columns].values
        filtered_values = eeg_processor.apply_bandpass_filter(
            raw_values,
            lowcut=config.lowcut,
            highcut=config.highcut
        )
        df[eeg_processor.eeg_columns] = filtered_values
        filtered_sample = df.head(3).to_dict(orient='records')
        
        # Normalization
        df[eeg_processor.eeg_columns] = eeg_processor.scaler.fit_transform(df[eeg_processor.eeg_columns])
        
        # Segmentation and show sample
        window_size = config.window_size or int(0.5 * eeg_processor.fs)
        segments = eeg_processor.segment_data(df, window_size)
        segment_sample = segments[0][:5].tolist() if len(segments) > 0 else []
        
        # Feature Extraction
        feature_list = [eeg_processor.extract_features(segment) for segment in segments]
        feature_df = pd.DataFrame(feature_list)
        feature_sample = feature_df.head(3).to_dict(orient='records')
        
        # Initialize Text-EEG Mapper
        text_mapper = TextEEGMapper(feature_df, config.sample_text)
        
        # Prepare training data
        X = feature_df
        y = [config.sample_text[min(i // (len(feature_df) // len(config.sample_text)), len(config.sample_text) - 1)] 
             for i in range(len(feature_df))]
        
        y_encoded = text_mapper.label_encoder.transform(y)
        
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y_encoded)
        
        return {
            "message": "Model trained successfully",
            "data_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "features_shape": {"rows": feature_df.shape[0], "columns": feature_df.shape[1]},
            "vocabulary_size": len(text_mapper.vocabulary),
            "raw_data_sample": raw_sample,
            "filtered_data_sample": filtered_sample,
            "segment_sample": segment_sample,
            "feature_sample": feature_sample
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/convert-text", response_model=ConversionResult)
async def convert_text(request: TextConversionRequest):
    """Convert text to EEG signals and back"""
    if not all([eeg_processor, text_mapper, rf_model]):
        raise HTTPException(
            status_code=400,
            detail="System not initialized. Please initialize and train the model first."
        )
    
    try:
        eeg_features = text_mapper.encode_text_to_eeg(request.text)
        reconstructed_text = text_mapper.decode_eeg_to_text(eeg_features, rf_model)
        
        accuracy = sum(a == b for a, b in zip(request.text.lower(), reconstructed_text)) / len(request.text)
        
        return ConversionResult(
            original_text=request.text,
            decoded_text=reconstructed_text,
            accuracy=accuracy
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during conversion: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy"}
