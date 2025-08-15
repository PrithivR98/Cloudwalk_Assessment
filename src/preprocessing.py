import pandas as pd
import numpy as np
import librosa
import io
import os
from sklearn.preprocessing import StandardScaler
import joblib

class AudioPreprocessor:
    def __init__(self, sample_rate=8000, target_length=8000, n_mfcc=13, processed_dir="processed_data"):
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.n_mfcc = n_mfcc
        self.scaler = StandardScaler()
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def bytes_to_audio(self, audio_bytes):
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate)
        return audio, sr

    def pad_audio(self, audio):
        if len(audio) < self.target_length:
            pad_width = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        else:
            audio = audio[:self.target_length]
        return audio

    def add_noise(self, audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        audio_noisy = audio + noise_factor * noise
        return np.clip(audio_noisy, -1.0, 1.0)

    def extract_features(self, audio):
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)

        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        chroma_mean = np.mean(chroma, axis=1)

        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        return np.concatenate([mfccs_mean, chroma_mean, [centroid, bandwidth, rolloff, zcr]])

    def preprocess_df(self, df, augment_noise=False, save_name=None):
        features, labels = [], []
        for _, row in df.iterrows():
            audio_bytes = row['audio']['bytes']
            label = row['label']
            audio, _ = self.bytes_to_audio(audio_bytes)
            audio = self.pad_audio(audio)

            features.append(self.extract_features(audio))
            labels.append(label)

            if augment_noise:
                audio_noisy = self.add_noise(audio)
                features.append(self.extract_features(audio_noisy))
                labels.append(label)

        features = np.array(features)
        labels = np.array(labels)

        if save_name:
            np.savez_compressed(os.path.join(self.processed_dir, save_name), X=features, y=labels)
            print(f"Saved preprocessed data to {save_name}.npz")

        return features, labels

    def scale_features(self, X_train, X_test, save_scaler=True):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        if save_scaler:
            joblib.dump(self.scaler, os.path.join(self.processed_dir, "scaler.pkl"))
        return X_train_scaled, X_test_scaled

    def load_processed(self, file_name):
        data = np.load(os.path.join(self.processed_dir, file_name))
        return data['X'], data['y']

    def load_scaler(self):
        self.scaler = joblib.load(os.path.join(self.processed_dir, "scaler.pkl"))
