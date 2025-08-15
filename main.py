import pandas as pd
from src import AudioPreprocessor, ModelTrainer, Evaluator

# Initialize classes
preprocessor = AudioPreprocessor()
trainer = ModelTrainer(n_splits=5)
evaluator = Evaluator()

# Load datasets
print("Loading Datasets")
train_df = pd.read_parquet("data/train-00000-of-00001.parquet")
test_df = pd.read_parquet("data/test-00000-of-00001.parquet")

# Preprocess data
print("Preprocessing Datasets")

X_train, y_train = preprocessor.preprocess_df(train_df, augment_noise=True)
X_test, y_test = preprocessor.preprocess_df(test_df, augment_noise=False)
X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

# Train & get CV scores
print("Training and CV Scores")
scores = trainer.train_with_stratified_kfold(X_train_scaled, y_train)
print("Cross-Validation Scores:", scores)

# Plot example spectrograms
for i in range(3):
    audio_bytes = train_df['audio'][i]['bytes']
    label = train_df['label'][i]
    audio, _ = preprocessor.bytes_to_audio(audio_bytes)
    audio = preprocessor.pad_audio(audio)
    evaluator.plot_log_mel(audio, title=f"Sample {i} - Label {label}")
