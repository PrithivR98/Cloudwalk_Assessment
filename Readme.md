---

## Development Recording Notes

During development, I recorded up to 30 minutes of the coding process to capture how I worked with an LLM. Key activities include:

- **Prompting for code structure**: Asked the LLM how to modularize preprocessing, modeling, and evaluation steps.
- **Feature extraction guidance**: Used LLM to recall `librosa` functions for MFCCs, chroma, spectral features, and zero crossing rate.
- **Debugging and testing**: LLM suggested fixes for audio padding, noise augmentation, and LightGBM training errors.
- **Iterative improvement**: Explored alternative model choices and cross-validation strategies with LLM assistance.
- **Real-time experimentation**: Tested small snippets for audio augmentation and spectrogram visualization before integrating into the pipeline.

- **Models and Results**: I used LightGBM model and I got an accuracy of 92%. I experimented with different models such as SVM, Logistic Regression and got the best output with LightGBM I sued Stratified K-validation. 
Cross-Validation Scores: {'LightGBM': 0.912962962962963}

This recording provides a **raw glimpse of the development workflow** and demonstrates how LLMs were used as active coding partners.


My other work : https://github.com/PrithivR98/Music-Classification-and-Recommendation/tree/main

Video Link : https://drive.google.com/file/d/1psYx3dIML20debSDdW_TN4Qsb91kIE4V/view?usp=sharing
