# DistilHuBERT Fine-Tuning for Music Genre Classification

An audio deep learning project exploring music genre classification using a pretrained transformer fine-tuned on raw audio waveforms from the GTZAN dataset.

The notebook implements the complete pipeline:

* dataset loading and exploration,
* audio preprocessing,
* waveform normalization and resampling,
* preprocessing with Hugging Face feature extractors,
* DistilHuBERT fine-tuning,
* evaluation with accuracy curves and confusion matrices.

Everything is contained inside a single Jupyter notebook.

---

## Overview

The goal of this project is to classify music excerpts into 10 genres directly from raw audio signals using a pretrained transformer model.

Instead of manually engineering spectral features, the project relies on DistilHuBERT, a lightweight distilled version of HuBERT that directly processes normalized waveforms.

The notebook also includes:

* waveform visualization,
* class distribution analysis,
* time-frequency visualization,
* training and validation curves,
* confusion matrix analysis.

---

## Model

The model used is **DistilHuBERT**, a distilled version of HuBERT with around 23 million parameters.

The notebook fine-tunes the pretrained transformer on GTZAN by replacing the original classification head with a new one adapted to the 10 music genres.

Training uses:

* transfer learning,
* Hugging Face Trainer,
* gradient accumulation,
* weight decay,
* two-stage fine-tuning.

The final model reaches approximately:

```text
~87% test accuracy
```

---

## Dataset

The project uses the **GTZAN** dataset:

* 1000 music excerpts,
* 10 genres,
* 30 seconds per sample.

Genres include:

* blues
* classical
* country
* disco
* hiphop
* jazz
* metal
* pop
* reggae
* rock

Audio samples are resampled to 16 kHz and processed on the fly during training.

---

## Running the notebook

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Open the notebook

Launch Jupyter Notebook or Google Colab and open:

```text
DistilHuBERT_fine_tune.ipynb
```

### 3. Run the notebook cells

The notebook contains:

* dependency installation cells,
* dataset loading,
* preprocessing,
* model training,
* evaluation and visualization.

Simply execute the cells sequentially from top to bottom.

---

## Notes

The notebook is fully self-contained.
No additional Python scripts are required.

Some cells generate figures and audio samples used in the portfolio article.
