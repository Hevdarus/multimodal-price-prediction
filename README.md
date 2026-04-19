# Multimodal Price Prediction

Diploma thesis project.

## Goal

Predict product prices using:
- textual product data (title, description, quantity)
- product images

---

## Approach

The project is built in stages:

- **Text model (baseline)**  
  DistilBERT-based regression on product descriptions

- **Image model (planned)**  
  CNN-based price estimation from product images

- **Multimodal model (planned)**  
  Fusion of text and image representations

---

## Project Structure

```
src/
├── data/ # data loading & preprocessing
├── models/ # model definitions
├── training/ # training scripts
├── utils/ # helper functions
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## Main libraries:
- PyTorch
- Transformers
- pandas, numpy
- scikit-learn

---

## Experiments

Model configurations are stored in:

```Experiments.txt```


Each experiment defines:
- learning rate (`lr`)
- max sequence length (`max_length`)
- number of epochs (`epochs`)
- experiment name

Example:

```
experiment_name-et: "text_distilbert_lr1e5_len64_ep3"
lr: 1e-5
max_length: 64
epochs: 3
```

---

## Training

Run a specific experiment:

```bash
python -m src.training.train_text \
  --config_path Experiments.txt \
  --experiment_name text_distilbert_lr1e5_len64_ep3
```
Outputs are saved to:
```outputs/models/```

---

### Author
Zsolt Dede