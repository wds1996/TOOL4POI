# TOOL4POI
AAAI2026, TOOL4POI: A Tool-Augmented LLM Framework for Next POI Recommendation.

## 📂 Dataset

You can download the original dataset from the following Google Drive link:

[Download Dataset](https://drive.google.com/file/d/1SKKSwjdEapQh5WOEpv8XkLZTTkhlKDg6/view?usp=sharing)

After downloading, please place the dataset in the appropriate directory (e.g., `data/`).

## 🛠️ Data Preprocessing

To prepare the dataset for experiments, follow these steps:

```bash
# Step 1: Filter raw data
python data/data_filter.py

# Step 2: Process and format the dataset
# You can run this notebook to generate final data files
data/data.ipynb
```

After completing the above steps, the processed dataset will be ready for model training and evaluation.

## 🚀 Running Experiments

Before running experiments, please ensure that the SGLang backend is properly launched.

The startup command for SGLang is provided in the sglang.txt file

Once everything is ready, you can run the following scripts to obtain the results.

```bash
# Run the base model
python baseRec_multi.py

# Run the TOOL4POI framework
python toolRec_multi.py
```
