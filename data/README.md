## Dataset

The dataset is publicly available on Kaggle:

**EEG Data for Mental Attention State Detection**  
(https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection)

### Download instructions

The dataset can be downloaded automatically using **KaggleHub**.

Install KaggleHub:

```bash
pip install kagglehub
```

Then run:

```python
import kagglehub

path = kagglehub.dataset_download(
    "inancigdem/eeg-data-for-mental-attention-state-detection"
)

print("Dataset path:", path)
```

After downloading, place or copy the dataset into:

```text
data/raw/
```

The raw dataset is intentionally excluded from the Git repository using `.gitignore` because of its size. Only the project code and documentation are tracked in Git.