import kagglehub
import os

path = kagglehub.dataset_download(
    "inancigdem/eeg-data-for-mental-attention-state-detection"
)

print("Dataset path:", path)

for root, dirs, files in os.walk(path):
    level = root.replace(path, "").count(os.sep)
    indent = "    " * level
    print(f"{indent}{os.path.basename(root)}/")

    for file in files:
        print(f"{indent}    {file}")