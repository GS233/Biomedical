# Enhanced Biomedical Relation Extraction through Ensemble Learning and Attention Mechanism
This repository contains the resources and code for my paper titled "Enhanced Biomedical Relation Extraction through Ensemble Learning and Attention Mechanism".

## Abstract
Relation extraction (RE) plays a crucial role in biomedical research, as it is essential for uncovering complex semantic relationships between entities in textual data. Given the significance of RE in biomedical informatics and the increasing volume of literature, there is an urgent need for advanced computational models capable of accurately and efficiently extracting these relationships on a large scale. This paper proposes a novel approach combining ensemble learning Stacking and attention mechanisms to promote the effect of biomedical relation extraction. Through leveraging the advantages of multiple pre-trained models, our model demonstrates improved adaptability and robustness across diverse domains and application scenarios by ensemble learning Stacking. Additionally, introducing attention mechanisms enables the model to capture and utilize key information in the text more accurately. This study achieves state-of-the-art performance on benchmark datasets for three relation extraction tasks, providing new solutions for advancing biomedical informatics.

## File Structure
The repository contains the following key components:
```
├── data/                   # Folder for dataset storage (not included, see instructions)
├── model_path/                 # Folder containing pre-trained models and saved model checkpoints
├── models.py               # Code file containing only the model architecture
├── tools.py                # Scripts for data preprocessing and feature extraction
├── y_ddi.py                # Complete training script, including data preprocessing, feature extraction, and model training
├── original_file/          # Directory for original code versions
│   ├── y_aim.py            # Script for target dataset processing and feature extraction
│   ├── y_ChemProtMS.py     # Script for ChemProtMS dataset processing and feature extraction
│   └── y_ddi.py            # Original version of the drug-drug interaction (DDI) script
└── README.md               # Project documentation (usage instructions and introduction)

```
## How to Access
You can access the paper and related resources by following these steps:
1. clone code
```
git clone https://github.com/yourusername/biomedical-relation-extraction.git
```
2. download dataset:
We provide a compressed file, and you can use the following code to extract the .gz file:
```
# Example of code to extract the .gz file
tar -xvzf dataset_file.gz
```
3. download bert models:
If you are in a location where accessing the Hugging Face servers is difficult, please download the models in advance and place them in the model_path folder.
4. set your own parameters:
Before running the code, you can configure the model parameters in the script. Feel free to modify them based on your experimental setup.
5. Run the model:
After setting up, you can start training the model:
```
python y_ddi.py
```
6.Troubleshooting:
If you encounter any issues, please use the files in the original_file/ directory.

## Run the model:
After setting up, you can start training the model:
```
python y_ddi.py
```
## Explore the code and data:
After training, you can evaluate the model's performance using:
```
python eval.py
```

## How to Cite
If you find this work useful for your research, please cite it using the following format:
```
@article{XXXXXXX,
  title={XXXXXXX},
  author={XXXXXXX},
  journal={XXXXXXX},
  year={2024},
  volume={XX},
  pages={XX--XX},
  doi={XXXXXXX}
}
```
