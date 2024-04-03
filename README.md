#### Liver Deliver
---
This repository contains my project for BWKI 2022. I concerned myself with liver and liver tumor segmentation using different U-Nets.

---
##### Technical Sources:

- Data-Liver (original):
https://competitions.codalab.org/competitions/17094

- Data-Liver (preprocessed):
https://www.kaggle.com/datasets/andrewmvd/lits-png 

- Data-Gl-Tract (original):
https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation

- Data-Gl-Tract (preprocessed)
https://www.kaggle.com/datasets/purplejester/uwm-images-and-masks

- U-Net (paper):
https://arxiv.org/abs/1505.04597

- Residual U-Net (paper):
https://arxiv.org/abs/1711.10684

- Attention U-Net (paper):
https://arxiv.org/abs/1804.03999

- Residual Attention U-Net (paper):
https://arxiv.org/abs/1909.10360

---
#### Requirements:
Create the conda environment **environment.yml**.

1. Create Environment:
```
conda env create -f environment.yml
```
2. Activate Environment:

```
conda activate BWKI_1
```

3. Check Environment:
```
conda env list
```

##### Requirements.txt:

Install using the requirements.txt file.

```
pip install -r requirements.txt
```
---
#### Usage:

You need a base-directory in which your augmented data (train_images_augmented..., val_images_augmented, test_masks, test_images) is. Otherwise your base-directory needs to contain the folders (train_images..., val_images). Once your data is ready, you can load it in via changing the hyperparamters using argparse. You can find many models in the resunet.py or models.py file, AttResUnet(small) is the most efficient one. 

