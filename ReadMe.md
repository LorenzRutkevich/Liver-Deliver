## Liver Deliver
---
###### Das Ziel dieses Projekts ist es, ein breites Verständnis für die Leber- und Lebertumor-Segmentation zu schaffen. Es beinhaltet Modelle basierend auf U-Net, die verbessert wurden. Herausgearbeitet wurden Arten des Residual Attention Unets, was sowohl wenige Parameter, aber auch eine gute Effizienz aufweist.
---
#### Technische Quellen:

- Daten (original):
https://competitions.codalab.org/competitions/17094

- Daten (verarbeitet):
https://www.kaggle.com/datasets/andrewmvd/lits-png 

- U-Net (paper):
https://arxiv.org/abs/1505.04597

- Residual U-Net (paper):
https://arxiv.org/abs/1711.10684

- Attention U-Net (paper):
https://arxiv.org/abs/1804.03999

- Residual Attention U-Net (paper):
https://arxiv.org/abs/1909.10360

- Metrics:
https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

- Streamlit
https://streamlit.io/

----
#### Installierung:
##### Conda:
Dem Projekt ist ein Conda-Environment anhängend. Dieses kann wie folgt installiert und ausgeführt werden:

1. Create Environment:

			conda env create -f environment.yml

2. Activate Environment:

			conda activate BWKI_1

3. Check Environment:

			conda env list

##### Requirements.txt:
Mit dem angehörigen requirements.txt können die Pakete mittels pip installiert werden.

			pip install -r requirements.txt

##### Dependencies:
Wen alles from source installiert werden soll, sehen Sie die benötigten Pakete hier aufgelistet.

			keras==2.9.0
			matplotlib==3.5.2
			numpy==1.23.1
			Pillow==9.2.0
			tensorflow==2.9.1
			tqdm==4.64.0
			opencv-python
----
#### Usage:

Der Hauptbestandteil ist Liver.ipynb/py. Hier können sie mittels argparse und den Argumenten die Hyperparameter o.ä. verändern. 

###### Daten/Augmentation:

**Sie brauchen ein base-dir,** in dem sich die **nicht augmentierten** Daten oder **augmentierten Daten** befinden. Insofern Sie die vorgefertigten Parameter nutzen und Daten augmentieren wollen, so **müssen** Ordner mit den Namen **"val_images, val_masks, training_images und training_masks" vorhanden sein.** Falls Sie Daten ohne Augmentation reinladen wollen, so **müssen** die Ordner **"val_images_augmented, val_masks_augmented, training_images_augmented, training_masks_augmented, test_masks und test_images"  vorhanden sein.** Test Images/Masks werden aus den Validierungs- und Trainings-Bildern/Masken während der Augmentation erstellt.** 


###### Training/Validierung/Evaluierung:

**Normal werden alle** Trainings- und Validierungs-Masken/Bilder in np.arrays gespeichert und dem Model zum Training zugeführt. Es gibt jedoch die Parameter 

			--skip_val;
			--skip_train;
			--skip_test;

mit denen Sie vor dem Training die Anzahl der zu überspringenden Daten spezifizieren können e.g --skip_val 5000.

Die einstelltbaren Parameter sind:

	args.add_argument('--train_img', type=str, default='None) # path to train images
	args.add_argument('--train_mask', type=str, default=None) # path to train masks
	args.add_argument('--val_img', type=str, default=None) # path to validation images
	args.add_argument('--val_mask', type=str, default=None) # path to the validation masks
	args.add_argument('--epochs', type=int, default=15) # number of epochs
	args.add_argument('--batch_size', type=int, default=32) # batch size
	args.add_argument('--predict', type=bool, default=False) # Change to True for prediction
	args.add_argument('--augment', type=bool, default=False) # Change to True to augment the data
	args.add_argument('--test_img', type=str, default='/home/lorenz/U-Net/archive/test_images/') # path to the image to predict
	args.add_argument('--test_mask', type=str, default='/home/lorenz/U-Net/archive/test_masks/') # path to the mask to predict
	args.add_argument('--img_width', type=int, default=128) # width of the image
	args.add_argument('--img_height', type=int, default=128) # height of the image
	args.add_argument('--img_channels', type=int, default=1) # channels of the image 
	args.add_argument('--save_augmentations', type=bool, default=True) # save the augmented images
	args.add_argument('--base_dir', type=str, default='/home/lorenz/U-Net/archive') # path to the parent directory of the training data
	args.add_argument('--model', type=str, default='NestedRes') # model to use
	args.add_argument('--show_summary', type=bool, default=True) # show the summary of the model
	args.add_argument('--measure', type=bool, default=True) # measure the predicted tumors
	args.add_argument('--skip_train', type=int, default=0) # skip the training for the specified number of images
	args.add_argument('--skip_val', type=int, default=0) # amount of images to skip in the validation set
	args.add_argument('--skip_test', type=int, default=0) # amount of test images to skip
	args.add_argument('--skip_paths', type=bool, default=False) # for direct loading of the paths
	args.add_argument('--save_predictions', type=bool, default=True) # save predictions


Um die **vollen Möglichkeiten des Programmes zu haben, sind 3 files von nöten**,  2  files mit Modellen **(models.py, reunet.py)** und **liver.py/ipynb**. Zum Projekt gehören des weiteren noch zusätlich die website **streamlit.py** und das Video **manim.py**. *Alle weiteren Daten und Folder werden während des Ausführens kreiert.*
Für die beste Erfahrung ist ein Linux-basiertes Betriebssystem vonnöten.

---
### Modelle:
#### U-Net:
![[Unet.png]]

#### Attention Residual Unet1:
![[att_res_unet.png]]
#### Attention Residual Unet2:
![[AttResUnet(small).png]]




