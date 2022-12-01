## Data

We use HBW and SSP-3D for evaluation. Model Agency Data, SPINX and Cureated Fits are used for training. Please navigate to the SHPAY (this repo) root folder and create a folder for datasets:
```
mkdir datasets
```

You can store the datasets in the dataset folder or somewhere else and create symlinks. We instruct the symlink option below. The folder structure should look like this:

shapy \
+-- datasets \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- HBW \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- SSP-3D \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- ModelAgencyData \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- SPINX \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- curated_fits \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- human36mx (if accessible) \
+-- attributes \
+-- data \
... 


## Training datasets

### Model Agency Data (body shape)
We use the model agency dataset to supervise body shape during training. Please download ModelAgencyData.zip from [here](https://shapy.is.tue.mpg.de), exctract it to $YOUR_MODEL_AGENCY_DATA_FOLDER, and follow the instructions to download the images. Afterwards, the folder should look like this:

$YOUR_MODEL_AGENCY_DATA_FOLDER \
--- cleaned_model_data.json \
--- README \
... \
+-- $model_agency_name \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- images \
|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; +-- $model_name 

```
ln -s $YOUR_MODEL_AGENCY_DATA_FOLDER datasets/ModelAgencyData
```

### SPINX and Curated fits (body pose)
We use training data from ExPose to supervise body pose during training. You can download the SMPL-X fits for SPINX and curated fits [here](https://expose.is.tue.mpg.de/download.php). Please follow the instructions on the ExPose website and download the images as well. Afterwards, the folders should look like this:

$YOUR_SPINX_FOLDER \
+-- images \
+-- data_npz \
|&nbsp; &nbsp; &nbsp; &nbsp; --- mpii.npz \
|&nbsp; &nbsp; &nbsp; &nbsp; --- lsp.npz \
|&nbsp; &nbsp; &nbsp; &nbsp; --- lspet.npz \
|&nbsp; &nbsp; &nbsp; &nbsp; --- coco.npz 

$YOUR_CURATED_FITS_FOLDER \
+-- images \
--- train.npz \
--- val.npz 

```
ln -s $YOUR_SPINX_FOLDER datasets/SPINX
ln -s $YOUR_CURATED_FITS_FOLDER datasets/curated_fits
```

### H3.6M (body pose)
We use SMPL-X fits to H3.6M to supervise body pose during training. Unfortunately, we can not publish these fits. If you do have access, the expected folder structure is:

$YOUR_H36M_FOLDER \
+-- npzs \
|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; --- human36m_smplx_train_rate_4.npz \
+-- train \
|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  +-- images \
|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  +-- keypoints 

```
ln -s $YOUR_H36M_FOLDER datasets/human36mx
```

## Evaluation datasets

### Human Bodies in The Wild (HBW)

HBW can be downloaded from [here](https://shapy.is.tue.mpg.de). For evaluation, use HBW.zip, extract it to $DATASETS/HBW and symlink:
```
ln -s $DATASETS/HBW datasets/HBW
```

To evaluate on HBW we provice a script that uses the validation dataset. The test set ground truth is not public. To evaluate you model on HBW Test, please follow the instructions on our [website](https://shapy.is.tue.mpg.de).

### SSP-3D
SSP-3D can be downloaded from [here](https://github.com/akashsengupta1997/SSP-3D):
```
wget https://github.com/akashsengupta1997/SSP-3D/raw/master/ssp_3d.zip 
unzip ssp_3d.zip -d datasets/
rm ssp_3d.zip
```
