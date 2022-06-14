## Data

Download and process our datasets for evaluation. Run all commands from the shapy root. Folder tree should be: 

shapy \
+-- datasets \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- HBW \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- SSP-3D \
|&nbsp; &nbsp; &nbsp; &nbsp; +-- ModelAgencyData \
+-- attributes \
+-- data \
... 


### Human Bodies in The Wild (HBW)

HBW can be downloaded from [here](https://shapy.is.tue.mpg.de). For evaluation, use HBW.zip, extract it to $DATASETS/HBW and symlink:
```
mkdir datasets
ln -s $DATASETS/HBW datasets/HBW
```

To evaluate on HBW we provice a script that uses the validation dataset. The test set ground truth is not public. To evaluate you model on HBW Test, please follow the instructions on our [website](https://shapy.is.tue.mpg.de).

### Model Agencies
For the Model agency data, please download the ModelAgencyData.zip from [here](https://shapy.is.tue.mpg.de) and follow the processing script.

### SSP-3D
SSP-3D can be downloaded from [here](https://github.com/akashsengupta1997/SSP-3D):
```
wget https://github.com/akashsengupta1997/SSP-3D/raw/master/ssp_3d.zip 
unzip ssp_3d.zip -d datasets/
rm ssp_3d.zip
```
