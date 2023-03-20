# HBW Challenge

Welcome to the Human Bodies in the Wild Body Shape Estimation Challenge. Here we provice instructions on how to evaluate your model on HBW. 


The submission format is a `.npz` file containing the image names (key image_name) and
the predicted meshes (key v_shaped). The predicted meshes must be in template pose with shape blend shaped (we call this v-shaped). You must provide a prediction for each image.

Here's some pseudo code for creating the npz file:
```
# create image name array
# np.array of shape (1631,) containing the relative HBW image paths
images_names = np.array([
    'test/001_noLab_22/Pictures_in_the_Wild/01329.png',
    ...,
    'test/035_84_20/Pictures_in_the_Wild/02570.png'
])

# create mesh array
# numpy array of shape (1631, 10475, 3) containing the SMPL-X predictions
v_shaped = []
for fn in images_names:
    mesh = my_results[fn]['v_shaped']
    v_shaped.append(mesh)
v_shaped = np.stack(v_shaped)
    
# save npz file
np.savez(
    'hbw_prediction', 
    image_name=images_names, 
    v_shaped=v_shaped
)
```

Please check you submission file formate before you submit it:
```
cd regressor/hbw_evaluation 
python test_submission_format.py --input-npz-file your_submission_file.npz
```

You can test the evaluation code on the validation set yourself. Note that 
`example_shapy_prediction.npz` wouldn't pass the submission format test, but 
it's still useful for demonstration purposes. 
```
cd regressor/hbw_evaluation
python evaluate_hbw.py --input-npz-file example_shapy_prediction.npz \
--model-type smplx --hbw-folder ../../datasets/HBW
```

If you are using other body models, i.e. not SMPL-X, please change the flags
accordingly. E.g. for SMPL predictions use:
```
cd regressor/hbw_evaluation
python evaluate_hbw.py --input-npz-file example_shapy_prediction_smpl.npz --model-type smpl --point-reg-fit ../../data/utility_files/evaluation/eval_point_set/HD_SMPL_sparse.pkl \
```

We are currently working on a submission website. If you urgently need to get your estimate
evaluated, please contact shapy@tue.mpg.de. We will evaluate your estimates for you. 