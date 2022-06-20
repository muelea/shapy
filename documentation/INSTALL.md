## Installation

### Code

The code has been tested with Python 3.8, CUDA 10.2 and PyTorch 1.7.1 on Ubuntu 18.04.

- Clone this repo, create virtual environment & install requirements
    ```
    git clone git@github.com:muelea/shapy.git
    cd shapy
    export PYTHONPATH=$PYTHONPATH:$(pwd)/attributes/

    python3.8 -m venv .venv/shapy
    source .venv/shapy/bin/activate
    pip install -r requirements.txt

    cd attributes
    python setup.py install

    cd ../mesh-mesh-intersection
    export CUDA_SAMPLES_INC=$(pwd)/include
    pip install -r requirements.txt
    python setup.py install
    ```

### Body model and model data

#### Folder structure

In `shapy/data`, you will need subfolders for the neutral SMPL-X body model (body_models) and ExPose and SHAPY models and utilities (expose_release, trained_models, utility_files). The final data structure should look like this:
```bash
data
├── body_models
├── expose_release
├── trained_models
└── utility_files
```

#### SMPL Model

Download the neutral SMPL-X Model
[from the official website](https://smpl-x.is.tue.mpg.de/).
You can also optionally [download SMPL](https://smpl.is.tue.mpg.de/)
Your body model subfolder should have the following structure:

```bash
data
├── body_models
    └── smpl
        ├── SMPL_NEUTRAL.pkl
        ├── SMPL_FEMALE.pkl
        ├── SMPL_MALE.pkl
    └── smplx
        ├── SMPLX_NEUTRAL.npz
        ├── SMPLX_FEMALE.npz
        ├── SMPLX_MALE.npz
```

#### ExPose and SHAPY utilities

##### Option 1 (chose if you have not yet registered on the SHAPY website)
Download `shapy_data.zip` from our [website](https://shapy.is.tue.mpg.de) and extract it in the data folder:

```bash
cd data
unzip shapy_data.zip
```

##### Option 2 (chose if you have already registered on the SHAPY website)
Run `download_data.sh`. This will request your username and password for the SHAPY website and then download and extract the SHAPY model data.

```bash
cd data
bash download_data.sh
```

#### Complete folder structure

After that, you should have the following structure:

```bash
data
├── body_models
│   ├── smpl
│   └── smplx
├── expose_release
│   ├── data
│   │   ├── all_means.pkl
│   │   └── SMPLX_to_J14.pkl
│   └── utility_files
│       └── flame
│           └── SMPL-X__FLAME_vertex_ids.npy
├── trained_models
│   ├── a2b
│   │   ├── caesar-female_smplx-female-10betas
│   │   ├── caesar-female_smplx-neutral-10betas
│   │   ├── caesar-male_smplx-male-10betas
│   │   └── caesar-male_smplx-neutral-10betas
│   ├── b2a
│   │   └── polynomial
│   │       ├── caesar-female_smplx-female-10betas
│   │       ├── caesar-female_smplx-neutral-10betas
│   │       ├── caesar-male_smplx-male-10betas
│   │       └── caesar-male_smplx-neutral-10betas
│   └── shapy
│       └── SHAPY_A
└── utility_files
    ├── evaluation
    │   └── eval_point_set
    │       ├── HD_SMPL_sparse.pkl
    │       └── HD_SMPLX_from_SMPL.pkl
    ├── measurements
    │   ├── measurement_defitions.yaml
    │   ├── smpl_measurement_vertices.yaml
    │   └── smplx_measurements.yaml
    ├── shape_priors
    │   ├── female_normal.npz
    │   └── male_normal.npz
    └── smplx
        ├── smplx_correspondences.npz
        └── smplx_extra_joints.yaml
```
