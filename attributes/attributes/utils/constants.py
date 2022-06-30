# ==== data directories ====
caesar_attributes_root = ''
models_attributes_root = ''
caesar_data_root = ''
bodytalk_data_dir=''

# Spencer et al. [2002] found that men and women overestimated height by 1.23 (2.57) cm 
# and 0.60 (2.68) cm, respectively (standard deviation in parentheses). 
# Men and women also underestimated their weight by 1.85 (2.92) kg and 1.40 (2.45) kg
SELF_REPORT_BIAS = {
    'female': {
        'weight': [1.40, 2.45],
        'height': [0.60, 2.68],
    },
    'male': {
        'weight':[1.85, 2.92],
        'height': [1.23, 2.57],
    }
}

MODEL_PATH = '../data/body_models/'

MODEL_EDGES = {
    'smplx': '../data/utility_files/smplx/smplx_edges.npy'
}

MEAS = {
    'smpl': '../data/utility_files/measurements/smpl_measurement_vertices.yaml',
    'smplx': '../data/utility_files/measurements/smplx_measurements.yaml',
    'definition': '../data/utility_files/measurements/measurement_defitions.yaml'
}

HD = {
    'smpl': '../data/utility_files/evaluation/eval_point_set/HD_SMPL_sparse.pkl',
    'smplx': '../data/utility_files/evaluation/eval_point_set/HD_SMPLX_from_SMPL.pkl',
}

ATTRIBUTE_NAMES = {
    'female': [
        'Big', 
        'Broad Shoulders', 
        'Feminine', 
        'Large Breasts', 
        'Long Legs', 
        'Long Neck', 
        'Long Torso', 
        'Muscular', 
        'Pear Shaped', 
        'Petite', 
        'Short', 
        'Short Arms', 
        'Skinny Legs', 
        'Slim Waist', 
        'Tall'
    ],
    'male': [
        'Average', 
        'Big', 
        'Broad Shoulders', 
        'Delicate Build', 
        'Long Legs', 
        'Long Neck', 
        'Long Torso', 
        'Masculine', 
        'Muscular', 
        'Rectangular', 
        'Short', 
        'Short Arms', 
        'Skinny Arms', 
        'Soft Body', 
        'Tall'
    ],
}

ATTRIBUTE_NAMES_SYNTHETIC_DATA = [
    'Attractive',
    'Average',
    'Big',
    'Broad Shoulders',
    'Built',
    'Curvy',
    'Feminine',
    'Fit',
    'Heavyset',
    'Hourglass',
    'Lean',
    'Long',
    'Long Legs',
    'Long Torso',
    'Masculine',
    'Muscular',
    'Pear Shaped',
    'Petite',
    'Proportioned',
    'Rectangular',
    'Round Apple',
    'Sexy',
    'Short',
    'Short Legs',
    'Short Torso',
    'Skinny',
    'Small',
    'Stocky',
    'Sturdy',
    'Tall'
]