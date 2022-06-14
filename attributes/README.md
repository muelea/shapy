# README

## Train AHW2S Model 
```
python main.py --exp-cfg configs/ahw2s.yaml --exp-opts output_dir=OUTPUT_FOLDER/caesar-female_smplx-neutral-10betas ds_gender=female model_gender=neutral num_shape_comps=10

python main.py --exp-cfg configs/ahw2s.yaml --exp-opts output_dir=OUTPUT_FOLDER/caesar-male_smplx-neutral-10betas ds_gender=male model_gender=neutral num_shape_comps=10
```


Bash script to run all A2S variations
``` 
#!/bin/bash

cd /is/cluster/lmueller2/projects/human_shape/attributes

OUTPUT_FOLDER=/ps/project/shapesemantics/trained_models_2/a2b_variations
CONFIGS=("01b_ah2s.yaml" "04a_hcwh2s.yaml" "02a_hw2s.yaml" "05b_ahwcwh2s.yaml" "03b_ac2s.yaml" "04b_ahcwh2s.yaml" "03a_c2s.yaml" "02b_ahw2s.yaml" "05a_hwcwh2s.yaml" "01a_h2s.yaml" "00_a2s.yaml")
DS_GENDERS=("female" "male")
MODEL_GENDERS=("female" "male")

#for DS_GENDER in ${DS_GENDERS[*]}; do
for index in ${!DS_GENDERS[*]}; do
    DS_GENDER=${DS_GENDERS[$index]};
    MODEL_GENDER=${MODEL_GENDERS[$index]};
    for CONFIG in ${CONFIGS[*]}; do
        mkdir -p $OUTPUT_FOLDER/caesar-$DS_GENDER\_smplx-$MODEL_GENDER-10betas/poynomial/$CONFIG;
        python fit_linear_regression.py --exp-cfg configs/a2s_variations_polynomial/$CONFIG --exp-opts output_dir=$OUTPUT_FOLDER/caesar-$DS_GENDER\_smplx-$MODEL_GENDER-10betas/poynomial/$CONFIG/ ds_gender=$DS_GENDER model_gender=$MODEL_GENDER num_shape_comps=10 train=False eval_test=True
    done
done
```