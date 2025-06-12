# Prepare your own data for fitting
Put your raw images under `./observations/XXX`, where `XXX` is the clothing type (i.e., Tshirt/Jacket/Trousers/Skirt).


## Step 1 - Segmentation
Although there are many off-the-shelf algorithms (e.g., [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)) that can be used, we recommend using [SAM](https://github.com/facebookresearch/segment-anything) and manual processing to obtain accurate garment segmentation masks. We assign the following values for the masks of different clothing types:
```
Tshirt - 60
Jacket - 120
Trousers - 180
Skirt - 240
```
After obtaining the segmentation masks, place them in  `./observations/mask-XXX`, and name the mask images using the same names as the raw images.


## Step 2 - Normal Estimation
Use [Sappiens](https://github.com/facebookresearch/sapiens) to get the normal estimation for the raw images, and save the results in `./observations/normal-XXX`. Name the normal images using the same names as the raw images.


## Step 3 - SMPL Body Parameters
Use [4DHumans](https://github.com/shubham-goel/4D-Humans) to estimate the SMPL parameters for the raw images. We have included a modified demo file `demo_modified.py` in this directory, which is based on the original `demo.py` from 4DHumans. This modified file saves the necessary estimations required by our method. Place it in the root directory of your 4DHumans installation and replace `4D-Humans/hmr2/utils/renderer.py` with `renderer.py` in this folder.

Execute `demo_modified.py` following the instructions provided in `demo.py` from 4DHumans. Make sure to include the arguments `--full_frame` and `--save_mesh` when running the script.

Put results in `./observations/smpl-XXX`. You should have some files named as `xxx_all.pt`.

## Step 3 - Alignment
To align the above estimations with the camera setting of our model, run
```
cd ./script
python ./step0_align_observation.py # change arguments when necessary
```