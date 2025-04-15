# Single View Garment Reconstruction Using Diffusion Mapping Via Pattern Coordinates
<p align="center"><img src="figs/result.jpg"></p>

This is the repo for [**Single View Garment Reconstruction Using Diffusion Mapping Via Pattern Coordinates**](https://liren2515.github.io/page/DMap/DMap.html).

## Setup & Install
See [INSTALL.md](doc/INSTALL.md)

## Fitting
You can use the scripts under `./fit` to recover garment from the prepared images in `./fitting-data`:
```
cd fit
python fit_xxx.py # xxx is the type of the garment.
```
The output garment mesh and the body mesh are saved at `./fitting-data/XXX/result/XXX/mesh_verts_opt.obj` and `./fitting-data/XXX/result/XXX/body.obj`, respectively.

## Prepare your own data
If you want to prepare your own data for fitting, please check [DATA_PREPARE.md](doc/DATA_PREPARE.md)

## Citation
If you find our work useful, please cite it as:
```
@inproceedings{li2024garment,
  author = {Li, Ren and Cao, Cong and Dumery, Corentin You, Yingxuan and Li, Hao and Fua, Pascal},
  title = {{Single View Garment Reconstruction Using Diffusion Mapping Via Pattern Coordinates}},
  booktitle = {ACM SIGGRAPH 2025 Conference Papers},
  year = {2025}
}
```