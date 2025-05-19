Use the scripts under `./scripts` to recover Skirt from the prepared images in `./data`. The codes use the data from `./data` as input, and save the results to `./fitting-results` by default.
```
cd scripts
```

Step 1: synthesize the normal for the invisible back.
```
python step1_back_normal.py # change the data path and save path with --data_root/--save_root if necessary.
```

Step 2: predict the UV coordinates and depth from the normal.
```
python step2_uv_mapping_FB.py # change the data path and save path with --data_root/--save_root if necessary.
```

Step 3: fit ISP to the incomplete mask to recover the complete mask/rest garment.
```
python step3_isp.py # change the save path with --save_root if necessary.
```

Step 4: fit the diffusion prior to the incomplete UV positional maps to recover the complete UV maps/deformed garment.
```
python step4_uv_inpainting.py # change the save path with --save_root if necessary.
```

Step 5: refine the recovered garment mesh using a set of observations and constraints (normal, mask, points, physical energy, etc.).
```
python step5_post.py # change the data path and save path with --data_root/--save_root if necessary.
```