# Interpretation2Adversary

This project provides an example of framework implementation for the paper: <br><br>

**Adversarial Detection with Model Interpretation**<br>
Ninghao Liu, Hongxia Yang, Xia Hu<br>
Proceedings of KDD'18, London, UK <br><br>

using LASSO as the local interpretation model, under the L2 norm constraint.

**Please cite our paper if you use the codes. Thanks!**


## Example to run the codes:
```
python __main__.py --input example_data/data_small.npy --seed-ratio 0.05 --dist-ratio-min 0.1 --dist-ratio-max 0.5 --dist-ratio-num 5
```
Some parameters are introduced as below:
- input: The path to the data file.
- seed-ratio: Ratio of evasion prone instances as seeds to the overall data.
- dist-ratio-min, dist-ratio-max: Here dist-ratio controls the perturnation magnitude for crafting adversarial samples. dist-ratio-min and dist-ratio-max determines the range of dist-ratio. 
- dist-ratio-num: Here dist-ratio-num denotes how many dist-ratio values are sampled within the given range.<br>

**Note:**
We only consider Random Forest as the target classifier in the codes.

### Dataset
We include a small dataset `data_small.npy` as an example for testing the codes. The dataset is in the form of a matrix, where the first column stores the labels and each row represents one data instance. The dataset is differnt from the ones used in our paper. Please refer to the paper for the data sources.
