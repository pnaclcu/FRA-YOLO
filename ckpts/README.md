# Pretrained models


## Note
1. Download these file and put them into root directory in this repo, ``./ckpts``
2. All you need to do is specifiy the dataset path in `*.yaml` file
3. The suffix name in VisDrone dataset ("officail" or "modified") means the original dataset and the modified dataset respectively.


### For single dataset validation, you need to specify the dataset `*.yaml` and `.pt` files in `eval.py`, then run 
```
python eval.py
```

### For formated results validation in these four dataset, run the following command to reproduce our results.
```
python batch_eval.py
```

