python prediction/train_without_spatial_witheof.py  --method "iTransformer" --area $1
python prediction/train_without_spatial_witheof.py  --method "CrossFormer" --area $1
python prediction/train_without_spatial_witheof.py  --method "tsmixer" --area $1
python prediction/train_witheof.py  --method "MTGNN" --area $1
python prediction/train_as_image_witheof.py  --method "PredRNN" --area $1
