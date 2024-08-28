for i in {0..9}  
do  
python prediction/train_without_spatial.py --index $i --method "iTransformer"
done
python prediction/train_without_spatial_imputation.py  --method "iTransformer"
