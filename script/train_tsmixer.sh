for i in {0..9}  
do  
python prediction/train_without_spatial.py --index $i --area $1
done
python prediction/train_without_spatial_imputation.py --area $1
