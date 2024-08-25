for i in {0..9}  
do  
python prediction/train.py --index $i
done
python prediction/train_without_imputation.py 