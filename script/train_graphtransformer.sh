for i in {0..9}  
do  
python prediction/train.py --index $i --area $1
done
python prediction/train_without_imputation.py --area $1
