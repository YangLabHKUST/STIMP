for i in {0..9}  
do  
python prediction/train.py --index $i --method "MTGNN" --area $1
done
python prediction/train_without_imputation.py --method "MTGNN" --area $1
