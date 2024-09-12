for i in {0..9}  
do  
python prediction/train_xgboost.py --index $i --area $1
done
python tmp/merge_predictions.py --area $1
# python prediction/train_xgboost_without_imputation.py --area $1
