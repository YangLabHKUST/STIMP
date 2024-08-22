for i in {0..9}  
do  
python prediction/train_xgboost.py --index $i 
done
