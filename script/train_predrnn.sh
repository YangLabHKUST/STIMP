for i in {0..9}  
do  
python prediction/train_as_image.py --index $i --area $1
done
python prediction/train_as_image_without_imputation.py --area $1
