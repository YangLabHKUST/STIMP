for i in {0..9}  
do  
python prediction/train.py --index $i --method "MTGNN"
done
