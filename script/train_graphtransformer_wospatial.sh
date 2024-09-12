for i in {0..9}  
do  
python prediction/train.py --index $i --area $1 --method GraphTransformer_ws
done
