for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
do
  python imputation/train_stimp.py --missing_ratio $i --area $1
done