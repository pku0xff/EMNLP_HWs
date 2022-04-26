mkdir "data/sst"
mkdir "data/yelp"
python preprocess.py --dataset "sst"
python preprocess.py --dataset "yelp"
# compare between models
mkdir "result"
python run.py --model "LogLinear" \
              --dataset "sst" \
              --batch_size 100 \
              --lr 0.01 \
              --epoch 8 \
              --save_path 'result/result_1.txt'
python run.py --model "NaiveBayes" \
              --dataset "sst" \
              --n_features 200 \
              --save_path 'result/result_2.txt'

python run.py --model "LogLinear" \
              --dataset "yelp" \
              --save_path 'result/result_3.txt'
python run.py --model "NaiveBayes" \
              --dataset "yelp" \
              --n_features 5000 \
              --save_path 'result/result_4.txt'


# different feature numbers
python run.py --model "LogLinear" \
              --dataset "sst" \
              --n_features 500 \
              --batch_size 100 \
              --lr 0.01 \
              --epoch 8 \
              --save_path 'result/result_5.txt'

python run.py --model "LogLinear" \
              --dataset "sst" \
              --n_features 5000 \
              --batch_size 100 \
              --lr 0.01 \
              --epoch 8 \
              --save_path 'result/result_6.txt'


# different methods to select features
python run.py --model "LogLinear" \
              --dataset "sst" \
              --method "IG" \
              --n_features 2000 \
              --batch_size 100 \
              --lr 0.01 \
              --epoch 8 \
              --save_path 'result/result_7.txt'

python run.py --model "NaiveBayes" \
              --dataset "sst" \
              --method "IG" \
              --n_features 200 \
              --save_path 'result/result_8.txt'