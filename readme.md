# Adversarial attack and transferability
This is based on paper 'Generating Natural Language Adversarial Examples'

Firstly, user is required to download several datasets whereas user might not have access to run `.sh` file. Therefore, `chmod` can be used and an example is shown below

```
%%shell
cd ./BERT-LSTM-SentimentAnalysis
chmod a+x download_dataset.sh
./download_dataset.sh
mv aclImdb ../
```
You can train the model with code:
```
python train_lstm.py --learning_rate=0.0005 --nlayer=2 --bidirection=True --kept_prob=0.73
```
Finally, run the code to generate attack examples given by:
```
python train.py --nlayer=2 --data=imdb --sample_size=15000 --test_size=1000
```
Tensorflow is required to be version of `1.x`
