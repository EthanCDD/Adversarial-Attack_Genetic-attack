# Adversarial attack and transferability
This is based on paper 'Generating Natural Language Adversarial Examples'. However, there are some changes:

-- Models are trained with PyTorch instead of TensorFlow

-- GPT-2 is applied to select synonyms instead of Google 1-billion LM which is also kept here. This increses the training speed very much

-- Sentiment analysis model achieves 90.32\% accuracy

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

However, the paths might need to be cared.
