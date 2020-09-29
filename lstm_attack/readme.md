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

You can train the LSTM-based model by running 'job_lstm_train_**.sh' file on cluster.

You can train the LSTM-based model of defense by running 'defense_train_**.sh' file on cluster.

You can generate adversarial examples without stop-words modification by running 'job_**.sh'

You can generate adversarial examples with stop-words modification by running 'job_stop_**.sh'

You can generate adversarial examples with LSTM-based model trained by modified training set by running 'job_defense_**.sh'

Tensorflow is required to be version of `1.x`

In addition, the paths of model parameters and others need to be cared.
