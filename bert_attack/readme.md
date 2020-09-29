# Transferability of adversarial attack
## Be careful the variables and file path in the files of '.sh'

-- Models are trained with PyTorch and Huggingface

-- GPT-2 is applied to select synonyms

-- Sentiment analysis model achieves 91.98\% accuracy

Firstly, user is required to download several datasets whereas user might not have access to run `.sh` file. Therefore, `chmod` can be used and an example is shown below

```
%%shell
cd ./BERT-LSTM-SentimentAnalysis
chmod a+x download_dataset.sh
./download_dataset.sh
mv aclImdb ../
```

You can train the BERT-based model by running 'b**_BToken_0.7.sh' file on cluster.

You can train the LSTM-based model of defense by running 'defense_train_**.sh' file on cluster.

You can generate adversarial examples with BERT-based model by running '**_job.sh'

You can investigate the transferability of adversarial attack by running 'transfer_**.sh'. However, you need to correctly add the path of the generated adversarial attack to code.

In addition, the paths of model parameters and others need to be cared.
