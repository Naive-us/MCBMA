# MCBMA
This directory contains code necessary to run the ERNIE-MCBMA.  ERNIE-MCBMA is a novel approach for Multi-Class Sentiment Analysis on Chinese Social Media.
# Dataset
The SMP2020-ewect dataset has been given in SMP2020/data/.
# Requirements
It is recommended to create an anaconda virtual environment to run the code. The python version is python-3.8. The pytorch version is 2.2.0.

Pre-trained model bert Download link：https://huggingface.co/google-bert/bert-base-chinese.  
Pre-trained model ernie Download link：https://huggingface.co/nghuyong/ernie-2.0-base-en.  
Word2Vec use sogou news Word + Character 300 d:https://github.com/embedding/chinese-word-vectors, [download address: https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ].
# Running the code
The pretrain_run.py is the main file for running the code.

```
python pretrain_run.py --model ERNIE-MCBMA
```

The run.py is a running file based on the Word2Vec model.

```
python run.py --model Word2Vec-MCBMA
```
