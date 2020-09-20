# BERT NER

Use google BERT to do CoNLL-2003 NER !

Train model using Python and TensorFlow 2.0


# Requirements

- `python3`
- `pip3 install -r requirements.txt`

### Download Pretrained Models from Tensorflow offical models
- [bert-base-cased](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/cased_L-12_H-768_A-12.tar.gz) unzip into `bert-base-cased`

code for pre-trained bert from [tensorflow-offical-models](https://github.com/tensorflow/models/tree/master/official/nlp) 

# Run

## Single GPU

To evaluate on valid dataset:

`python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --output_dir=model_sep20 --max_seq_length=128 --do_train --num_train_epochs 3 --do_eval --eval_on dev`

To evaluate on test dataset:

`python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --output_dir=model_sep20 --max_seq_length=128  --num_train_epochs 3 --do_eval --eval_on test`

# Result

## BERT-BASE

### Validation Data
```
           precision    recall  f1-score   support

     MISC     0.8883    0.9143    0.9011       922
      PER     0.9693    0.9783    0.9738      1842
      LOC     0.9713    0.9575    0.9644      1837
      ORG     0.9148    0.9292    0.9219      1341

micro avg     0.9440    0.9509    0.9474      5942
macro avg     0.9451    0.9509    0.9479      5942
```
### Test Data
```
           precision    recall  f1-score   support

      LOC     0.9325    0.9353    0.9339      1668
      PER     0.9546    0.9629    0.9587      1617
      ORG     0.8892    0.9031    0.8961      1661
     MISC     0.7770    0.8291    0.8022       702

micro avg     0.9054    0.9205    0.9129      5648
macro avg     0.9068    0.9205    0.9135      5648
```



# Inference 

Refer infer_singlesent.py

```python
from bert import Ner

model = Ner("model_sep20/")

output = model.predict("Steve went to Paris")

print(output)
'''
    [
        {
            "confidence":  0.99796665,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.99980587,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence":  0.99981683,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9993082,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
'''
```
