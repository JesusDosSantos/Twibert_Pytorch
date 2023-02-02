# Twibert_Pytorch
Binary Classification Model using BERT for classifying tweets from bots

Pytorch and BERT from Hugginface's transformers library to classify tweets as being written by a Twitter bot or by a human.
This code is part of a larger model aimed to classifying Twitter users as being a bot or not via RGCN's and utilizing their whole profile information.

The .py files are separated as GPU memory is limited at the time of testing, so separating preprocessing, training, and evaluation into three different scrips reduces the strain on the GPU memory.
This model was trained on a mobile NVIDIA RTX 3060 and produced the following results after evaluating over the whole data set which consists of around 1,700,000 tweets:

        Test Accuracy: 0.8109 Test Loss: 0.40737080574035645
                      precision    recall  f1-score   support
        
                   0       0.76      0.80      0.78      8315
                   1       0.85      0.82      0.83     11685
        
            accuracy                           0.81     20000
           macro avg       0.81      0.81      0.81     20000
        weighted avg       0.81      0.81      0.81     20000
