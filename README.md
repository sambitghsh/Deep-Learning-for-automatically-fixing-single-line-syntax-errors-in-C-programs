*In this Programming Assignment I have attempted the LINE TO LINE FIXING MODEL*. 

*Here, to achieve the target, I tried three possible models. The first one is one hot encoder which is given in the assignment. In this case, the bleu score is almost 70, but to predict the valid.csv it was taking almost 20 minutes. That's why I changed the one-hot representation and use an index array. In this case, the bleu score decreases drastically to 29 and the time taken is almost the same as the first one. To make it more efficient and fast I tried another way using GRU with attention. Using this I got a bleu score of almost 65(refer to GRU_W_ATTN.ipynb). But the main advantage is it is taking approx 4-6 minute, which is almost 4 times faster than the previous two. *

*To do this assignment, I have referred to many online tutorials for TensorFlow and deep learning. Because of this, there can be some resembles between my code and the official documents of TensorFlow. Although I am attaching all the links which i used for this assignment. But I tried my utmost to understand what is deep learning and how it works by using this material.*


1.https://www.youtube.com/watch?v=bBBYPuVUnug&ab_channel=ConfEngine <br />
2.https://www.youtube.com/watch?v=f-JCCOHwx1c&t=713s&ab_channel=KrishNaik <br />
3.https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z <br />
4.https://www.tensorflow.org/api_docs <br />
5.https://www.tensorflow.org/tutorials/text/nmt_with_attention <br />
6.https://towardsdatascience.com/implementing-neural-machine-translation-with-attention-using-tensorflow-fc9c6f26155f <br />
7.https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint?version=nightly <br />
8.http://cs229.stanford.edu/syllabus.html <br />
9.https://www.youtube.com/watch?v=SysgYptB198&t=493s&ab_channel=DeepLearningAI <br />
10.https://medium.com/@dev.elect.iitd/neural-machine-translation-using-word-level-seq2seq-model-47538cba8cd7 <br />

