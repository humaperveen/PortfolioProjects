# Abstract
The main purpose of this assignment is to detect propaganda using a subset of propaganda techniques identified in the Propaganda Techniques Corpus [1]. For the report, I analysed the given dataset by using n-gram modelling and Naïve Bayes Classifier, word2vec, neural networks (Convolutional Neural Network and Bidirectional Long Short Term Memory), and the BERT. The report will explain all the approaches in detail and the findings of the analysis. 
# Introduction
Identifying bias in news articles is not easy. Sometimes, news authors emphasize positive and negative aspects purposefully which leads to a lack of naturality of news and includes bias in the news articles [2]. In another scenario, the writer deliberately introduces bias into his writings, intending to influence his readers, this condition is known as propaganda. Propaganda is a shape of opinion or action by individuals or groups deliberately designed to change the perspective or actions of other individuals or groups regarding predetermined ends [3]. Da San Martino et al. [1] organized the SemEval 2020 Task 11: Detection of Propaganda Techniques in News Articles. 

This report focuses on two tasks. The first task is to build and evaluate at least 2 approaches for the classification of whether a sentence is a propaganda or not. The second task is to build and classify minimum of 2 approaches to identify propaganda techniques that have been used. Task 1 is a binary classification problem to detect a text fragment whether it contains propaganda or not. For task 2, there are a total of 8 propaganda techniques for span text present between <BOS> and <EOS> identifiers which makes it a multiclass classification problem. 

It is dangerous to ignore propaganda which shapes the information and changes our perspective on worldwide news. For a better understanding of how propaganda is used by news writers and what are the different techniques, Natural Language Processing (NLP) is helpful to develop various machine learning algorithms to detect and classify propaganda in a given information.
# Overview
## Dataset
The dataset contains two files one for training and another for testing. Each file is in tab-separated-value (.tsv) format with 2 columns as illustrated below. 
label	sentence
flag waving	I want to get <BOS> our soldiers <EOS> out. 
not propaganda 	No, <BOS> he <EOS> will not be confirmed.

The first column contains a label from a set of 9 possibilities which are:
1. flag waving
2. appeal to fear prejudice
3. causal simplification 
4. doubt
5. exaggeration,minimisation 
6. loaded language
7. name calling,labeling
8. repetition
9. not propaganda 

The additional tokens <BOS> and <EOS> indicate the beginning and end of the span of text (within the sentence) which is annotated with the given propaganda technique. There is a total of 2560 rows in the training data file and 640 rows in the testing data file.
**Assigned Task:**
	Build and evaluate at least 2 approaches to classify whether a span text in the given sentence is propaganda or not.
	Build and evaluate at least 2 approaches to classify the propaganda techniques.
**Approaches:**
	Text probability based on n-gram language models
	Text similarity or classification based on uncontextualised word embedding methods e.g., word2vec 
	Neural language models 
	Pretrained large language models e.g., BERT 
## Methodology
I applied all the given 4 approaches for task 1 and 3 approaches for task 2 along with other machine learning algorithms like Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, MultinomialNB, XGBoost, etc. The findings are included in the result section.
1. N-gram language model with Naïve Bayes Classifier
N-grams are simple, powerful, and easy-to-use techniques. N-grams are a continuous sequence of words (tokens) where the number of words is based on n, like in unigram there is a single word in each token in the given sentence. Then store the n-gram token and then count in the dictionary. They have a wide range of applications, like language models, semantic features, spelling correction, machine translation, text mining, etc. The major disadvantages are N-grams cannot deal Out Of Vocabulary (OOV) words and large sparsity concerns. 
A Naive Bayes classifier is a probabilistic machine learning model that's used for classification tasks based on the Bayes theorem with a strong assumption that features are independent of each other and give each feature the same level of importance. This faces the ‘zero-frequency problem’ where it assigns zero probability to a categorical variable whose category is only in the test dataset. 
The algorithm works by using Bayes’ theorem to calculate the probability of a given class, given the values of the input features. Bayes’ theorem states that the probability of a hypothesis (in this case, the class) given some evidence (in this case, the feature values) is proportional to the probability of the evidence given the hypothesis, multiplied by the prior probability of the hypothesis.
P(A│B)=  (P(B│A)P(A))/(P(B))
Where P(A|B) = Posterior Probability of a class given features, P(B|A) = Likelihood or probability of feature given class, P(A) = Prior Probability of class, P(B) = normalizing constant or probability of features 
2. TF-IDF Vectorization
In the context of information retrieval, TF-IDF (term frequency-inverse document frequency), reflects how important a word is to a document in a collection or corpus. It is often used as a weighting factor for information retrieval, and text mining. The TF-IDF value increases proportionally to the number of times a word appears in the document. 
Term Frequency is the relative frequency of a term (word) within a given document. It is obtained as the number of times a word appears in a text, divided by the total number of words appearing in the text.
TF(t,d)=  f_(t,d)/(∑_(t∈d)▒〖ft,d〗)
Inverse Document Frequency measures how common or rare a word is across all documents. IDF is calculated by dividing the total number of documents by the number of documents in the collection containing the term. 
IDF(t,D)=log⁡〖N/(|{d∈D,t∈d}|)〗
If a term does not appear in the corpus it leads to dividing by zero to overcome this we add 1.
TF-IDF is the product of these two terms TF and IDF.
TF-IDF(t,d,D)=TF(t,d)×IDF(t,D)
The value of a word increases proportionally to count, but it is inversely proportional to the frequency of the word in the corpus. but it has drawbacks that cause it to assign low values to relatively important words, TF-IDF cannot help carry semantic meaning and it can suffer from memory inefficiency because of the sparsity of the matrix. 
I used the Tf-Idf vectorizer from the scikit-learn to create vector representation. Then I trained different machine learning models (Logistic Regression, KNN Classifier, Decision Tree, Linear SVM, Random Forest, SGD Classifier, and XGBoost Classifier). The accuracy and f1-score obtained are included in the results section. 

3. Word Embedding methods (Gensim Word2Vec and GoogleNews vectors negative 300)
Word embedding is a technique where words are represented in vector form. These vectors are calculated from the probability distribution for each word appearing before or after another. Words of the same context usually appear together in the corpus, so they will be close in the vector space as well. The word2vec algorithms include skip-gram and CBOW models, using either hierarchical softmax or negative sampling. 
Continuous Bag of Words (CBOW): predict the current word given the context 
Skip-gram: predict the context given the current word 

In word2vec training embeddings are randomly initialized. It iterates over the text and computes objective functions for the target word using positive and negative samples. Stochastic gradient descent or backpropagation is used to improve the weights.

Word2vec is capable of capturing multiple degrees of similarity between words using simple vector arithmetic. Patterns like “man is to the woman as king is to the queen” can be obtained through arithmetic operations like “king” — “man” + “woman” = “queen” where “queen” will be the closest vector representation of the word itself. It is also capable of syntactic relationships like present & past tense & semantic relationships like country-capital relationships. 

A few limitations are: Word2Vec cannot handle out-of-vocabulary words. If we want to train word2vec in a new language, it requires a new embedding matrix. The semantic representation of a word relies only on its neighbours. It is not represented as state-of-the-art architecture.

I used Gensim Word2Vec and pre-trained GoogleNews vectors negative models. First, I converted sentences into a list of tokenized sentences and then I created a feature vector by averaging all embeddings of all sentences. I trained different machine learning models (Logistic Regression, KNN Classifier, Decision Tree, Linear SVM, Random Forest, SGD Classifier, and XGBoost Classifier) on these feature embeddings. The accuracy and f1-score obtained are included in the results section.

4. Neural Network (Convolutional Neural Network and Bidirectional Long Short Term Memory)
Neural networks are inspired by the neural networks in the human brain. They consist of neurons (also called nodes) which are connected to each other.
 
The input layer consists of feature vectors and the values are then fed forward to the hidden layer (one or more). At each connection, the value is fed forward, while the value is multiplied by a weight, and a bias is added to the value. This happens at every connection and at the end reached an output layer with one or more output nodes. All of those values are to be then summed and passed to an activation function. Various functions can be used depending on the layer or the problem. It is generally common to use the rectified linear unit (ReLU) for hidden layers, a sigmoid function for the output layer in a binary classification problem, or a softmax function for the output layer of multi-class classification problems.
In the beginning, weights are initialized with a random value and then they are trained with backpropagation methods like gradient descent to reduce the error between computed and actual output. The error is determined by a loss function whose loss is minimized with the optimizer. Adam is a widely used optimizer and it works well in most problems. 
**Convolutional Neural Networks (CNN)**
CNN is used mostly for image classification and to detect more complex patterns. But they are also used in NLP tasks like text classification, sequence classification, sentiment analysis, etc. When we are working with sequential data, like text, we work with one-dimensional convolutions, but the idea and the application stay the same as two-dimensional convolutions. 
It starts by taking a patch of input features depending on the size of the filter kernel, the dot product of this patch with the filter weights is mapped on the output features. This one-dimensional convolution is helpful to find patterns in text data. 
 
I used TensorFlow Keras deep learning API for the implementation of CNN. Keras Tokenizer utility is used to convert a token into a list of integers, then I applied padding because sentences have different lengths. Keras embedding layer is used to generate embedding for the 1D convolution layer. The dropout value is 0.5 with pool size 2 in the MaxPooling1D layer. The hidden layer activation function is ReLU and for the output layer softmax function is applied. The loss function is sparse categorical cross entropy and the optimizer is Adam.
**Bi-Directional Long Short Term Memory (BiLSTM)**
A bidirectional LSTM, often known as a BiLSTM, is a sequence processing model that consists of two LSTMs, the first model takes the input as it is, and the second model takes a backward direction copy of the sequence. This special architecture of BiLSTM effectively increases the quantity of data available, giving the algorithm better context. I used Glove word embedding for word vector representation.
 
5. Bidirectional Encoder Representations from Transformers (BERT)
Bidirectional Encoder Representations from Transformers (BERT) is a family of masked-language models introduced in 2018 by researchers at Google. Bert uses the bidirectional context of the language model, it tries to mask both left-to-right & right-to-left to create intermediate tokens to be used for the prediction tasks hence the term bidirectional. It uses a transformer network & attention mechanism that learns the contextual relationship between words. It is widely used for the tasks like named entity recognition and question answering.
## Model Evaluation Metrics
There are several evaluation metrics used to measure the performance of machine learning algorithms like, accuracy, precision, recall (sensitivity), f1-score, and confusion matrix [14]. 
**Confusion Matrix**
The confusion matrix is constructed based on True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN). 
	True Positive (TP): Correctly predicted positive values, i.e., actual class and predicted class are both true. 
	True Negative (TN): Correctly predicted negative values, i.e., actual class and predicted class are both false. 
	False Positive (FP): Incorrectly predicted positive values, i.e., actual class is false and predicted class is true. 
	False Negative (FN): Incorrectly predicted negative values, i.e., actual class is true and predicted class is false. 
 
**Accuracy** 
Accuracy is the most widely used and simple measure to test if a model is performing well. It calculates the correctly predicted instances out of the total number of predictions. Accuracy will range from 0 to 1, with 1 being the most accurate. I used the scikit-learn function accuracy_score to calculate this. 
Accuracy=  (( TP+TN))/((TP+TN+FP+FN))
**Precision** 
Precision is a measure that tells us the accuracy or precision of the model from predicted instances, i.e., actual positives from the total positive predictions. 
Precision=  TP/(TP+FP)
**Recall** 
Recall calculates the actual positives made out of all positive instances in the dataset, i.e., how many actual positives the model captures out of the total actual positive. 
Recall=  TP/(TP+FN)
**F1-Score** 
F1 – Score measures the harmonic mean of precision and recall. It is a good measure for an imbalanced dataset. 
F1-Score=  (2(Precision*Recall))/(Precision+Recall)
**Receiver operating characteristics (ROC) curve** 
The ROC curve is used to depict the model performance. The large Area Under Curve (AUC) is better. An AUC score of around 1 means excellent model performance with the discriminative ability for a target while an AUC score of around 0 shows lower performance [15, 16]. 
#Data Analysis and Results
I pre-processed sentences by converting them into lowercase, removing special characters and stop words. 
 
Task 1
For task one, I used n-gram modelling with a Naïve Bayes Classifier to detect propaganda in a given sentence. 

I experimented with 1-5 grams and got the highest frequency of the unigram by using the nltk NaiveBayesClassifier. 

For the second approach, I applied  tf-idf vectorizer and machine learning models, 
                  
MultinomialNB is performing better by removing stop words but the rest of the models are performing much better without stop words removal.
The third approach, Gensim word2vec and Google news vector negative for word embedding is used.

Performance for models better with google news negative vectors. By applying Gensim word embedding random forest classifier had the highest f1-score (0.63) and AUC value of 0.68 while using google news negative vectors performance of the XGBoost Classifier is highest with f1-score (0.72) and AUC value 0.80.
In the fourth approach, CNN and BiLSTM neural networks are used. 

Three convolutional layers are used with an input layer, embedding layer, conv1D layer, dropout, max-pooling layer, and a flatten layer, then all three layers are merged and a dense layer with output size 10 is used. The final output layer has 2 nodes. The conv1D layer has 32 filters, with kernel sizes 8,6, and 4 in three layers respectively. The dropout value is set to 0.5 to overcome the overfitting problem. In the maxpooling1D layer, the pool size is chosen 2. Flatten layer is used to make the multidimensional input one-dimensional, commonly used in the transition from the convolution layer to the fully connected layer. Then a dense layer with output size 10 is applied and finally, the output layer has 2 nodes. ReLU activation function is applied for the con1D layer and dense layer. I used the softmax activation function for the Output layer. Loss function ‘Sparse_categotical_crossentropy’  is used. Optimizer Adam, batch size 32, and the number of epochs 10 are applied. 
Next BiLSTM deep neural network model is applied to detect propaganda. 
Global Vector or GloVe is an unsupervised learning algorithm for obtaining vector representations for words. I used glove.6B.100d to get word vector representation. This pre-trained embedding matrix is used in the embedding layer. Hyperparameters for the LSTM layer are 600 units (300 for each direction), activation='tanh', recurrent_activation='sigmoid', dropout 0.3, and recurrent dropout 0.3. Then dense layer has 32 units and the activation function is ReLU. The final output layer has 2 units with the activation function softmax. The optimizer, loss function, and evaluation metrics are the same as the CNN model. 

Neural Network Model	Test Loss	Test Accuracy	F1-Score
CNN	1.68	0.57	0.55
BiLSTM	0.70	0.62	0.61
Table 2: Performance of CNN and BiLSTM models for Task 1
In the final approach, I tried pre-trained TFBertForSequenceClassification for transfer learning. I kept sentences without removing stop words because Bert uses the bidirectional context of a language model. In another approach, I trained Bert Model. I used the whole sentence and assign 1 to words in the span text and 0 to the rest of the sentence. Further, I used the encoded vector with the scikit-learn MLP Classifier to detect propaganda classification tasks. Input representation is the sum of token embeddings, segment embeddings, and masking strategy. The optimizer, loss function, and evaluation metrics are the same as the CNN model. 

Tokenized sentences: 
 ['the', 'obama', 'administration', 'misled', 'the', 'bos', 'american', 'people', 'eos', 'and', 'congress', 'because', 'they', 'were', 'desperate', 'to', 'get', 'a', 'deal', 'with', 'iran', 'said', 'sen']

 ['bos', 'but', 'who', 'authorized', 'this', 'strategic', 'commitment', 'eos', 'of', 'indefinite', 'duration', 'in', 'syria', 'when', 'near', 'two', 'decades', 'in', 'afghanistan', 'have', 'failed', 'to', 'secure', 'that', 'nation', 'against', 'the', 'return', 'of', 'al', 'qaida', 'and', 'isis']
------------------
Segment Ids: 
 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

 [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 
For Task 1 (whether a sentence contains propaganda or not), I got the best performance by training BERT on a dataset with the highest f1-score 0.79.

Task 2
Only observations containing propaganda techniques are taken into consideration for task 2. Total of 1291 and 309 observations containing various propaganda techniques in the training and testing dataset respectively.
Firstly, the tf-idf vectorizer is used for the vector representation of span text, and the performance of various models is evaluated.
 
The second approach is the same as for task 1. Findings are in the below figures.
     
With google news negative vector word embeddings performance of the different models is improved effectively and logistic regression had got highest f1-score of 0.49.
Third is the neural network approach, the same architecture is followed as described in task 1 except in CNN, conv1D layers have 64 filters and kernel size is 3,4, and 5 in three convolutional layers. 
 
The final approach for training Bert and transfer learning is the same as for task 1. Performance obtained by fine-tuning pre-trained TFBertForSequenceClassification Bert model on the dataset with the highest f1-score of 0.64.
 
# Conclusion
Based on the exploratory data analysis (EDA) and experimental analysis with various classifiers and neural network and transformer model (BERT), the efficient and accurate model for task 1 is BERT with MLP Classifier with f1-score 79.0% and for task 2 is fine-tuning BERT (TFBertForSequenceClassification) with the f1-score 64.0%. Ultimately, this proves the perfectness of the BERT model for propaganda detection tasks.
# Future work
In the future, the evaluation metrics for neural network models can be improved by tuning hyperparameters and by increasing the more hidden layers. The performance of BERT can also be further analyzed by adding BiLSTM or other neural networks together to build a more sophisticated model.
# References
1. Giovanni Da San Martino, Alberto Barr ́on-Ceden ̃o, Henning Wachsmuth, Rostislav Petrov, and Preslav Nakov. 2020. SemEval-2020 task 11: Detection of propaganda techniques in news articles. 
2. Garth Jowett and Victoria O’Donnell. 2006. Propaganda and persuasion. Thousand Oaks, Calif.:Sage, 4th ed edition. Includes bibliographical references (p. 369-391) and indexes.
3. R. Jackall. 1995. Propaganda. Main Trends of the Modern World. NYU Press.
4. https://radimrehurek.com/gensim/models/word2vec.html
5. Mikolov, Yih, and Zweig (NAACL, 20130: Linguistic regularities in continuous space word representations 
6. https://www.tensorflow.org/tutorials/images/cnn
7. https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 
8. https://scikit- learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html 
9. https://scikit- learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html 
10. https://scikit- learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html 
11. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html 
12. https://scikit- 
learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
13. https://scikit- 
learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html 
14. https://scikit-learn.org/stable/modules/model_evaluation.html 
15. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html 
16. https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx- 
glr-auto-examples-model-selection-plot-roc-py
17. Francois Chollet, Deep learning with Python, book: 2018

