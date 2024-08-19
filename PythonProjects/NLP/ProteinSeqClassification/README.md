# Abstract
Proteins are essential to numerous biological functions, with their sequences determining their roles within organisms. Traditional methods for determining protein function are time-consuming and labor-intensive. This study addresses the increasing demand for precise, effective, and automated protein sequence classification methods by employing natural language processing (NLP) techniques on a dataset comprising 75 target protein classes. We explored various machine learning and deep learning models, including K-Nearest Neighbors (KNN), Multinomial Naïve Bayes, Logistic Regression, Multi-Layer Perceptron (MLP), Decision Tree, Random Forest, XGBoost, Voting and Stacking classifiers, CNN, LSTM, and transformer models (BertForSequenceClassification, DistilBERT, and ProtBERT). Experiments were conducted using amino acid ranges of 1-4 grams for machine learning models and different sequence lengths for CNN and LSTM models. The KNN algorithm performed best on tri-gram data with 70.0% accuracy and a macro F1 score of 63.0%. The Voting classifier achieved best performance with 74.0% accuracy and an F1 score of 65.0%, while the Stacking classifier reached 75.0% accuracy and an F1 score of 64.0%. ProtBERT demonstrated the highest performance among transformer models, with a accuracy 76.0% and F1 score 61.0% which is same for all three transformer models. Advanced NLP techniques, particularly ensemble methods and transformer models, show great potential in protein classification. Our results demonstrate that ensemble methods, particularly Voting Soft classifiers, achieved superior results, highlighting the importance of sufficient training data and addressing sequence similarity across different classes.
# Introduction
Proteins play crucial roles in living organisms, including catalyzing metabolic processes, replicating DNA, reacting to stimuli, providing structure, and transporting molecules [1]. Proteins are composed of long chains of amino acids, and the sequence of these amino acids determines the protein's structure and function [2]. Understanding the relationship between amino acid sequence and protein function has significant scientific implications, such as identifying errors in biological processes and clarifying protein synthesis mechanisms.
The study of proteins and other molecules to ascertain the function of many novel proteins has become the foundation of contemporary biological information science. Various methods have been developed to encode biological sequences into feature vectors and classify them using machine learning algorithms. In an experiment, Dongardive et al found that biological sequences are encoded into feature vectors using the N-gram algorithm. 717 sequences divided unevenly into seven classes make up the dataset used for the studies. The closest neighbors are determined using the Euclidean distance and the cosine coefficient similarity metrics [4]. In their 2017 study, Li M. et al. classified the protein sequences of GCPRs (G-protein Coupled Receptors). The dataset included 1019 different protein sequences from the GCPR superfamily.  These sequences have all been examined in UniProtKB. The data was pre-processed, and the feature selection methods Term Frequency - Inverse Document Frequency (TF-IDF) and N-gram were utilized [5].

In their work, Lee T. and Nguyen T. used the analysis of unprocessed protein sequences to learn dense vector representation. The information was gathered from the 3,17,460 protein sequences and 589 families in the Universal Protein Resource (UniProt) database. Using Global Vectors for Word Representation (GloVe), a distributed representation was made by encoding each sequence as a collection of trigrams that overlapped [6]. According to Vazhayil A. et al., a protein family is a group of proteins that have the same functions and share similar structures at the molecular and sequence levels. Although a sizable number of sequences are known, it is noted that little is known about the functional characteristics of the protein sequences. Swiss-Prot's Protein Family Database (Pfam), which has 40433 protein sequences from 30 distinct families, was used as the source of the data for this study. Redundancy in the dataset was checked, and it was discovered that there were no redundant protein sequences. To represent discrete letters as vectors of continuous numbers, the text data was first processed using Keras word embedding and N-gram [7].

Shinde et al considered the structural protein sequences with 10 classes and were able to get 90 % accuracy using a convolutional neural network [8]. A ResNet-based protein neural network architecture is proposed by Maxwell et al. for Pfam dataset families [9]. ProteinBERT is a deep language model created exclusively for proteins, according to Nadav et al. ProteinBERT's architecture combines local and global representations, enabling the processing of inputs and outputs from beginning to end.  Even with limited labelled data, ProteinBERT offers an effective framework for quickly developing protein predictors [10]. The ProtBert model was pre-trained on Uniref100 [11] a dataset consisting of 217 million protein sequences [12].


Traditional methods for determining protein functions, such as crystallography and biochemical studies, are time-consuming [13]. To enhance protein classification, we propose a system that utilizes machine learning, deep learning, and natural language processing (NLP) techniques. Natural Language Processing (NLP) has emerged as a powerful tool for classifying protein sequences. By treating protein sequences like text, NLP techniques, such as n-grams and word embeddings, enable efficient classification and functional prediction. Models like BERT enhance these processes by learning contextual relationships within the sequences, significantly improving the extraction of valuable insights from complex biological data. 

# Methodology
## Dataset
The dataset used in this study is the structural protein sequences dataset from Kaggle, derived from the Protein Data Bank (PDB) of the Research Collaboratory for Structural Bioinformatics (RCSB). It contains over 400,000 protein structural sequences, organised into two files: `pdb_data_no_dups.csv` (protein metadata) and `data_seq.csv` (protein sequences).

## Exploratory Data Analysis (EDA)

The initial preprocessing steps involved merging the two CSV files on the structure ID column, removing duplicates, dropping null values, and selecting data where the macromolecule type is protein.

### Sequence Length Analysis

In our analysis of the sequence lengths, as depicted in Fig 1, it is evident that the distribution is highly skewed. The majority of unaligned amino acid sequences fall within a character count range of 50 to 450. This skewness indicates a significant variation in sequence lengths within the dataset, which could have implications for downstream analyses and the computational approaches employed. Understanding the distribution of sequence lengths is crucial for optimizing alignment algorithms and improving the accuracy of subsequent bioinformatics analyses.

![Distribution of Sequence length](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig1.png) 
Fig 1. Distribution of Sequence length

### Amino acid Frequency Analysis

Amino acid sequences are represented with their corresponding 1-letter code, for example, the code for alanine is (A), arginine is (R), and so on. In our analysis, Fig 2 highlights the frequency distribution of amino acids in the dataset. It is evident that leucine (L) appears most frequently, succeeded by alanine (A), glycine (G), and valine (V). This observation aligns with the known biological abundance of these amino acids in various proteins. Additionally, our sequence encoding approach focused on the 20 standard amino acids, deliberately excluding the rare amino acids such as X (any amino acid), U (selenocysteine), B (asparagine or aspartic acid), O (pyrrolysine), and Z (glutamine or glutamic acid) to streamline the analysis and ensure consistency.

![Amino Acid Distribution](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig2.png) 
Fig 2. Amino Acid Distribution


## Machine Learning Methods
**Data Transformation**

Sequences and classes were initially converted into numerical form using CountVectorizer and LabelEncoder. The dataset was then split into training and test sets with an 80:20 ratio. To optimize the model's hyperparameters, RandomizedSearchCV was employed, ensuring the best possible configuration for each machine learning method.

**Methods**

Several machine learning algorithms were applied to the dataset. These included K-Nearest Neighbors (KNN), Multinomial Naive Bayes, Logistic Regression, Multilayer Perceptron, and Decision Tree Classifier. Additionally, various ensemble methods were utilized, such as Random Forest Classifier, XGBoost Classifier, Voting Classifier, and Stacking Classifier, to improve classification performance through model combination and aggregation.


## Deep Learning Methods
**Further pre-processing of sequences with Keras**

Further pre-processing of sequences was conducted using Keras. The sequences were tokenized, translating each character into a number, and padded to ensure uniform length with maximum lengths of 100, 256, and 512 for evaluating model efficiency and performance. For instance, a sequence ‘GSAFCNLARCELSCRSLGLLGKCIGEECKCVPY’ will be converted into encoding like this [ 6, 16, 1, 5, 2, 12, 10, 1, 15, 2, 4, 10, 16, 2, 15, 16, 10, 6, 10, 10, 6, 9, 2, 8, 6, 4, 4, 2, 9, 2, 18, 13, 20]. 
The padded sequence would be like the below:
[ 6, 16, 1, 5, 2, 12, 10, 1, 15, 2, 4, 10, 16, 2, 15, 16, 10, 6, 10, 10, 6, 9, 2, 8, 6, 4, 4, 2, 9, 2, 18, 13, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

The data was split into training, validation, and test sets in a 70:10:20 ratio. For class transformation, sequences with counts greater than 100 were selected, and labels were transformed to one-hot representation using LabelBinarizer. Class weights were assigned using sklearn’s compute_class_weight module to address class imbalance. Early stopping was implemented as a regularization technique to prevent overfitting, with performance monitored after each epoch. The deep learning models used included Long Short-Term Memory (LSTM) and Convolutional Neural Network 1D (CNN 1D).

1. **LSTM**
   
An embedding layer for the mapping input layer is used. Then, a layer of CuDNNLSTM is used for faster implementation. The dropout layer with 20% neuron dropout applied that adds representational capacity to the model and prevents overfitting. Eventually, a dense layer is used as the output layer with softmax activation. The model is trained using categorical cross entropy and is compiled using Adam optimizer. An outline of proposed implementation steps is drawn in Fig 3.

![LSTM Implementation Outline](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig3.png) 
Fig 3. LSTM Implementation Outline

2. **CNN 1D**

Recent success in NLP suggests using word embeddings which are already implemented as a Keras Embedding layer. Note that in this dataset, there are only 20 different words (for each amino acid). Instead of using every n-gram, using 1D-convolution on the embedded sequences is considered. The size of the convolutional kernel can be seen as the size of n-grams and the number of filters as the number of words as shown in Fig 4 below.

![CNN1D Mechanism](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig4.png)  
Fig 4. CNN1D Mechanism

A multichannel architecture is used for Convolutional neural network implementation. There are 3 channels. Filter size is 128 and kernel size is 12 in first channel. In second channel filter size is 64 and kernel size is 6 while in third channel filter size and kernel size are  32 and 3 respectively. These filters slide across the input sequence, performing element-wise multiplications and summations to produce feature maps. The filters capture local patterns and dependencies within the sequence of kernel size like n-gram. Each filter learns different features, allowing the model to capture diverse aspects of the input sequence. The filters with different widths help to capture patterns of varying lengths. An activation function ReLU (Rectified Linear Unit) is applied element-wise to introduce non-linearity into the model which sets negative values to zero. 

Dropout value is set 0.2, dropout works by "dropping out" a fraction of the neurons in a layer with a specified probability of 0.2. This means that the output of those neurons is temporarily ignored or set to zero. Maxpooling1D layer pool size is kept 2. It selects the maximum value within a fixed window size, reducing the dimensionality of the feature maps while preserving the most salient information. The output of the pooling layers is flattened into a one-dimensional vector. After flattening three channels are concatenated together. Then, a fully connected dense layer is  applied that adds representational capacity to the model. Finally, a dense layer is used as the output layer with softmax activation. The model is trained using categorical cross entropy and is compiled using Adam optimizer. An outline of proposed implementation of convolutional neural network is drawn in Fig 5 below.

![Multi-channel CNN Implementation Outline](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig5.png)
Fig 5. Multi-channel CNN Implementation Outline

 
## Bidirectional Encoder Representations from Transformers (BERT) Models

Bidirectional Encoder Representations from Transformers (BERT) is a family of masked-language models introduced in 2018 by researchers at Google. Bert uses the bidirectional context of the language model, it tries to mask both left-to-right & right-to-left to create intermediate tokens to be used for the prediction tasks hence the term bidirectional. It uses a transformer network & attention mechanism that learns the contextual relationship between words. It is widely used for the tasks like named entity recognition and question answering.

Three different BERT models were used in this study: BertForSequenceClassification, DistilBERT, and ProtBERT.

### Implementation Steps

a.	**Data Pre-processing:** The preprocessed PDB protein dataset, consisting of protein sequences and their corresponding labels, was loaded for analysis. The dataset was split into training, validation, and testing sets (80:10:10) to evaluate the model's performance. To prepare the protein sequences for input into the model, the BERT tokenizer was used. This tokenizer converts each sequence into a list of tokens, smaller units that BERT can process. The tokens were then converted into input features, including token IDs, attention masks, and segment IDs, which help BERT understand the relationships within the protein sequences.

b.	**Model Initialization:** The model was initialized with pre-trained weights, capturing knowledge from a large corpus of text data to enhance performance on the protein classification task.

c.	**Training:** Hyperparameters for training, such as learning rate (2e-5), batch size (4), and number of epochs (30), were set. During each epoch, the model learned from the protein sequences and their corresponding labels in batches. For each batch, a forward pass was performed, obtaining predicted logits. The cross-entropy loss between the predicted logits and true labels was computed to measure performance. A backward pass computed gradients of the loss with respect to the model's parameters. The optimizer, AdamW, updated the model's parameters based on the gradients to minimize the loss.

d.	**Validation:** After training, the model's performance was evaluated on the validation set containing unseen protein sequences. A forward pass was performed to obtain predicted logits, which were compared to true labels to assess accuracy, precision, recall, and F1-score.

e.	**Fine-tuning and Optimization:** If the model's performance was unsatisfactory, hyperparameters were fine-tuned or different optimization techniques were experimented with. The training loop was repeated with updated settings until the desired performance was achieved.

f.	**Evaluation and Save model:** Once trained and optimized, the model's weights and architecture were saved for future use. Evaluation was performed on testing data to determine the model's performance on unseen data.

## Model Evaluation Metrics

There are several evaluation metrics used to measure the performance of machine learning algorithms like, accuracy, precision, recall (sensitivity), f1-score, and confusion matrix [27]. 

**Confusion Matrix**

The confusion matrix is constructed based on True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN). 
	True Positive (TP): Correctly predicted positive values, i.e., actual class and predicted class are both true. 
	True Negative (TN): Correctly predicted negative values, i.e., actual class and predicted class are both false. 
	False Positive (FP): Incorrectly predicted positive values, i.e., actual class is false and predicted class is true. 
	False Negative (FN): Incorrectly predicted negative values, i.e., actual class is true and predicted class is false. 
 
**Accuracy** 

Accuracy is the most widely used and simple measure to test if a model is performing well. It calculates the correctly predicted instances out of the total number of predictions. Accuracy will range from 0 to 1, with 1 being the most accurate. I used the scikit-learn function accuracy_score to calculate this. 

$$Accuracy=  (( TP+TN))/((TP+TN+FP+FN))$$

**Precision** 

Precision is a measure that tells us the accuracy or precision of the model from predicted instances, i.e., actual positives from the total positive predictions. 

$$Precision=  TP/(TP+FP)$$

**Recall** 

Recall calculates the actual positives made out of all positive instances in the dataset, i.e., how many actual positives the model captures out of the total actual positive. 

$$Recall=  TP/(TP+FN)$$

**F1-Score** 

F1 – Score measures the harmonic mean of precision and recall. It is a good measure for an imbalanced dataset.

$$F1-Score=  (2(Precision*Recall))/(Precision+Recall)$$

# Results

## Machine Learning

Fig 6 presents a visual comparison of the F1 scores achieved by various machine learning models across different n-gram ranges (uni-gram, bi-gram, tri-gram, and 4-gram). The analysis indicates that models such as K-Nearest Neighbors (KNN), Random Forest, XGBoost, Voting, and Stacking Classifiers maintain consistent F1 scores between 60.0% and 65.0% regardless of the n-gram range, demonstrating their robustness to changes in text representation. The Multi-Layer Perceptron (MLP) shows the lowest performance with the uni-gram model, but its performance improves with higher n-gram ranges, plateauing between the 3-gram and 4-gram models. On the other hand, Multinomial Naïve Bayes and Logistic Regression models exhibit significant improvements in accuracy and F1 score when moving from bi-gram to 3-gram, indicating a strong dependency on n-gram range. Notably, ensemble methods, particularly the Voting Soft classifier, outperform individual models, achieving the highest accuracy (74%), weighted F1 score (74%) and macro f1 score (65%), emphasizing their effectiveness in handling imbalanced datasets.

![Comparison of models’ macro f1 score for 1-4 grams](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig6.jpg)
Fig 6. Comparison of models’ macro f1 score for 1-4 grams

## Deep Learning
 
6 experiments are carried out for three different protein sequence lengths with and without class weight for CNN and 3 experiments for LSTM. The LSTM model shows moderate training and validation accuracy, indicating a reasonable but not optimal fit to both the training and validation data. The loss values indicate a moderate level of error in predictions. The F1 scores, particularly the macro F1 score, suggest that while the model performs reasonably well on more frequent classes, its performance on less frequent classes is less reliable and quite variable 

![Accuracy and loss for LSTM model](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig7.png) 
Fig 7. Accuracy and loss for LSTM model for different sequence length (512, 256, 100)

The CNN model shows good training accuracy, indicating effective learning, though with notable performance variability. The training loss reflects accurate predictions but inconsistent errors. Validation metrics indicate moderate generalization with more stable performance than training. F1 scores suggest the model handles frequent classes well, but performance varies across different classes.

Fig 8 illustrates that the CNN model's accuracy and loss trends are consistent for sequence lengths of 256 and 512, indicating stable model performance across these lengths. However, when the sequence length is reduced to 100, the model struggles to accurately classify protein sequences, resulting in higher loss values. This increased loss suggests a significant drop in model performance, highlighting that shorter sequence lengths are less effective for identifying the correct class in protein sequence classification tasks. The data suggests that longer sequences provide more information, leading to more reliable model predictions and better overall performance.
 
![Accuracy and loss for CNN model](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig8.png)
Fig 8. Accuracy and loss for CNN model for different sequence length (512, 256, 100)

![Comparison of Deep learning approach](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig9.png)  
Fig 9. Comparison of Deep learning approach

From the Fig 19, it is clearly visible that CNN model without class weight demonstrates superior performance in terms of accuracy and F1 scores, indicating better overall classification capability. The CNN model with class weight shows a slight decrease in performance, suggesting that while class weighting can help in addressing class imbalances, it may also introduce complexities that reduce overall effectiveness. The LSTM model underperforms compared to both CNN models, highlighting its limitations in identifying hidden patterns in longer sequences.

## Transformer Models

Fig 10 presents the performance of three different BERT-based transformer models on the test dataset, highlighting key metrics such as test loss, accuracy, and F1 scores. All three models show competitive accuracy, with ProtBERT achieving the highest at 77%. BertForSequenceClassification and DistilBERT both have an accuracy of 73%. ProtBERT has the lowest test loss (1.37), indicating better performance in terms of prediction error. DistilBERT shows a lower test loss (1.48) compared to BertForSequenceClassification (1.78).

ProtBERT again leads with a weighted F1 score of 76%, indicating it performs well across classes considering the class distribution. BertForSequenceClassification and DistilBERT had similar performances with slight variations. BertForSequenceClassification and DistilBERT have similar weighted F1 scores, 73% and 72%, respectively. All models have the same macro F1 score of 61%, reflecting their balanced performance across all classes, irrespective of class distribution.

![Comparison of Bert Models](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig10.png) 
Fig 10. Comparison of Bert Models

## Error Analysis
    
![Error Analysis](https://github.com/humaperveen/PortfolioProjects/blob/main/PythonProjects/NLP/ProteinSeqClassification/assets/Fig11.png) 
Figure 11. Error Analysis

In general, proteins can be different types of enzymes, signalling proteins, structural proteins, and a variety of other options. Since many proteins are designed to bind in the same locations as one another, they frequently exhibit extremely similar properties. A Hydrolase enzyme and a Hydrolase inhibitor protein, for instance, will have similar structures since they focus on the same regions. 

high precision and high recall that means model is able to identify them because these classes either have enough sample or their structure are not similar to others for instance, Allergan, apoptosis, immune system, isomerase, hydrolase. 

Classes have high precision and low recall that means model is not identify them correctly, example RNA binding proteins, DNA binding proteins, and transcription proteins, all share characteristics with gene regulator proteins and Cell invasion, Cell adhesion having similarity with cell cycle that make model difficult to identify them. 

Classes are showing low precision and high recall like electron transport and oxygen storage, model is able to well detect these classes but also includes observations from other classes as well, for instance oxygen storage might be make model misleading for oxygen transport. 

Classes e.g. Phosphotransferase, Transcription inhibitor having low precision and low recall which means model is not able to identify correct classes and model is not doing well on the entire test dataset to find correct classes. On the other side some classes like, Ribosome having low precision and low recall because of similarity with ribosomal protein (fundamental building blocks for ribosome) that means model is not able to detect correct class.

 
# Conclusion
The study demonstrates that NLP techniques can significantly enhance protein sequence classification. The use of n-grams proved effective in improving classifier performance, and ensemble methods showcased their potential in handling imbalanced datasets. While CNN outperformed LSTM in handling longer sequences, transformer models, particularly ProtBERT, demonstrated superior accuracy and F1 scores, albeit with higher computational requirements. 

The findings of this study highlight the significant potential of natural language processing (NLP) techniques in protein sequence classification. Given the growing amount of biological data, efficient and automated methods are essential. The results demonstrate that various machine learning models, when applied to amino acid n-grams, can achieve noteworthy accuracy and F1 scores, underscoring the effectiveness of this approach.

The K-Nearest Neighbors (KNN) algorithm performed particularly well on tri-gram data, indicating its strength in capturing local sequence similarities. In contrast, models like Logistic Regression, MLP, and Random Forests showed their highest performance with 4-gram data, suggesting that these models benefit from a broader contextual understanding provided by longer n-grams.

The transformer models, particularly ProtBERT, showed promise with competitive F1 scores, although they required significant computational resources. This emphasizes the importance of access to high-performance computing facilities for training such models efficiently. The challenges faced due to limited GPU access and session expiry constraints highlight a practical limitation in the current study, suggesting a need for more robust computational infrastructure for future research.

Despite these promising results, several challenges remain. The primary source of error across models was the imbalanced dataset, with some classes having significantly fewer samples. This imbalance likely hindered the models' ability to generalize well across all classes. Future work could explore advanced techniques for handling class imbalance, such as data augmentation or more sophisticated weighting schemes.

The similarity between sequences from different classes also posed a challenge, potentially confusing the models and reducing classification accuracy. Advanced sequence embedding techniques or incorporating additional biological context could help mitigate this issue.

Furthermore, the results indicate that while CNNs and LSTMs showed reasonable performance, transformer models like BERT variants provided more consistent results across different sequence lengths and configurations. This suggests that transformers may offer a more robust framework for protein sequence classification, benefiting from their ability to capture long-range dependencies and contextual information effectively.

In summary, this study demonstrates the feasibility and effectiveness of using NLP techniques for protein sequence classification. The insights gained here pave the way for further exploration and optimization of these methods, with the potential to significantly enhance our ability to analyze and interpret complex biological data. Future research should focus on addressing the challenges of class imbalance and sequence similarity, as well as leveraging more advanced computational resources to fully realize the potential of these techniques.

# Future work
In the future, the evaluation metrics for neural network models can be improved by tuning hyperparameters and by increasing the more hidden layers. The performance of BERT can also be further analyzed by adding BiLSTM or other neural networks together to build a more sophisticated model.

# References
1.	Protein, https://en.wikipedia.org/wiki/protein
2.	Amino acids, https://en.wikipedia.org/wiki/Amino_acid
3.	Essential amino acids: chart, abbreviations and structures, https://www.technologynetworks.com/applied-sciences/articles/essential-amino-acids-chart-abbreviations-and-structure-324357
4.	Dongardive, J., & Abraham, S. (2016). Protein sequence classification based on N-gram and K-nearest neighbor algorithm. In H. Behera & D. Mohapatra (Eds.), Computational intelligence in data mining—Volume 2 (Vol. 411, pp. 185-194). Springer. https://doi.org/10.1007/978-81-322-2731-1_15
5.	Li, M., Ling, C., & Gao, J. (2017). An efficient CNN-based classification on G-protein coupled receptors using TF-IDF and N-gram. IEEE Symposium on Computers and Communications (ISCC), 1-8. https://doi.org/10.1109/ISCC.2017.8024644
6.	Lee, T., & Nguyen, T. (2019). Protein family classification with neural network. Stanford University, 1-9.
7.	Vazhayil, A., Vinayakumar, R., & Soman, K. P. (2019). DeepProteomics: Protein family classification using shallow and deep networks. Center for Computational Engineering and Networking (CEN), 1-17. https://doi.org/10.1101/414631
8.	Shinde, A., & D’Silva, M. (2019). Protein sequence classification using natural language processing. International Journal of Engineering Development and Research, 169-175.
9.	Bileschi, L. M., Belanger, D., Bryant, D., et al. (2019). Using deep learning to annotate protein universe. bioRxiv. https://doi.org/10.1101/626507
10.	Brandes, N., Ofer, D., Peleg, Y., Rappoport, N., & Linial, M. (2022). ProteinBERT: A universal deep-learning model of protein sequence and function. Bioinformatics, 38(8), 2102-2110. https://doi.org/10.1093/bioinformatics/btac020
11.	Uniprot. (n.d.). Uniref100. Retrieved from https://www.uniprot.org/help/downloads
12.	Elnaggar, A., Heinzinger, M., Dallago, C., et al. (2020). ProtTrans towards cracking the language of life. bioRxiv. https://www.biorxiv.org/content/early/2020/07/21/2020.07.12.199554
13.	Kaggle. (n.d.). Structural protein sequences. Retrieved from https://www.kaggle.com/datasets/shahir/protein-data-set
14.	Hochreiter, S., & Obermayer, K. (2005). Sequence classification for protein analysis. pp. 1-2.
15.	ResearchGate. (n.d.). The self-attention mechanism calculation process. Retrieved from https://www.researchgate.net/figure/The-self-attention-mechanism-calculation-process-We-can-get-three-vectors-that-are-a_fig4_344301413
16.	Wang et al. (2020). Biomedical document triage using a hierarchical attention-based capsule network. *BMC Bioinformatics, 21*(Suppl 13), 380. https://doi.org/10.1186/s12859-020-03673-5

17. https://radimrehurek.com/gensim/models/word2vec.html
18. Mikolov, Yih, and Zweig (NAACL, 20130: Linguistic regularities in continuous space word representations 
19. https://www.tensorflow.org/tutorials/images/cnn
20. https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 
21. https://scikit- learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html 
22. https://scikit- learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html 
23. https://scikit- learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html  
24. https://scikit- 
learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
25. https://scikit- 
learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html 
26.	https://xgboost.readthedocs.io/en/stable/
27. https://scikit-learn.org/stable/modules/model_evaluation.html 
28. Francois Chollet, Deep learning with Python, book: 2018

