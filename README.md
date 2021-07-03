# DC-LCR-Rot-hop_plus_plus
Using Diagnostic classifiers to evaluate the information encoded in the LCR-Rot-hop++ model which is used in Aspect based sentiment classification


All software is written in PYTHON3 (https://www.python.org/) and makes use of the TensorFlow framework (https://www.tensorflow.org/).


## Installation Instructions:
### Dowload required files and add them to the data/external_data folder:
1. Download ontology: https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData
2. Download SemEval2016 Dataset: http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
3. Download Stanford CoreNLP:https://stanfordnlp.github.io/CoreNLP/download.html
4. Download the files for the LCR-Rot-hop++ model.


## Software explanation:
#### main environment:
- main_2.py: program to run . Each method can be activated by setting its corresponding boolean to True e.g. to run the lcr_Rot method set lcr_Rot = True.
- config.py: contains parameter configurations that can be changed such as: dataset_year, batch_size, iterations.
- utils.py: contains methods used in other files.
#### Aspect-Based sentiment classifiers:
- abs_classifiers/lcr_v4.py:  implementation for the LCR-Rot-hop++ algorithm version 4, a subclass of neural language model
- abs_classifiers/neural_language_model.py: implementation of neural language model algorithm, main class
#### data pre-processing steps:
- data_setup/ontology_tagging.py: code creating hypothesis related to the ontology.
#### layers for the neural language models:
- model_layers/attention_layers.py: implementation of the attention function
- model_layers/nn_layers.py: implementation of the Bi-LSTM, and softmax layer



## Related Work: ##
This code uses ideas and code of the following related papers:
- Zheng, S. and Xia, R. (2018). Left-center-right separated neural network for aspect-based sentiment analysis with rotatory attention. arXiv preprint arXiv:1802.00892.
- Schouten, K. and Frasincar, F. (2018). Ontology-driven sentiment analysis of product and service aspects. In Proceedings of the 15th Extended Semantic Web Conference (ESWC 2018), pages 608–623.
- Wallaart, O. and Frasincar, F. (2019). A hybrid approach for aspect-based sentiment analysis using a lexicalized domain ontology and attentional neural models. In The Semantic Web - 16th Extended Semantic Web Conference (ESWC 2019), pages 363–378.
- Hupkes, D., Veldhoen, S., and Zuidema, W. H. (2018). Visualisation and ’diagnostic classifiers’ reveal how recurrent and recursive neural networks process hierarchical structure. J. Artif. Intell. Res., 61:907–926.
