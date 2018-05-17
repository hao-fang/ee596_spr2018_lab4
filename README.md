# EE596 Lab 4 -- Build An Recurrent Neural Network Language Model in Python

Course Webpage: [EE596 -- Conversational Artificial Intelligence](https://hao-fang.github.io/ee596_spr2018/)

## Task 1: Recurrent neural network (RNN) basics
* This part you will learn the math behind RNNs and how to do the [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) with them.
	* If you are not familiar with RNNs, please follow this [link](https://en.wikipedia.org/wiki/Recurrent_neural_network).
	* In a nutshell, the RNN is the backbone for many sequential modeling problems. There are many sequential modeling problems in NLP, such as language modeling, part-of-speech tagging, and named entity recogition. Many non-sequence problems can be also cast into sequence modeling problems.
	* Here, we are going to code up the forward and backpropagation for RNNs using [Numpy](http://www.numpy.org/).

* Open the [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html) under the folder `notebook` and follow the instruction to code the RNN unit.


## Task 2: Build a RNN language model (LM)

### Preparation:
* First, based on Task 1, please complete `src/neuralnet/rnn_unit.py`. To test whether you have the right RNN unit, simply run the `python rnn_unit.py`.
* Please complete `src/neuralnet/rnn.py` by re-using the forward and backward functions in Task 1. Again, to test whether you have the right RNN, simply run `python rnn.py`
* Now you've finished the essential parts for a RNN LM. Next, we are going to apply the RNN LM to solve two problems: sorting numbers and character-level language modeling.

### Sorting numbers
* Please take a look at the data under `data/sorting_numbers`.
* For this problem, a training sample would be like 
```
4 1 2 5 6 <sort> 1 2 4 5 6
```
The numbers before <sort> are sorted in ascending order afterwards. Here, we only look at number sequences with length 5 containing integer numbers from 0 to 9.
* To train the RNN LM for sorting
```
$ ./train_rnn_sort.sh
```
During the training, there will be print-outs, such as loglikelihood and [perpexlity](https://en.wikipedia.org/wiki/Perplexity). If the model is correct, the model perpexlity should be decreasing and the loglikelihood should be increasing.
* After the model converges, you can then test whether it can sort the number sequence properly by
```
$ ./decode_rnn_sort.sh
```

###  Character-level language modeling
Here, we are going to train a LM on the character sequence. In other words, the trained LM can predict next characters given those previous ones.
* To train the character RNN LM,
```
$ ./train_rnn_char_lm.sh
```
Similar to the previous case, if the model is correct, the model perpexlity should be decreasing and the loglikelihood should be increasing.
* After the model converges, you can then test whether it can generate a sentence properly by
```
$ ./sample_rnn_lm.sh
```

* If you're not sure your models are good enough or not, we provide you with two relatively good models for both problems under expts folder.
* You can modify the inmodel in decode_rnn_sort.sh and sample_rnn_lm.sh to those models under expts to compare with your models.


## Lab Checkoff
* Task 1:
  * Show the output of your jupyter notebook.
* Task 2:
  * Show whether tests in rnn_unit.py and rnn.py are passed.
	* Tests the model for sorting numbers.
	* Tests the model for sampling a sentence starting with a specific character.

## Lab Report
* Please report your model perpexlity on sorting number for test set.
* Please report your model perplexity on character LM for test set.
* Discuss other findings and issues.
