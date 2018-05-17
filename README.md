# EE596 Lab 4 -- Build An Recurrent Neural Network Language Model in Python

Course Webpage: [EE596 -- Conversational Artificial Intelligence](https://hao-fang.github.io/ee596_spr2018/)

## Task 1: Recurrent neural network (RNN) basics
The RNN has lots of applications in NLP, such as language modeling, part-of-speech tagging, and named entity recogition. In this part, you will learn the math behind RNNs and how to do the [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) with them.

### Steps
* Install [Numpy](http://www.numpy.org/).
* Please follow instruction in `notebook/rnn_basics.ipynb` to code an RNN unit.

## Task 2: Build an RNN language model (LM)

### Preparation
* First, based on Task 1, please complete `src/neuralnet/rnn_unit.py`. To test whether you have the right RNN unit, simply run the `python rnn_unit.py`.
* Please complete `src/neuralnet/rnn.py` by re-using the forward and backward functions in Task 1. Again, to test whether you have the right RNN, simply run `python rnn.py`
* Now you've finished the essential parts for a RNN LM. Next, we are going to apply the RNN LM to solve two problems: sorting numbers and character-level language modeling.

### Sorting numbers
* Please take a look at the data under `data/sorting_numbers`. For this problem, a training sample could be
	```
	4 1 2 5 6 <sort> 1 2 4 5 6
	```
* The numbers before `<sort>` are unsorted, and numbers after `<sort>` are sorted in ascending order. 
* In this task, we only look at number sequences with length 5 containing integer numbers from 0 to 9.
* To train an RNN LM for sorting numbers, you can run `./train_rnn.sort.sh`. During the training, you will see messages such as log-likelihood and [perpexlity](https://en.wikipedia.org/wiki/Perplexity). If the model is correct, the model perpexlity should be decreasing and the log-likelihood should be increasing.
* After the model converges, you can then test whether it can sort the number sequence properly by running `./decode_rnn_sort.sh`.
* We provide a relatively good model `pretrained_models/converged_sort.model` for this task. If you're not sure whether your models are good enough, you can modify the `inmodel` parameter in `decode_rnn_sort.sh` and compare the results with your models.

###  Character-level language modeling
Now we are going to train an RNN LM on the character sequence. In other words, the trained LM can predict next characters given those previous ones.

* To train the character RNN LM, you can run `./train_rnn_char_lm.sh`. Similar to the previous case, if the model is correct, the model perplexity should be decreasing and the log-likelihood should be increasing.
* After the model converges, you can then test whether it can generate a sentence properly by running `./sample_rnn_lm.sh`.
* We provide a relatively good model `pretrained_models/converged_char_lm.model` for this task. If you're not sure whether your models are good enough, you can modify the `inmodel` parameter in `sample_rnn_lm.sh` and compare the results with your models.

## Lab Checkoff
* Task 1: 
	* Show the output of your Python codes.
* Task 2: 
	* Make sure you pass tests in rnn_unit.py and rnn.py.
	* Tests the model for sorting numbers.
	* Tests the model for sampling a sentence starting with a specific character.

## Lab Report
* Please report your model perpexlity on sorting number for test set.
* Please report your model perplexity on character LM for test set.
* Discuss other findings and issues.
