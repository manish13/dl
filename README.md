# Deep Learning in Characteristics-Sorted Factor Models
project for CS230 class

# Usage of "dl_alpha" function

## Input 

* "data": a dictionary of all kinds of data. The keys are 
  - "characteristics"
  - "stock_return"
  - "target_return" (i.e. portoflio return)
  - "factor" (i.e. benchmark factors)

* "layer_size": a list of hidden layer size where the last element is the number of deep factors to be constructed. For example, [32,16,8,4,2].

* "para": a dictionary of all training parameters. The keys are
  - "split": the way to split data into training set and test set. See "data_split" function.
  - "train_ratio": the proportion of training set if "split" is "future", between 0 and 1.
  - "batch_size": batch size for training. For example, 32.
  - "train_algo": optimization method. For example, tf.train.AdamOptimizer.
  - "learning_rate": parameter for "train_algo".
  - "activation": activation function when constructing deep characteristics. For example, tf.nn.tanh.
  - "epoch": the number of epochs for training.
  - "Lambda": tunning parameter for pricing error regularization
  - "Lambda2": tuning parameter for weight matrix regularization

## Output

* "factor": deep factors

* "deep_char": deep characteristics

* "losses" : train, val, test losses


  
