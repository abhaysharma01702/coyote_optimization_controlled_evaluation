# Is Metaheuristic Tuning Worth It? A Controlled Evaluation of Coyote Optimization for Deep Sentiment Classifiers

This repository contains the code for a controlled evaluation of the Coyote Optimization Algorithm (COA) for tuning the hyperparameters of deep sentiment classifiers. COA tunes a feed-forward network and a CNN+BiLSTM built on FastText embeddings, on Amazon Polarity, Yelp Polarity, Sentiment140 and a subset of Amazon Fine Food Reviews. The tuned models are compared with default configurations trained on exactly the same schedule, over ten independent training runs per configuration. COA is also compared with particle swarm optimization, random search and TPE at an equal evaluation budget.

The experiments run as a pipeline of Kaggle notebooks, in which each notebook reads the outputs of the earlier ones. Under these controls, COA tuning gave no consistent improvement over the default configurations, and the four optimizers performed alike. The training schedule and the architecture mattered more than the tuned hyperparameters.

## Dataset

All datasets used in this study are available at: https://www.kaggle.com/datasets/abhaysharma01702/coyote-dataset
