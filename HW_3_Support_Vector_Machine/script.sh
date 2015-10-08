#!/bin/bash 

java cs475/Classify -mode train -algorithm pegasos -model_file model/bio.pegasos.model -data Data/bio.train 

java cs475/Classify -mode test -model_file model/bio.logisticRegression.model -data DATA/bio.dev -predictions_file prediction/bio.dev.logisticRegressionPred

java cs475/Classify -mode train -algorithm logistic_regression -model_file model/nlp.logisticRegression.model -data Data/nlp.train 

java cs475/Classify -mode test -model_file model/nlp.logisticRegression.model -data DATA/nlp.dev -predictions_file prediction/nlp.dev.logisticRegressionPred

java cs475/Classify -mode train -algorithm logistic_regression -model_file model/speech.logisticRegression.model -data Data/speech.train 

java cs475/Classify -mode test -model_file model/speech.logisticRegression.model -data DATA/speech.dev -predictions_file prediction/speech.dev.logisticRegressionPred

java cs475/Classify -mode train -algorithm logistic_regression -model_file model/finance.logisticRegression.model -data Data/finance.train 

java cs475/Classify -mode test -model_file model/finance.logisticRegression.model -data DATA/finance.dev -predictions_file prediction/finance.dev.logisticRegressionPred

java cs475/Classify -mode train -algorithm logistic_regression -model_file model/vision.logisticRegression.model -data Data/vision.train 

java cs475/Classify -mode test -model_file model/vision.logisticRegression.model -data DATA/vision.dev -predictions_file prediction/vision.dev.logisticRegressionPred

java cs475/Classify -mode train -algorithm logistic_regression -model_file model/easy.logisticRegression.model -data Data/easy.train 

java cs475/Classify -mode test -model_file model/easy.logisticRegression.model -data DATA/easy.dev -predictions_file prediction/easy.dev.logisticRegressionPred

java cs475/Classify -mode train -algorithm logistic_regression -model_file model/hard.logisticRegression.model -data Data/hard.train 

java cs475/Classify -mode test -model_file model/hard.logisticRegression.model -data DATA/hard.dev -predictions_file prediction/hard.dev.logisticRegressionPred