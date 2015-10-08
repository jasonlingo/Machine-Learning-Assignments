#!/bin/bash 
bio
java cs475/Classify -mode train -algorithm pegasos -model_file model/bio.pegasos.model -data Data/bio.train 

java cs475/Classify -mode test -model_file model/bio.pegasos.model -data DATA/bio.dev -predictions_file prediction/bio.dev.pegasosPred

nlp
java cs475/Classify -mode train -algorithm pegasos -model_file model/nlp.pegasos.model -data Data/nlp.train 

java cs475/Classify -mode test -model_file model/nlp.pegasos.model -data DATA/nlp.dev -predictions_file prediction/nlp.dev.pegasosPred

speech
java cs475/Classify -mode train -algorithm pegasos -model_file model/speech.pegasos.model -data Data/speech.train 

java cs475/Classify -mode test -model_file model/speech.pegasos.model -data DATA/speech.dev -predictions_file prediction/speech.dev.pegasosPred

finance
java cs475/Classify -mode train -algorithm pegasos -model_file model/finance.pegasos.model -data Data/finance.train 

java cs475/Classify -mode test -model_file model/finance.pegasos.model -data DATA/finance.dev -predictions_file prediction/finance.dev.pegasosPred

vision
java cs475/Classify -mode train -algorithm pegasos -model_file model/vision.pegasos.model -data Data/vision.train 

java cs475/Classify -mode test -model_file model/vision.pegasos.model -data DATA/vision.dev -predictions_file prediction/vision.dev.pegasosPred

easy
java cs475/Classify -mode train -algorithm pegasos -model_file model/easy.pegasos.model -data Data/easy.train 

java cs475/Classify -mode test -model_file model/easy.pegasos.model -data DATA/easy.dev -predictions_file prediction/easy.dev.pegasosPred

hard
java cs475/Classify -mode train -algorithm pegasos -model_file model/hard.pegasos.model -data Data/hard.train 

java cs475/Classify -mode test -model_file model/hard.pegasos.model -data DATA/hard.dev -predictions_file prediction/hard.dev.pegasosPred