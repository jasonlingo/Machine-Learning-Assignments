package cs475;

import cs475.NeuralNetwork.FeatureMatrix;

import java.io.Serializable;

public abstract class Predictor implements Serializable {
	private static final long serialVersionUID = 1L;

	public abstract void train(FeatureMatrix featureMatrix);
	
	public abstract Label predict(Instance instance);
}
