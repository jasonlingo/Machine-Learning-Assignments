package cs475;

import java.util.*;

/**
 * A Support Vector Machine class.
 * Created by Li-Yi Lin on 10/7/15.
 */
public class SVM extends Predictor{


    // Parameters
    private Map<Integer, Double> parameters;
    // Iteration limit
    private int iteration;
    // Initial pegasos_lambda;
    private double pegasos_lambda;
    // Label list
    private List<Integer> labels;
    // Feature vector list
    private List<FeatureVector> featureVectors;

    /**
     * Constructor
     * @param iteration
     * @param pegasos_lambda
     */
    public SVM(int iteration, double pegasos_lambda){
        this.iteration = iteration;
        this.pegasos_lambda = pegasos_lambda;

    }

    /**
     * A SVM traing process.
     * @param instances
     */
    @Override
    public void train(List<Instance> instances){
        // ===================================================================
        // Initialize training.
        // keep all the labels and find the number of features to build
        // parameters.
        // ===================================================================
        this.labels = new ArrayList<>();
        this.featureVectors = new ArrayList<>();

        int featureNum = 0;
        for (Instance inst : instances){
            Label label = inst.getLabel();
            FeatureVector fv = inst.getFeatureVector();
            if (label != null && fv != null){
                // Add labels, convert class label 0 to -1
                int labelInt = Integer.parseInt(label.toString());
                this.labels.add(labelInt == 0? -1:labelInt);
                // Add featureVector
                this.featureVectors.add(fv);
                // Find the total number of features
                Iterator it = fv.iterator();
                while (it.hasNext()){
                    Map.Entry pair = (Map.Entry)it.next();
                    if ((int)pair.getKey() > featureNum){
                        featureNum = (int)pair.getKey();
                    }
                }
            }
        }
        // Initialize the parameters
        this.parameters = new HashMap<>();
        for (int ii = 1; ii <= featureNum; ii++){
            this.parameters.put(ii, 0.0);
        }

        // ===================================================================
        // Start training.
        // ===================================================================
        int timeStep = 1;
        for (int iter = 1; iter <= this.iteration; iter++){
            for (int index = 0; index < this.featureVectors.size(); index++) {
                // Get the information (feature vector and label) about this instance.
                FeatureVector fv = this.featureVectors.get(index);
                int yLabel = this.labels.get(index);

                // Set eta
                double eta = 1.0 / (pegasos_lambda * (double)timeStep);

                // Get the value of (parameters . feature vector).
                double wx = innerProd(fv);
                // Update every parameter
                if (yLabel * wx < 1) {
                    for (Map.Entry<Integer, Double> pair : parameters.entrySet()){
                        double featValue = 0.0;
                        // Check whether the feature is existing
                        if (fv.containsKey(pair.getKey())){
                            featValue = fv.get(pair.getKey());
                        }
                        double newW = (1.0 - eta * pegasos_lambda) * pair.getValue() +
                                       eta * yLabel * featValue;
                        parameters.put(pair.getKey(), newW);
                    }
                } else { // yLabel * wx >= 1
                    for (Map.Entry<Integer, Double> pair : parameters.entrySet()){
                        double newW = (1.0 - eta * pegasos_lambda) * pair.getValue();
                        parameters.put(pair.getKey(), newW);
                    }
                }
                // Increase timeStep by one after process one instance
                timeStep++;
            }
        }
    }


    /**
     * Calculate hypothesis value of given feature vector.
     * @param fv
     * @return hypothesis value
     */
    private double innerProd(FeatureVector fv){
        double wx = 0.0;
        Iterator it = fv.iterator();
        while (it.hasNext()){
            Map.Entry pair = (Map.Entry)it.next();
            if (this.parameters.containsKey(pair.getKey())){
                wx += this.parameters.get(pair.getKey()) * (double)pair.getValue();
            }
        }
        return wx;
    }

    @Override
    public Label predict(Instance instance) {
        FeatureVector fv = instance.getFeatureVector();
        double wx = innerProd(fv);
        return new ClassificationLabel(wx >= 0? 1:0);
    }

}
