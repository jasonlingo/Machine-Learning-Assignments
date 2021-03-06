package cs475;

/**
 * Created by Li-Yi Lin on 9/21/15.
 */

import java.util.*;

/**
 * A class that implements logistic regression model.
 */
public class LogisticRegression extends Predictor {


    // ===================================================================
    // Instance variables.
    // ===================================================================
    // Used for each feature
    private Map<Integer, Double> parameters;
    private List<FeatureVector> featureVectors;
    private List<Integer> labels;

    // Iteration limit
    private int iteration;

    // Parameters for eta.
    private final double IJ = 1.0;

    // eta0
    private double etaZero;

    // Store each eta at step i for feature j
    private Map<Integer, Double> etas;

    // Partial gradient of the objective function at time t for feature j
    private Map<Integer, Double> Fij;

    /**
     * Constructor, initializing variables.
     * @param iteration
     * @param eta0
     */
    public LogisticRegression(int iteration, double eta0){
        this.iteration = iteration;
        this.etaZero = eta0;
        this.parameters = new HashMap<>();
        this.etas = new HashMap<>();
        this.Fij = new HashMap<>();
        this.labels = new ArrayList<>();
        this.featureVectors = new ArrayList<>();
    }


    /**
     * Train a logistic regression model using stochastic gradient descent and
     * AdaGrad (adaptively choosing the learning rate) techniques.
     * @param instances
     */
    @Override
    public void train(List<Instance> instances){
        // ===================================================================
        // Initialize training.
        // keep all the labels and find the number of features to build
        // parameters.
        // ===================================================================
        int featureNum = 0;
        for (Instance inst : instances){
            Label label = inst.getLabel();
            FeatureVector fv = inst.getFeatureVector();
            if (label != null && fv != null){
                // Add labels
                this.labels.add(Integer.parseInt(label.toString()));
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
        // Initialize the value in etas', parameters, and Fij
        for (int ii = 1; ii <= featureNum; ii++){
            this.parameters.put(ii, 0.0);
            this.etas.put(ii, etaZero);
            this.Fij.put(ii, 0.0);
        }


        // ===================================================================
        // Start training
        // ===================================================================

        for (int iter = 0; iter < this.iteration; iter++){
            for (int index = 0; index < this.featureVectors.size(); index++) {
                // Get the information (feature vector and label) about this instance.
                FeatureVector fv = this.featureVectors.get(index);
                int yLabel = this.labels.get(index);

                // Get the value of (parameters . feature vector).
                double wx = hypothesis(fv);

                // Stochastic Gradient descent
                updateParameter(fv, wx, yLabel);
            }
        }
    }


    /**
     * Classify the instance.
     * @param instance
     * @return a predicted label.
     */
    @Override
    public Label predict(Instance instance) {
        FeatureVector fv = instance.getFeatureVector();
        double hp = hypothesis(fv);
        return new ClassificationLabel(sigmoid(hp) >= 0.5? 1:0);
    }


    /**
     * Update parameters (Fij and W_j).
     * @param fv
     * @param wx
     * @param y
     */
    private void updateParameter(FeatureVector fv, double wx, int y){
        Iterator it = fv.iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();

            double delta = y * sigmoid(-wx) * (double)pair.getValue() +
                          (1 - y) * sigmoid(wx) * (-(double)pair.getValue());

            // Update Fij
            double preFtj = this.Fij.get(pair.getKey());
            this.Fij.put((int)pair.getKey(), Math.pow(delta,2) + preFtj);
            AdaGrad((int) pair.getKey());

            // Update W_j
            // W_j' = W_j + eta_ij * delta_i (feature i)
            double newW_j = this.parameters.get(pair.getKey()) + this.etas.get(pair.getKey()) * delta;
            this.parameters.put((int)pair.getKey(), newW_j);
        }
    }


    /**
     * Calculate hypothesis value of given feature vector.
     * @param fv
     * @return hypothesis value
     */
    private double hypothesis(FeatureVector fv){
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


    private double sigmoid(double z){
        return 1 / (1 + Math.exp(-z));
    }


    /**
     * Adaptively Choosing the Learning Rate
     * @param featIdx
     */
    private void AdaGrad(int featIdx){
        double newEta = etaZero / (Math.sqrt(IJ + Fij.get(featIdx)));
        this.etas.put(featIdx, newEta);
    }


}
