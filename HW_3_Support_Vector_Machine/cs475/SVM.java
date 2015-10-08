package cs475;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * A Support Vector Machine class.
 * Created by Li-Yi Lin on 10/7/15.
 */
public class SVM extends Predictor{


    // Parameters
    private double[] parameters;
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
        this.parameters = new double[featureNum + 1]; // Feature index starts from 1


        // ===================================================================
        // Start training.
        // ===================================================================
        for (int iter = 1; iter <= this.iteration; iter++){
            
        }


    }



    @Override
    public Label predict(Instance instance) {
        return null;
    }

}
