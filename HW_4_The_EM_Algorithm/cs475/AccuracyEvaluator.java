package cs475;

import java.util.List;

/**
 * A class that can calculate the accuracy of training and testing result.
 * Created by Li-Yi Lin on 9/12/15.
 */
public class AccuracyEvaluator extends Evaluator{

    /**
     * Constructor an AccuracyEvaluator.
     */
    public AccuracyEvaluator(){}


    /**
     * Calculate the accuracy by comparing the predicted labels and original
     * labels and ignore data without labels.
     * @param instances
     * @param predictor
     * @return
     */
    @Override
    public double evaluate(List<Instance> instances, Predictor predictor){
        int correct = 0;   // the total number of correct predictions.
        int totalInst = 0; // the total number of instances that already have labels.
        for (Instance inst: instances) {
            String predict = predictor.predict(inst).toString();
            if (inst.getLabel() != null) {
                totalInst++;
                if (inst.getLabel().toString().equals(predict)) {
                    correct++;
                }
            }
        }
        System.out.printf("the accuracy is %f (%d / %d)\n",
                           (double)correct / (double)totalInst, correct, totalInst);
        return (double)correct / (double)totalInst;
    }
}
