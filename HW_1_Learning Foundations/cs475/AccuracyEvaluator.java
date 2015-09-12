package cs475;

import java.util.List;

/**
 * Created by Li-Yi Lin on 9/12/15.
 */
public class AccuracyEvaluator extends Evaluator{


    /**
     * Constructor an AccuracyEvaluator.
     */
    public AccuracyEvaluator(){

    }

    @Override
    public double evaluate(List<Instance> instances, Predictor predictor){
        int match = 0;
        for (Instance inst: instances) {
            String pred = predictor.predict(inst).toString();
            if (inst.getLabel().toString().equals(pred)) {
                match++;
            }
        }
        return (double)match / (double)instances.size();
    }
}
