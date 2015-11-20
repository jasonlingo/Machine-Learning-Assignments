/**
 * Created by Li-Yi Lin on 9/11/15.
 */
package cs475;

import java.util.List;

public class Majority extends Predictor{

    // The majority of training data
    int _majority;

    /**
     * Construct a "majority" predictor
     */
    public Majority(){
        // Set _majority to -1 in order to prevent using predict before
        // the predictor is trained.
        this._majority = -1;
    }

    /**
     * Find the majority label value in training data. When two labels are
     * tied for occurring the most often then the majority classifier picks
     * label “1”
     * @param instances
     */
    @Override
    public void train(List<Instance> instances){
        int totZeroes = 0;
        int totOnes = 0;
        for (Instance inst: instances){
            if (inst.getLabel().toString().equals("1")){
                totOnes++;
            } else if(inst.getLabel().toString().equals("0")){
                totZeroes++;
            } else {
                System.out.println("Found a wrong label during training.");
            }
        }
        this._majority = totOnes >= totZeroes? 1:0;
    }

    @Override
    public Label predict(Instance instance){
        // Check _majority != -1 to make sure the predictor is trained.
        if (this._majority == -1){
            System.out.println("This majority predictor is not trained yet.");
            return null;
        }
        return new ClassificationLabel(this._majority);
    }

}
