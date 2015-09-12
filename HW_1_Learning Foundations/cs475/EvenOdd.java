package cs475;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Created by Jason on 9/11/15.
 */
public class EvenOdd extends Predictor{

    private int _predict;

    public EvenOdd(){
        this._predict = -1;
    }

    @Override
    public void train(List<Instance> instances){
        double evenSum = 0.0;
        double oddSum = 0.0;

        for (Instance inst: instances){
            Iterator it = inst.getFeatureVector().iterator();
            while (it.hasNext()){
                Map.Entry pair = (Map.Entry)it.next();
                if ((int)pair.getKey() % 2 == 0){
                    evenSum += (double)pair.getValue();
                } else {
                    oddSum += (double)pair.getValue();
                }
            }
        }
        this._predict = evenSum >= oddSum? 1:0;
    }

    @Override
    public Label predict(Instance instance){
        if (this._predict == -1){
            System.out.println("The even-odd predictor is not trained yet.");
            return null;
        }

        double evenSum = 0.0;
        double oddSum = 0.0;

        Iterator it = instance.getFeatureVector().iterator();
        while (it.hasNext()){
            Map.Entry pair = (Map.Entry)it.next();
            if ((int)pair.getKey() % 2 == 0){
                evenSum += (double)pair.getValue();
            } else {
                oddSum += (double)pair.getValue();
            }
        }
        return new ClassificationLabel(evenSum >= oddSum? 1:0);
    }

}
