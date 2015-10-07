package cs475;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Created by Jason on 9/11/15.
 */
public class EvenOdd extends Predictor{

    public EvenOdd(){
    }

    @Override
    public void train(List<Instance> instances){
        // The training data has nothing to do with the prediction.
        // So this training function will do nothing.

//        double evenSum = 0.0;
//        double oddSum = 0.0;
//
//        for (Instance inst: instances){
//            Iterator it = inst.getFeatureVector().iterator();
//            while (it.hasNext()){
//                Map.Entry pair = (Map.Entry)it.next();
//                if ((int)pair.getKey() % 2 == 0){
//                    evenSum += (double)pair.getValue();
//                } else {
//                    oddSum += (double)pair.getValue();
//                }
//            }
//        }
    }

    @Override
    public Label predict(Instance instance){
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
