package cs475.NeuralNetwork;

import java.io.Serializable;
import java.util.*;

/**
 * Created by Jason on 11/23/15.
 */
public class FeatureMatrix implements Serializable {

    private Map<Integer, double[]> labels;

    // store the input features (already has a bias in each instance)
    private Map<Integer, double[]> features;

    private int classNum;

    // store the number of node in each layer from the input layer to output layer.
    private Map<Integer, Integer> layerInfo;

    public FeatureMatrix(int classNum){
        this.labels = new HashMap<>();
        this.features = new HashMap<>();
        this.classNum = classNum;
    }

    public void addLabel(int idx, double[] label){
        this.labels.put(idx, label);
    }

    /**
     * Add a feature set into the features list.
     * @param idx: the index for this feature.
     * @param features: the feature array for an example.
     */
    public void addFeatures(int idx, double[] features){
        this.features.put(idx, features);
    }

    /**
     * Get the idx-th example.
     * @param idx: index
     * @return an double array that stores the idx-th example.
     */
    public double[] getFeatures(int idx){
        return this.features.get(idx);
    }

    /**
     * Store all the instances' features in a two-dimensional array.
     * @return: a two-dimensional array with first index representing the number of instances and
     * second index representing the number of features => inst[instance][feature]
     */
    public double[][] getAllInstance(){
        double[][] inst = new double[this.features.size()][this.features.get(1).length];
        int i = 0;
        for (double[] x: this.features.values()){
            inst[i] = x;
            i++;
        }
        return inst;
    }

    public double[][] getAllLabels(){
        double[][] allLabels = new double[this.labels.size()][this.labels.get(1).length];
        int idx = 0;
        for(double[] label: this.labels.values()){
            allLabels[idx] = label;
            idx++;
        }
        return allLabels;
    }

    /**
     * Get the idx-th label.
     * @param idx: index
     * @return: an array that store the label information of the idx-th example.
     */
    public double[] getLabels(int idx){
        return this.labels.get(idx);
    }

    /**
     * Set the number of hidden layers and hidden nodes.
     * @param layerInfo
     */
    public void setLayerInfo(Map<Integer, Integer> layerInfo){
        this.layerInfo = layerInfo;
    }

    public Map<Integer, Integer> getLayerInfo(){
        return this.layerInfo;
    }

    public int getLayerNum(){
        return this.layerInfo.size();
    }

    /**
     * @return: return the number of examples
     */
    public int examSize() {
        if (this.labels.size() != this.features.size()) {
            System.out.println("Numbers of labels and features are not the same!!!");
            System.exit(0);
        }
        return this.labels.size();
    }

    public boolean containsKey(int index){
        return true;
    }

    /**
     * Create an iterator for the feature vector.
     * @return
     */
    public Iterator iterator(){
//        return _featureVector.entrySet().iterator();
        return null;
    }
}
