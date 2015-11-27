package cs475.NeuralNetwork;

import cs475.Instance;
import cs475.Label;
import cs475.Predictor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


/**
 * A neural network classifier for digit recognition.
 */
public class NeuralNetwork extends Predictor {

    // number of layers
    private FeatureMatrix fm; // a class that stores labels, features, and layer information.

    // parameters for layers
    private Map<Integer, double[][]> theta;
    private Map<Integer, double[][]> thetaGrad;

    // for initializing theta
    private final double EPISLON = 0.12;

    // lambda for regularization
    private final double LAMBDA = 1.0;

    private final int MAX_Iteration = 1;

    // for storing layer nodes before taking sigmoid
    private Map<Integer, double[][]> zs;

    // for storing layer nodes after taking sigmoid
    private Map<Integer, double[][]> as;

    // for backpropagation
    private Map<Integer, double[][]> delta;
    private Map<Integer, double[][]> ds;

    private int totalLayers;


    public NeuralNetwork(){
        this.theta = new HashMap<>();
        this.as = new HashMap<>();
        this.zs = new HashMap<>();
        this.delta = new HashMap<>();
        this.ds = new HashMap<>();
        this.thetaGrad = new HashMap<>();

    }


    /* ====================================================
        Training
     ======================================================*/

    public void train(FeatureMatrix fm){
        System.out.println("Start training Neural Network");
        this.fm = fm;
        this.init();

        int iter = 0;
        Boolean converge = false;
        while (!converge && iter < MAX_Iteration){
            iter++;

            feedforward();
            double cost = cost();
            System.out.printf("cost = %f\n", cost);
            backpropagation();

        }
        removeData(); // for saving data space
    }

    /**
     * Randomly initialize theta for each layer.
     * theta = rand(nxm) * 2 * EPISLON - EPISON
     * so that -EPISLON <= theta <= EPISLON
     */
    private void init(){
        this.totalLayers = this.fm.getLayerNum();
        this.as.put(1, this.fm.getAllInstance());

        Map<Integer, Integer> layerInfo = this.fm.getLayerInfo();
        for (int i = 1; i < this.totalLayers; i++){
            double[][] th_i = Matrix.random(layerInfo.get(i+1), layerInfo.get(i)+1);
            th_i = Matrix.multiply(th_i, 2 * this.EPISLON);
            th_i = Matrix.substract(th_i, this.EPISLON);
            this.theta.put(i, th_i);
        }
    }

    private void feedforward(){
        System.out.println("Feedforward");
        Map<Integer, Integer> layerInfo = this.fm.getLayerInfo();
        double[][] x = this.fm.getAllInstance();

        for (int i = 1; i < this.totalLayers; i++){
            System.out.printf("---- %d-th layer ----\n", i);
            double[][] th_i = this.theta.get(i);
            x = Matrix.multiply(x, Matrix.transpose(th_i)); // theta^T x
            this.zs.put(i+1, x);

            x = sigmoidMatrix(x);
            if (i + 1 < this.totalLayers)
                x = addAllBias(x);
//            else
//                x = hFunc(x); // FIXME: should it be 0/1 vector?
            this.as.put(i+1, x);
        }
    }

    private double cost(){
        System.out.print("Calulate the cost");
        double[][] y = this.fm.getAllLabels();                          // the label matrix (y)
        double[][] minusY = Matrix.multiply(y, -1.0);
        double[][] h = this.as.get(this.totalLayers);       // the output layer
        double[][] logH = logMatrix(h);
        double[][] allOnesH = Matrix.allOnes(h.length, h[0].length);
        double[][] allOnesY = Matrix.allOnes(y.length, y[0].length);

        //         (   A      -      B   )  =  C
        // cost = (-y.*log(h) - (1-y).*log(1-h)) / example size
        double[][] A = Matrix.dotMultiply(minusY, logH); //-y.*log(h)
        double[][] B = Matrix.dotMultiply(Matrix.subtract(allOnesY, y), logMatrix(Matrix.subtract(allOnesH, h))); //(1-y).*log(1-h)
        double[][] C = Matrix.multiply(Matrix.subtract(A, B), 1.0 / y.length); // (A - B) / example size
        double cost = Matrix.sum(C);

        System.out.println("\nCalculate the regularization");
        // regularization: sum over all theta exception those for biases.
        double reg = 0.0;
        for (double[][] th: this.theta.values()){
            reg += Matrix.sumExceptFirst(Matrix.powOfElement(th, 2));
        }
        reg = reg * this.LAMBDA / y.length;

        return cost + reg;
    }

    private void backpropagation(){
        System.out.println("Backpropagation");

        for (int i = this.totalLayers; i > 1; i--){
            if (i == this.totalLayers) { // for the output layer
                double[][] h = this.as.get(this.totalLayers);
                double[][] y = this.fm.getAllLabels();
                double[][] d = Matrix.subtract(h, y);
                this.delta.put(i, d);
                this.ds.put(i, d);
            } else { // for other layers
                double[][] preD = this.ds.get(i + 1);        //d_i+1
                double[][] th = this.theta.get(i);           //theta_i
                double[][] curZ = this.zs.get(i);            //z_i
                double[][] curD = Matrix.multiply(preD, th); //d_i
                curD = removeAllBias(curD);
                double[][] sigGrad = sigmoidGradient(curZ);
                curD = Matrix.dotMultiply(curD, sigGrad);
                double[][] curDelta = Matrix.multiply(Matrix.transpose(this.as.get(i)), preD);
                this.delta.put(i, curDelta);
                this.ds.put(i, curD);
            }
        }
        // compute theta gradient
        for(int i = 2; i < this.totalLayers; i++){
            double[][] grad = this.delta.get(i);
            this.thetaGrad.put(i, Matrix.multiply(grad, 1.0 / this.fm.examSize()));
        }
    }



    /**
     * Check the result of backpropagation is reasonable. Turn off this function
     * after the first checking is done.
     */
    private void gradientChecking(){

    }

    private void updateTheta(){

    }

    private void removeData(){
        this.delta = null;
        this.ds = null;
        this.as = null;
        this.zs = null;
        this.fm = null;
        this.thetaGrad = null;
    }


    /* ====================================================
        Prediction
     ======================================================*/

    public Label predict(Instance instance){
        return null;
    }


    /* ====================================================
        Helper functions
     ======================================================*/

    private static double sigmoid(double z){
        double a = 1.0 / (1.0 + Math.exp(-z));
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * Perform sigmoid on each element in the given matrix.
     * @param A
     * @return
     */
    private static double[][] sigmoidMatrix(double[][] A){
        double[][] B = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++){
            for (int j = 0; j < A[0].length; j++){
                B[i][j] = sigmoid(A[i][j]);
            }
        }
        return B;
    }

    /**
     * Calculate the sigmoid gradient descent of each element in the given matrix.
     * @param A: the given matrix
     * @return: a sigmoid gradient descent matrix.
     */
    private static double[][] sigmoidGradient(double[][] A){
        double[][] grad = new double[A.length][A[0].length];
        for(int i = 0; i < A.length; i++){
            for(int j = 0; j < A[0].length; j++){
                grad[i][j] = sigmoid(A[i][j]) * (1 - sigmoid(A[i][j]));
            }
        }
        return grad;
    }

    /**
     * Compute the log value of each element in the given matrix.
     * @param A: the given matrix
     * @return: a matrix with the same dimension as the given matrix but
     *          with each value performed with log function.
     */
    private static double[][] logMatrix(double[][] A){
        double[][] C = new double[A.length][A[0].length];
        for(int i = 0; i < A.length; i++){
            for(int j = 0; j < A[0].length; j++){
                C[i][j] = Math.log(A[i][j]);
            }
        }
        return C;
    }

    private static double[][] hFunc(double[][] y){
        for (int i = 0; i < y.length; i++){
            double max = y[i][0];
            int maxIdx = 0;
            for (int j = 0; j < y[i].length; j++){
                if (y[i][j] > max){
                    max = y[i][j];
                    y[i][j] = 1.0;
                    y[i][maxIdx] = 0.0;
                    maxIdx = j;
                } else {
                    y[i][j] = 0.0;
                }
            }
        }
        return y;
    }

    /**
     * Remove all the theta for bias from the given matrix.
     * @param A: the given matrix
     * @return: a theta matrix without bias.
     */
    private static double[][] removeAllBias(double[][] A){
        double[][] C = new double[A.length][A[0].length - 1];
        for (int i = 0; i < C.length; i++){
            C[i] = Arrays.copyOfRange(A[i], 1, A[i].length);
        }
        return C;
    }

    /**
     * Add a bias node in all instances.
     * @param orgMatrix: the original instances.
     * @return: a new matrix with an additional bias node in each instance.
     */
    private static double[][] addAllBias(double[][] orgMatrix){
        double[][] newMatrix = new double[orgMatrix.length][orgMatrix[0].length + 1];
        for (int i = 0; i < newMatrix.length; i++){
            for (int j = 0; j < newMatrix[0].length; j++){
                if (j == 0)
                    newMatrix[i][j] = 1.0;
                else
                    newMatrix[i][j] = orgMatrix[i][j - 1];
            }
        }
        return newMatrix;
    }
}
