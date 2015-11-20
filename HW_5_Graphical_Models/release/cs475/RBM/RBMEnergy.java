package cs475.RBM;

import java.util.HashMap;
import java.util.Map;

public class RBMEnergy {
	private RBMParameters _parameters;
	private int _iters;
	private double _eta;

	// TODO: Add the required data structures and methods.
	private int M; // number of visible nodes x
	private int N; // number of hidden nodes h
	private int T; // number of example
	private int[][] binaryCombination;
	private double[][] deWeights; // partial derivative weight
	private double[] deB; // partial derivative b
	private double[] deD; // partial derivative d
	private int[][] x; // combination of x
	private int[][] h; // combination of h
	private Map<Integer, Double> energy;
	private double[][] subEnergy; // for saving sumForSigmoid() result

	public RBMEnergy(RBMParameters parameters, int iters, double eta) {
		this._parameters = parameters;
		this._iters = iters;
		this._eta = eta;
	}

	public void learning() {
		// TODO: Add code here
//		System.out.println("Start learnign===========================");
		// initialize parameters
		initialize();

		// Start training
		for (int iterNum = 1; iterNum <= _iters; iterNum++){
//			System.out.printf("========== Iteration %2d ==========\n", iterNum);
			energy = new HashMap<>();
			subEnergy = new double[T][N];

			eStep();
			mStep();
//			computeLogLikelihood(z);
		}
	}


//	private void computeLogLikelihood(double z){
//		int totalCombination = (int)Math.pow(2, N);
//		int[][] tempCombination = new int[totalCombination][N];
//
//		int col = 0;
//		int row = 0;
//		for (int i=0;i<totalCombination;i++){
//			int mask = totalCombination >> 1;
//			while (mask > 0){
//				if ((mask & i) == 0){
//					tempCombination[col][row] = 0;
//				} else {
//					tempCombination[col][row] = 1;
//				}
//				mask = mask >> 1;
//				row = (row + 1) % N;
//			}
//			col += 1;
//		}
//
////		for(int c = 0; c< tempCombination.length; c++){
////			for (int j = 0 ; j < N; j++) {
////				System.out.printf("%d ", tempCombination[c][j]);
////			}
////			System.out.println();
////		}
//
//		double loglikelihood = 0.0;
//
//		for (int t = 0; t < T; t++){
//			double tmpEnergy = 0.0;
//			for (int c = 0; c < tempCombination.length; c++) {
//				double eTheta = 0.0; // E_theta(x, h)
//				// x^T W h
//				double[] xw = new double[N];
//				for (int j = 0; j < N; j++) {
//					for (int i = 0; i < M; i++) {
//						xw[j] += _parameters.getExample(t, i) * _parameters.getWeight(i, j);
//					}
//					eTheta -= xw[j] * tempCombination[c][j];
//				}
//
//				// b^T x
//				for (int i = 0; i < M; i++) {
//					eTheta -= _parameters.getVisibleBias(i) * _parameters.getExample(t, i);
//				}
//
//				// d^T h
//				for (int j = 0; j < N; j++) {
//					eTheta -= _parameters.getHiddenBias(j) * tempCombination[c][j];
//				}
//				tmpEnergy += Math.exp(-eTheta) / z;
//			}
//			loglikelihood += Math.log(tmpEnergy);
//		}
//		System.out.printf("log likelihood = %f\n\n", loglikelihood);
//	}

	/**
	 * Compute the expections of the variables using the current model parameters.
	 */
	private void eStep(){
		double z = computeZ(); // partition function for normalization

		// initialize the derivative parameters
		deWeights = new double[M][N];
		deB = new double[M];
		deD = new double[N];

		// derive weights
		deriveWeight(z);

		// derive b (visible variable vector)
		deriveVisibleB(z);

		// derive d (hidden variable vector)
		deriveHiddenD(z);

//		return z;
	}


	/**
	 * Use gradient based optimization to update the parameters.
	 */
	private void mStep(){
		// update weights
		double w = 0.0;
		for (int i = 0; i < M; i++){
			for (int j = 0; j < N; j++){
				w = _parameters.getWeight(i, j) + _eta * deWeights[i][j];
				_parameters.setWeight(i, j, w);
			}
		}

		// update visible bias vector
		double b = 0.0;
		for (int i = 0; i < M; i++){
			b = _parameters.getVisibleBias(i) + _eta * deB[i];
			_parameters.setVisibleBias(i, b);
		}

		// update hidden bias vector
		double d = 0.0;
		for (int j = 0; j < N; j++){
			d = _parameters.getHiddenBias(j) + _eta * deD[j];
			_parameters.setHiddenBias(j, d);
		}
	}


	/**
	 * Derive weights
	 * @param z: partition function for normalization
     */
	private void deriveWeight(double z){
		for (int i = 0; i < M; i++){
			for (int j = 0; j < N; j++){
				// data side
				for (int t = 0; t < T; t++){
					deWeights[i][j] += -sigmoid(sumForSigmoid(t, j)) * (-_parameters.getExample(t, i));
				}
				// model side
				for (int c = 0; c < binaryCombination.length; c++){
					if (x[c][i] == 1 && h[c][j] == 1){
						deWeights[i][j] += -T * Math.exp(-energy.get(c)) / z;
					}
				}
			}
		}
	}


	/**
	 * Derive visible bias
	 * @param z: partition function for normalization
     */
	private void deriveVisibleB(double z){
		for (int i = 0; i < M; i++){
			for (int t = 0; t < T; t++){
				deB[i] += _parameters.getExample(t, i);
			}
			for (int c = 0; c < binaryCombination.length; c++){
				if (x[c][i] == 1){
					deB[i] += -T * Math.exp(-energy.get(c)) / z;
				}
			}
		}
	}


	/**
	 * Derive hidden bias
	 * @param z: partition function for normalization
     */
	private void deriveHiddenD(double z){
		for (int j = 0; j < N; j++){
			for (int t = 0; t < T; t++){
				deD[j] += sigmoid(sumForSigmoid(t, j));
			}
			for (int c = 0; c < binaryCombination.length; c++){
				if (h[c][j] == 1){
					deD[j] += -T * Math.exp(-energy.get(c)) / z;
				}
			}
		}
	}


	/**
	 * calculate x'w_,j + dj
	 * @param t: t-th example
	 * @param j: j-th value in a column of the weight parameter
     * @return
     */
	private double sumForSigmoid(int t, int j){
		if (subEnergy[t][j] != 0){
			return subEnergy[t][j];
		}

		double xwd = 0.0; // x(t)' * W_,j + dj
		for (int i = 0; i < M; i++){
			xwd += _parameters.getExample(t, i) * _parameters.getWeight(i, j);
		}
		xwd += _parameters.getHiddenBias(j);

		// save for next query
		subEnergy[t][j] = xwd;

		return xwd;
	}


	/**
	 * Compute the partition function that normalizes the distribution.
	 */
	private double computeZ(){
		double z = 0.0;
		// for each combination, calculate its energy function value
		for (int c = 0; c < binaryCombination.length; c++){
			z += Math.exp(-energyFunc(c));
		}
//		System.out.printf("\nZ = %f\n", z);

		return z;
	}


	/**
	 * calculate the energy function value:
	 * E_theta(x, h) = -x'Wh -b'x -d'h
	 * @param c: the c-th combination
	 * @return
     */
	private double energyFunc(int c){
		double eTheta = 0.0; // E_theta(x, h)

		// x'Wh
		double[] xw = new double[N];
		for (int j = 0; j < N; j++){
			for (int i = 0; i < M; i++){
				xw[j] += x[c][i] * _parameters.getWeight(i, j);
			}
			eTheta -= xw[j] * h[c][j];
		}

		// b'x
		for (int i = 0; i < M; i++){
			eTheta -= _parameters.getVisibleBias(i) * x[c][i];
		}

		// d'h
		for (int j = 0; j < N; j++){
			eTheta -= _parameters.getHiddenBias(j) * h[c][j];
		}

		energy.put(c, eTheta);
		return eTheta;
	}


    private double sigmoid(double number){
		return 1.0 / (1.0 + Math.exp(-number));
	}


	/**
	 * Initialize parameters: weights and two visible bias vectors
	 */
	private void initialize(){
		// Get parameters' vector/matrix dimension
		M = _parameters.numVisibleNodes();
		N = _parameters.numHiddenNodes();
		T = _parameters.numExamples();

		// initialize weights
		for(int i = 0; i < M; i++){
			for(int j = 1; j < N; j += 2){
				_parameters.setWeight(i, j, 1.0);
			}
		}

		// initialize visible bias vector
		for(int i = 1; i < M; i += 2){
			_parameters.setVisibleBias(i, 1.0);
		}

		// initialize hidden bias vector
		for(int j = 1; j < N; j += 2){
			_parameters.setHiddenBias(j, 1.0);
		}

		// initialize all binary combination
		buildAllBinaryCombinations(M + N);
	}

	/**
	 * build all binary combinations for num digits.
	 * @param num
     */
	private void buildAllBinaryCombinations(int num){
		int totalCombination = (int)Math.pow(2, num);
		binaryCombination = new int[totalCombination][num];

		int col = 0;
		int row = 0;
		for (int i=0;i<totalCombination;i++){
			int mask = totalCombination >> 1;
			while (mask > 0){
				if ((mask & i) == 0){
					binaryCombination[col][row] = 0;
				} else {
					binaryCombination[col][row] = 1;
				}
				mask = mask >> 1;
				row = (row + 1) % num;
			}
			col += 1;
		}
		divideCombinations();
	}

	/**
	 * Divide the binary combinations into two corresponding
	 * binary combinations.
	 */
	private void divideCombinations(){
		x = new int[(int)Math.pow(2, M+N)][M];
		h = new int[(int)Math.pow(2, M+N)][N];

		for (int c = 0; c < Math.pow(2, M+N); c++){
			for (int i = 0; i < M; i++){
				x[c][i] = binaryCombination[c][i];
			}
			for (int j = 0; j < N; j++){
				h[c][j] = binaryCombination[c][M+j];
			}
		}
	}


	public void printParameters() {
		//NOTE: Do not modify this function
		for (int i=0; i<_parameters.numVisibleNodes(); i++)
			System.out.println("b_" + i + "=" + _parameters.getVisibleBias(i));
		for (int i=0; i<_parameters.numHiddenNodes(); i++)
			System.out.println("d_" + i + "=" + _parameters.getHiddenBias(i));
		for (int i=0; i<_parameters.numVisibleNodes(); i++)
			for (int j=0; j<_parameters.numHiddenNodes(); j++)
				System.out.println("W_" + i + "_" + j + "=" + _parameters.getWeight(i,j));
	}

}
