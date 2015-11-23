package cs475.RBM2;


import java.util.Random;

public class RBMEnergy {
	private RBMParameters parameters;
	private int num_samples;
	
	// TODO: Add the required data structures and methods.
	private int mVisible; // number of visible nodes
	private int nHidden;  // number of hidden nodes
	private int[][] x;    // visible variables
	private int[][] h;    // hidden variables
	private Random rand;

	public RBMEnergy(RBMParameters parameters, int numSamples) {
		this.parameters = parameters;
		this.num_samples = numSamples;

		this.init();
		this.generateSamele();
	}

	/**
	 * Initialize variables.
	 */
	private void init(){
		this.mVisible = this.parameters.numVisibleNodes();
		this.nHidden = this.parameters.numHiddenNodes();
		this.x = new int[num_samples + 1][mVisible + 1]; // index for variables starts from 1
		this.h = new int[num_samples + 1][nHidden + 1];
		this.rand = new Random(0);

		// initialize xi = 1  if i is even
		for (int i = 2; i <= mVisible; i += 2){
			x[0][i] = 1;
		}
	}

	/**
	 * Generate samples (x, h). It will first generate h_1 based on x_0,
	 * and then generate x_1 based on h_1 and so on for num_samples iteration.
	 */
	private void generateSamele(){
		for (int t = 1; t <= num_samples; t++){
			// generate h
			for (int j = 1; j <= nHidden; j++){
				if (rand.nextDouble() < sigmoid(computeForHj(t, j)))
					h[t][j] = 1;
				else
					h[t][j] = 0;
			}

			// generate x
			for (int i = 1; i <= mVisible; i++){
				if (rand.nextDouble() < sigmoid(computeForXi(t, i)))
					x[t][i] = 1;
				else
					x[t][i] = 0;
			}
		}
	}

	/**
	 * Compute the value of x^TW + dj using x from sample t-1
	 * @param t: identify the t-th sample
	 * @param j: the index of h
     * @return: the value of x^TW + dj
     */
	private double computeForHj(int t, int j){
		double result = 0.0;
		for (int i = 1; i <= mVisible; i++){
			result += x[t-1][i] * parameters.weight(i, j);
		}
		result += parameters.hiddenBias(j);

		return result;
	}

	/**
	 * Compute the value of h^TW^T + bi using h from sample t
	 * @param t: identify the t-th sample
	 * @param i: the index of x
     * @return: the value of h^TW^T + bi
     */
	private double computeForXi(int t, int i){
		double result = 0.0;
		for (int j = 1; j <= nHidden; j++){
			result += h[t][j] * parameters.weight(i, j);
		}
		result += parameters.visibleBias(i);

		return result;
	}

	/**
	 * To compute the margin probability of hj, we will count the
	 * number of hj = 1 in the all samples divided by the total number
	 * of samples.
	 *             number of samples that hj = 1
	 * p(hj = 1) = -----------------------------
	 *                total number of samples
	 * @param j
	 * @return
     */
	public double computeMarginal(int j) {
		// TODO: Add code here
		int hjOne = 0;
		for (int t = 1; t <= num_samples; t++){
			if (h[t][j] == 1)
				hjOne++;
		}
		return hjOne / (double)num_samples;
	}

	private double sigmoid(double z){
		return 1.0 / (1 + Math.exp(-z));
	}
}
