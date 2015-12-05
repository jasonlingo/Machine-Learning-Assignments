
package cs475.loopMRF;

import java.util.ArrayList;

public class LoopyBP {

	private LoopMRFPotentials potentials;
	private int iterations;
	// add whatever data structures needed
	private int n; //loopLength
	private int k; //numXValues
	private double[][][] muXtoF;
	private double[][][] muFtoX;

	public LoopyBP(LoopMRFPotentials p, int iterations) {
		this.potentials = p;
		this.iterations = iterations;

		this.n = this.potentials.loopLength();
		this.k = this.potentials.numXValues();

		this.muXtoF = new double[n + 1][n * 2 + 1][k + 1];  // index starts from 1
		this.muFtoX = new double[n * 2 + 1][n + 1][k + 1];  // index starts from 1
		this.initialize();
		this.computeMessage();
	}

	/**
	 * Initialize muXtoF array
	 */
	private void initialize(){
		for (int i = 1; i <= k; i++) {
			muXtoF[1][n + 1][i] = 1.0;
			muXtoF[1][2 * n][i] = 1.0;
		}
	}

	/**
	 * compute the marginal probability of x_i (k-ary array)
	 * The marginal probability of x_i is the product of all the
	 * message coming into x_i.
	 * @param x_i
	 * @return
     */
	public double[] marginalProbability(int x_i) {
		// TODO
		double[] probX = new double[k + 1];

		for (int x = 1; x <= k; x++){
			probX[x] = muFtoX[n + x_i][x_i][x] * muFtoX[n + 1 + (x_i - 2 + n) % n][x_i][x] * this.potentials.potential(x_i, x);
		}

		// normalization
		double z = 0.0;
		for (int x = 1; x <= k; x++){
			z += probX[x];
		}

		for (int x = 1; x <= k; x++){
			probX[x] = probX[x] / z;
		}

		return probX;
	}

	/**
	 * Update message for the given iteration.
	 */
	private void computeMessage(){
		for (int t = 1; t <= this.iterations; t++) {
//			System.out.printf("----- %d-th iteration -----\n", t);
			for (int i = 1; i <= n; i++) {
				computeFtoX(n + i, 1 + i % n, false);
				computeXtoF(1 + i % n, n + 1 + i % n);
			}
			for (int i = n; i >= 1; i--) {
				computeFtoX(n + i, i, true);
				computeXtoF(i, n + 1 + (i - 2 + n) % n);
			}
		}
	}

	/**
	 * Update message mu_{fi -> xi}
	 * @param fi
	 * @param xi
	 * @param reverse
     */
	private void computeFtoX(int fi, int xi, Boolean reverse){
		ArrayList<Integer> ne = neighborF(fi, xi);

		for (int x = 1; x <= k; x++){ // target xi
			double result = 0.0;
			for (int xf = 1; xf <= k; xf++){ //
				if (!reverse)
					result += this.potentials.potential(fi, xf, x) * muXtoF[ne.get(0)][fi][xf];
				else
					result += this.potentials.potential(fi, x, xf) * muXtoF[ne.get(0)][fi][xf];
			}
			muFtoX[fi][xi][x] = result;
//			System.out.printf("F%d->X%d (%d) = %f\n", fi, xi, x, result);
		}
	}

	/**
	 * Update message mu_{xi -> fi}
	 * @param xi
	 * @param fi
     */
	private void computeXtoF(int xi, int fi){
		ArrayList<Integer> ne = neighborX(xi, fi);

		for (int x = 1; x <= k; x++){
			double result = 1.0;
			for (int f: ne){
				if (f == xi)
					result *= this.potentials.potential(xi, x);
				else
					result *= muFtoX[f][xi][x];
			}
			muXtoF[xi][fi][x] = result;
//			System.out.printf("X%d->F%d (%d) = %f\n", xi, fi, x, result);
		}
	}

	/**
	 * Find the neighbor of xi except the "except" factor node.
	 * @param xi
	 * @param except: exclude this factor node
     * @return: an ArrayList of neighbor indices (fi)
     */
	private ArrayList<Integer> neighborX(int xi, int except){
		ArrayList<Integer> neighbor = new ArrayList<>();
		neighbor.add(xi);

		if (xi + n != except)
			neighbor.add(xi + n);

		if (xi == 1){
			if (except != 2 * n)
				neighbor.add(2 * n);
		} else {
			if (except != n + xi - 1)
				neighbor.add(n + xi - 1);
		}

		return neighbor;
	}

	/**
	 * Find the neighbor of fi except the "except" variable.
	 * @param fi
	 * @param except: exclude this variable
     * @return an ArrayList of neighbor indices (xi).
     */
	private ArrayList<Integer> neighborF(int fi, int except){
		ArrayList<Integer> neighbor = new ArrayList<>();
		if (except != fi - n)
			neighbor.add(fi - n);

		if (fi == 2 * n) {
			if (except != 1)
				neighbor.add(1);
		} else {
			if (except != fi - n + 1)
				neighbor.add(fi - n + 1);
		}

		return neighbor;
	}

}

