
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

	public double[] marginalProbability(int x_i) {
		// TODO
		double[] probX = new double[k + 1];

		for (int x = 1; x <= k; x++){
//			System.out.println("Marginal prob");
//			System.out.println(muFtoX[n + x_i][x_i][x]);
//			System.out.println(muFtoX[n + 1 + (x_i - 2 + n) % n][x_i][x]);
//			System.out.println(this.potentials.potential(x_i, x));
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

	private void computeMessage(){
		for (int t = 1; t <= this.iterations; t++) {
			System.out.printf("\n===== %d-th iteration =====", t);

			for (int i = 1; i <= n; i++) {
				computeFtoX(n + i, 1 + i % n);
				computeXtoF(1 + i % n, n + 1 + i % n);
			}
			System.out.println("\n======== reverse ========");
			for (int i = n; i >= 1; i--) {
				computeFtoXR(n + i, i);
				computeXtoFR(i, n + 1 + (i - 2 + n) % n);
			}
		}
	}


	private void computeFtoX(int fi, int xi){
		System.out.printf("\n\nmu_{f%d -> x%d} += ", fi, xi);

		ArrayList<Integer> ne = neighborF(fi, xi);

		for (int x = 1; x <= k; x++){ // target xi
			double result = 0.0;
			for (int xf = 1; xf <= k; xf++){ //
//				if (fi == 2 * n)
					result += this.potentials.potential(fi, xf, x) * muXtoF[ne.get(0)][fi][xf];
//				else
//					result += this.potentials.potential(fi, x, xf) * muXtoF[ne.get(0)][fi][xf];
			}
			muFtoX[fi][xi][x] = result;
		}
	}

	private void computeFtoXR(int fi, int xi){
		System.out.printf("\n\nmu_{f%d -> x%d} += ", fi, xi);

		ArrayList<Integer> ne = neighborF(fi, xi);

		for (int x = 1; x <= k; x++){ // target xi
			double result = 0.0;
			for (int xf = 1; xf <= k; xf++){ //
//				if (fi == 2 * n)
					result += this.potentials.potential(fi, x, xf) * muXtoF[ne.get(0)][fi][xf];
//				else
//					result += this.potentials.potential(fi, x, xf) * muXtoF[ne.get(0)][fi][xf];
			}
			muFtoX[fi][xi][x] = result;
		}
	}

	private void computeXtoF(int xi, int fi){
		System.out.printf("\n\nmu_{x%d -> f%d} += ", xi, fi);

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
		}
	}

	private void computeXtoFR(int xi, int fi){
		System.out.printf("\n\nmu_{x%d -> f%d} += ", xi, fi);

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
		}
	}

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

