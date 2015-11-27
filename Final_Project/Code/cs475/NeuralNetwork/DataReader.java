package cs475.NeuralNetwork;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.*;


public class DataReader {

//	private Scanner _scanner;
//	private String _hiddenLayerInfo; // hidden nodes in every hidden layers
	private Map<Integer, Integer> layerInfo;
	private final int CLASS_NUM = 10;

	public DataReader(){

	}

	public FeatureMatrix readData(String featFilename, String layerFilename) throws FileNotFoundException  {
		System.out.print("loading data");
		Scanner scanner = new Scanner(new BufferedInputStream(new FileInputStream(featFilename)));
		FeatureMatrix fm = new FeatureMatrix(this.CLASS_NUM);

		if (layerFilename != null)
			this.layerInfo = readLayerInfo(layerFilename);

			fm.setLayerInfo(this.layerInfo);

		int instanceNum = 0;
		while (scanner.hasNextLine()) {
			instanceNum++;
			if (instanceNum % 500 == 0)
				System.out.print(".");

			String line = scanner.nextLine();
			if (line.trim().length() == 0)
				   continue;

			// divide the line into features and label.
			String[] split_line = line.split("\t");

			// extract label
			int int_label = Integer.parseInt(split_line[0]);
			fm.addLabel(instanceNum, genLabel(int_label));

			// extract features
			double[] features = new double[split_line.length];
			features[0] = 1.0; // set first element as bias
			for (int i = 1; i < split_line.length; i++) {
				features[i] = Double.parseDouble(split_line[i]);
			}
			fm.addFeatures(instanceNum, features);
		}

		scanner.close();
		System.out.printf("\nTotal training data: %d\n", fm.examSize());
		return fm;
	}

	private Map<Integer, Integer> readLayerInfo(String filename) throws FileNotFoundException {
		Scanner scanner = new Scanner(new BufferedInputStream(new FileInputStream(filename)));

		Map<Integer, Integer> layerInfo = new HashMap<>();

		int layerIdx = 0;
		while (scanner.hasNextLine()) {
			layerIdx++;

			String line = scanner.nextLine();
			if (line.trim().length() == 0)
				continue;

			layerInfo.put(layerIdx, Integer.parseInt(line));
		}
		scanner.close();

		return layerInfo;
	}

	public Map<Integer, Integer> getLayerInfo(){
		return this.layerInfo;
	}

	/**
	 * Generate a label array with idx-th element set to 1.
	 * @param idx
	 * @return: an integer array with idx-th element set to 1.
     */
	private double[] genLabel(int idx){
		double[] label = new double[this.CLASS_NUM];
		label[idx] = 1.0;
		return label;
	}
}
