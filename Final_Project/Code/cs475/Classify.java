package cs475;

import cs475.NeuralNetwork.DataReader;
import cs475.NeuralNetwork.FeatureMatrix;
import cs475.NeuralNetwork.NeuralNetwork;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

import java.io.*;
import java.util.LinkedList;
import java.util.List;

public class Classify {
	static public LinkedList<Option> options = new LinkedList<Option>();

	final static int DEFAULT_SGD_ITERATION = 20;
	public static int sgd_iterations;

	public static void main(String[] args) throws IOException {
		// Parse the command line.
		String[] manditory_args = { "mode"};
		createCommandLineOptions();
		CommandLineUtilities.initCommandLineParameters(args, Classify.options, manditory_args);
	
		String mode = CommandLineUtilities.getOptionValue("mode");
		String data = CommandLineUtilities.getOptionValue("data");
		String predictions_file = CommandLineUtilities.getOptionValue("predictions_file");
		String model_file = CommandLineUtilities.getOptionValue("model_file");
		String layerInfo = CommandLineUtilities.getOptionValue("layer_info");

		sgd_iterations = DEFAULT_SGD_ITERATION;
		if (CommandLineUtilities.hasArg("sgd_iterations"))
			sgd_iterations = CommandLineUtilities.getOptionValueAsInt("sgd_iterations");

		if (mode.equalsIgnoreCase("train")) {
			if (data == null || model_file == null || layerInfo == null) {
				System.out.println("Train requires the following arguments: data, model_file, layer_info");
				System.exit(0);
			}

			// Load the training data.
			DataReader data_reader = new DataReader();
		    FeatureMatrix fm = data_reader.readData(data, layerInfo);

			// Train the model.
			Predictor predictor = train(fm);
			saveObject(predictor, model_file);
			System.out.println("Training ended");
			
		} else if (mode.equalsIgnoreCase("test")) {
			if (data == null || predictions_file == null || model_file == null) {
				System.out.println("Train requires the following arguments: data, predictions_file, model_file");
				System.exit(0);
			}
			
			// Load the test data.
			DataReader data_reader = new DataReader();
			FeatureMatrix fm = data_reader.readData(data, null);

			// Load the model.
			Predictor predictor = (Predictor)loadObject(model_file);
//			evaluateAndSavePredictions(predictor, fm, predictions_file);
		} else {
			System.out.println("Requires mode argument.");
		}
	}

	/**
	 * Train a model according to the given algorithm on training data.
	 * @param fm
	 * @return a trained predictor.
	 */
	private static Predictor train(FeatureMatrix fm) {
		// TODO Evaluate the model
		Predictor predictor = new NeuralNetwork();
		predictor.train(fm);

		// Evaluate the trained model.
		AccuracyEvaluator acuEva = new AccuracyEvaluator();
//		double accuracy = acuEva.evaluate(fm, predictor);
//		System.out.printf("Accuracy of training is %f\n", accuracy);

		return predictor;
	}

	private static void evaluateAndSavePredictions(Predictor predictor,
			List<Instance> instances, String predictions_file) throws IOException {
		PredictionsWriter writer = new PredictionsWriter(predictions_file);
		// TODO Evaluate the model if labels are available. 

		// If the data is too much, may consider the marked method below that performs
		// prediction and evaluation at the same time.
		for (Instance instance : instances) {
			Label label = predictor.predict(instance);
			writer.writePrediction(label);
		}

		// Evaluate the testing result.
		AccuracyEvaluator acuEva = new AccuracyEvaluator();
		double accuracy = acuEva.evaluate(instances, predictor);
		System.out.printf("Accuracy of testing is %f\n", accuracy);
		
		writer.close();
		
	}

	public static void saveObject(Object object, String file_name) {
		try {
			ObjectOutputStream oos =
				new ObjectOutputStream(new BufferedOutputStream(
						new FileOutputStream(new File(file_name))));
			oos.writeObject(object);
			oos.close();
		}
		catch (IOException e) {
			System.err.println("Exception writing file " + file_name + ": " + e);
		}
	}

	/**
	 * Load a single object from a filename. 
	 * @param file_name
	 * @return
	 */
	public static Object loadObject(String file_name) {
		ObjectInputStream ois;
		try {
			ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(new File(file_name))));
			Object object = ois.readObject();
			ois.close();
			return object;
		} catch (IOException e) {
			System.err.println("Error loading: " + file_name);
		} catch (ClassNotFoundException e) {
			System.err.println("Error loading: " + file_name);
		}
		return null;
	}
	
	public static void registerOption(String option_name, String arg_name, boolean has_arg, String description) {
		OptionBuilder.withArgName(arg_name);
		OptionBuilder.hasArg(has_arg);
		OptionBuilder.withDescription(description);
		Option option = OptionBuilder.create(option_name);
		
		Classify.options.add(option);		
	}
	
	private static void createCommandLineOptions() {
		registerOption("data", "String", true, "The data to use.");
		registerOption("mode", "String", true, "Operating mode: train or test.");
		registerOption("predictions_file", "String", true, "The predictions file to create.");
		registerOption("model_file", "String", true, "The name of the model file to create/load.");
		registerOption("sgd_iterations", "int", true, "The number of SGD iterations.");
		registerOption("layer_info", "String", true, "The name of layer information.");
//		registerOption("sgd_eta0", "double", true, "The constant scalar for learning rate in AdaGrad.");
//		registerOption("pegasos_lambda", "double", true, "The regularization parameter for Pegasos.");

		// Other options will be added here.
	}
}
