package cs475;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

public class Classify {
	static public LinkedList<Option> options = new LinkedList<Option>();

	final static int DEFAULT_SGD_ITERATION = 20;
	final static double DEFAULT_ETA_ZERO = 0.01;

	public static int sgdIterations;
	public static double etaZero;

	public static void main(String[] args) throws IOException {
		// Parse the command line.
		String[] manditory_args = { "mode"};
		createCommandLineOptions();
		CommandLineUtilities.initCommandLineParameters(args, Classify.options, manditory_args);
	
		String mode = CommandLineUtilities.getOptionValue("mode");
		String data = CommandLineUtilities.getOptionValue("data");
		String predictions_file = CommandLineUtilities.getOptionValue("predictions_file");
		String algorithm = CommandLineUtilities.getOptionValue("algorithm");
		String model_file = CommandLineUtilities.getOptionValue("model_file");
		String sgd_iterations = CommandLineUtilities.getOptionValue("sgd_iterations");
		String sgd_eta0 = CommandLineUtilities.getOptionValue("sgd_eta0");

		if (mode.equalsIgnoreCase("train")) {
			if (data == null || algorithm == null || model_file == null) {
				System.out.println("Train requires the following arguments: data, algorithm, model_file");
				System.exit(0);
			}
			// Load the training data.
			DataReader data_reader = new DataReader(data, true);
			List<Instance> instances = data_reader.readData();
			data_reader.close();
			
			// Train the model.
			// Parse parameters for Logistic regression
			// sgd_iterations
			if (sgd_iterations != null){
				sgdIterations = Integer.parseInt(sgd_iterations);
			} else {
				sgdIterations = DEFAULT_SGD_ITERATION;
			}
			// eta0
			if (sgd_eta0 != null){
				etaZero = Double.parseDouble(sgd_eta0);
			} else {
				etaZero = DEFAULT_ETA_ZERO;
			}
			Predictor predictor = train(instances, algorithm);
			saveObject(predictor, model_file);
			
		} else if (mode.equalsIgnoreCase("test")) {
			if (data == null || predictions_file == null || model_file == null) {
				System.out.println("Train requires the following arguments: data, predictions_file, model_file");
				System.exit(0);
			}
			
			// Load the test data.
			DataReader data_reader = new DataReader(data, true);
			List<Instance> instances = data_reader.readData();
			data_reader.close();
			
			// Load the model.
			Predictor predictor = (Predictor)loadObject(model_file);
			evaluateAndSavePredictions(predictor, instances, predictions_file);
		} else {
			System.out.println("Requires mode argument.");
		}
	}

	/**
	 * Train a model according to the given algorithm on training data.
	 * @param instances
	 * @param algorithm
	 * @return a trained predictor.
	 */
	private static Predictor train(List<Instance> instances, String algorithm) {
		// TODO Train the model using "algorithm" on "data"
		// TODO Evaluate the model
		Predictor predictor;
		switch (algorithm){
			case "majority":
				predictor = new Majority();
				break;
			case "even_odd":
				predictor = new EvenOdd();
				break;
			case "logistic_regression":
				// Set total number of iterations.
				predictor = new LogisticRegression(sgdIterations, etaZero);
				break;
			default:
				System.out.println("Please check the algorithm's name.");
				return null;
		}
		predictor.train(instances);

		// Evaluate the trained model.
		AccuracyEvaluator acuEva = new AccuracyEvaluator();
		double accuracy = acuEva.evaluate(instances, predictor);
		System.out.printf("Accuracy of training is %f\n", accuracy);

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
		registerOption("algorithm", "String", true, "The name of the algorithm for training.");
		registerOption("model_file", "String", true, "The name of the model file to create/load.");
		registerOption("sgd_iterations", "Integer", true, "The number of total iterations for regression.");
		registerOption("sgd_eta0", "Double", true, "The value of eta0.");

		// Other options will be added here.
	}
}
