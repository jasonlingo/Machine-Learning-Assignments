package cs475;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

public class BernoulliLikelihood {
	static public LinkedList<Option> options = new LinkedList<Option>();
	
	public static void main(String[] args) throws IOException {
		// Parse the command line.
		String[] manditory_args = { "data"};
		createCommandLineOptions();
		CommandLineUtilities.initCommandLineParameters(args, BernoulliLikelihood.options, manditory_args);
		
		String dataFile = CommandLineUtilities.getOptionValue("data");
		BernoulliLikelihood bl = new BernoulliLikelihood();
		ArrayList<Integer> data = bl.readData(dataFile);
		
		double parameter = bl.computeMaximumLikelihood(data);
		double llhood = bl.computeLogLikelihood(data, parameter);
		System.out.println("Maximum Likelihood Solution: " + Double.toString(parameter));
		System.out.println("Log-likelihood: " + Double.toString(llhood));

		
	}
	
	public static void registerOption(String option_name, String arg_name, boolean has_arg, String description) {
		OptionBuilder.withArgName(arg_name);
		OptionBuilder.hasArg(has_arg);
		OptionBuilder.withDescription(description);
		Option option = OptionBuilder.create(option_name);
		
		BernoulliLikelihood.options.add(option);		
	}
	
	private static void createCommandLineOptions() {
		registerOption("data", "String", true, "The data file to read.");
	}
	
	public double computeMaximumLikelihood(ArrayList<Integer> data) {
		// TODO: Fill in here
		List<Integer> count = countOnes(data);
		if (data.size() - count.get(1) <= 0){
			System.out.printf("Divisor is %d. Please check the input data.\n", (data.size() - count.get(1)));
			return 0.0;
		}
		return (double)count.get(0) / (double)(data.size() - count.get(1));
	}
	
	public double computeLogLikelihood(ArrayList<Integer> data, double parameter) {
		// TODO: Fill in here
		// log(0) is undefined.
		if (parameter == 0.0 || parameter == 1.0){
			System.out.printf("The parameter of computerLogLikelihood is %f. Log(0) is not defined.\n.", parameter);
			return 0.0;
		}

		List<Integer> count = countOnes(data);
		if (data.size() - count.get(1) <= 0){
			System.out.println("All input are illegal. Please check the input data.");
			return 0.0;
		}

		return count.get(0) * Math.log(parameter) +
				(data.size() - count.get(1) - count.get(0)) * Math.log(1.0 - parameter);
	}

	/**
	 * Count the number of ones happens in the data set and also the number of
	 * illegal input (neither 1 nor 0).
	 * @param data
	 * @return a list of counts.
	 */
	public List<Integer> countOnes(ArrayList<Integer> data){
		ArrayList<Integer> count = new ArrayList<>();
		int totOnes = 0;
		int discard = 0;
		for (int d: data){
			if (d == 1){
				totOnes++;
			} else if (d != 0){
				System.out.println("data is not 0 and 1.");
				discard++;
			}
		}
		count.add(totOnes); //count[0] stores the total number of ones
		count.add(discard); //count[1] stores the total number of discarded data (instance)
		                    //illegal labelling (not in set {0,1}).

		return count;
	}
	
	public ArrayList<Integer> readData(String filename) throws FileNotFoundException {
		Scanner scanner = new Scanner(new BufferedInputStream(new FileInputStream(filename)));
		ArrayList<Integer> data = new ArrayList<Integer>();
		
		while (scanner.hasNextLine()) {
			String line = scanner.nextLine();
			if (line.trim().length() == 0)
				   continue;
			String result = line.trim();
			int value = Integer.parseInt(result);
			data.add(value);
		}
		
		scanner.close();
		return data;
	}
}
