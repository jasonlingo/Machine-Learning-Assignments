package cs475;

/**
 * @author: Li-Yi Lin
 */
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class FeatureVector {

	// Feature vector.
	private Map<Integer, Double> featureVector;


	/**
	 * Construct a feature vector.
	 */
	public FeatureVector(){
		this.featureVector = new HashMap<>();
	}

	public void add(int index, double value) {
		// TODO Auto-generated method stub
		this.featureVector.put(index, value);
	}
	
	public double get(int index) {
		// TODO Auto-generated method stub
		return this.featureVector.get(index);
	}

	public int size() {
		// TODO Auto-generated method stub
		return this.featureVector.size();
	}

	/**
	 * Create an iterator for the feature vector.
	 * @return
	 */
	public Iterator iterator(){
		Iterator it = featureVector.entrySet().iterator();
		return it;
	}
}
