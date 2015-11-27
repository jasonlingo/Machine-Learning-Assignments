package cs475;

/**
 * @author: Li-Yi Lin
 */
import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class FeatureVector implements Serializable {

	// Sparse feature vector.
	private Map<Integer, Double> _featureVector;

	/**
	 * Construct a feature vector.
	 */
	public FeatureVector(){
		this._featureVector = new HashMap<>();
	}

	public void add(int index, double value) {
		this._featureVector.put(index, value);
	}
	
	public double get(int index) {
		return this._featureVector.get(index);
	}

	public int size() {
		return this._featureVector.size();
	}

	public boolean containsKey(int index){
		return this._featureVector.containsKey(index);
	}


	/**
	 * Create an iterator for the feature vector.
	 * @return
	 */
	public Iterator iterator(){
		return _featureVector.entrySet().iterator();
	}
}
