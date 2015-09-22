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
		// TODO Auto-generated method stub
		this._featureVector.put(index, value);
	}
	
	public double get(int index) {
		// TODO Auto-generated method stub
		return this._featureVector.get(index);
	}

	public int size() {
		// TODO Auto-generated method stub
		return this._featureVector.size();
	}


	/**
	 * Create an iterator for the feature vector.
	 * @return
	 */
	public Iterator iterator(){
		return _featureVector.entrySet().iterator();
	}
}
