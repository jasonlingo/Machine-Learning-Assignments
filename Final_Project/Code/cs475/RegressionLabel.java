package cs475;

import java.io.Serializable;

public class RegressionLabel extends Label implements Serializable {

	private double _label;

	public RegressionLabel(double label) {
		this._label = label;
	}

	@Override
	public String toString() {
		return Double.toString(this._label);
	}

}
