package weka.core;


public interface UpdateableDistanceFunction extends DistanceFunction{
	
	/**Update the distance function with the information of the newly added instance*/
	public void add(Instance instance);
	
	/**Update the distance function with the information of the newly remove instance*/
	public void remove(Instance instance);
}
