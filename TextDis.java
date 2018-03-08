package weka.core;

/**
 *  Implementation of the Heterogeneous Value Difference Metric
 *  @author Omar Alejandro Mainegra Sarduy (omainegra@uclv.edu.cu)
 */

import java.util.*;
import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.neighboursearch.PerformanceStats;


public class TextDis implements UpdateableDistanceFunction, Serializable, Cloneable{

    /** Dataset */
    protected Instances mData = null;
    
    protected int mNumClasses = 0;
    
    protected int mNumAtt = 0;         
 /** max_min Freq **/
    protected double[] max_freq ;
    protected double[] min_freq ;
    double [] pc; //p(c)
    double [][] pwc;//p(c,wi)
    double [] pw; // p(w)
   
    
 
    
    
    /**
     *  Default Constructor
     *  If you use it, after you must call "setInstances" methods
     * @param max
     * @param min
     * @param pw2
     * @param wordsPerClass
     * @param wordGivenClass
     */
    
    public TextDis(double []max,double []min,double [] pw2 , double[]wordsPerClass, double [][]wordGivenClass){
          max_freq = max;
        min_freq = min;
        pc = wordsPerClass; // p(c).
        pw = pw2; // p(w).
        pwc = wordGivenClass; // p(w,c).
     // System.out.println("pc "+pc.length+"wordsPerClass "+wordsPerClass.length);
     // for(int c=0;c<pc.length;c++)
       //   System.out.println("pc "+pc[c]+"wordsPerClass "+wordsPerClass[c]);
          
    }
	
    public TextDis(Instances instances){
        mData = instances;
        mNumClasses = mData.numClasses();
        mNumAtt = mData.numAttributes();
        
    }
    
   
    public String globalInfo() {      
      return "HVDM implements VDM for nominals attributes and normalizes continuous by Standard Deviation";
    }       
    
    public void setInstances(Instances instances){
        mData = instances;
        mNumClasses = mData.numClasses();
        mNumAtt = mData.numAttributes();
       
    }
    
    public Instances getInstances(){
    	return mData;
    }                

    @Override
    public Enumeration listOptions() {
        return new Vector().elements();
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
    }
  
    @Override
    public String [] getOptions() {
        return new String[]{""};    	
    } 

    /** 
    *   Calculates the distance (or similarity) between two instances. 
    *   @param first the first instance (test).
    *   @param second the second instance (train).
    *
    *   @return the distance between the two given instances.
    */
    
    @Override
    public double distance(Instance first, Instance second) {
        return distance(first,second,Double.MAX_VALUE);
    }
    
    // change
    @Override
    public double distance(Instance x, Instance y, double cutOffValue)   {
     // x is test instance / y is train instance
        double distance = 0.0; 
        double term=0.0,term2;
       /* int num1=0,num2=0;
        
        if(x.numValues() >= y.numValues())
           {num1 = x.numValues();}
        else
           num1 = y.numValues();
        
        if(x.numValues() <= y.numValues())
           {num2 = x.numValues();}
        else
           num2 = y.numValues();
        */
       // System.out.println(x.numValues());         System.out.println(y.numValues());         System.out.println(num);

               
       //  System.out.println("max "+max_freq+" min "+min_freq);
        for (int ax =0 ; ax <x.numValues() ; ax++){
          
            if (x.index(ax) != mData.classIndex()){
                
                if(!x.isMissing(ax)){
                    int ay=search(y,x.index(ax)); // search for attribute postion in train instance y whose index equals to test attribute index, if not found return -1
                    if(ay != -1){
                       // System.out.println( max_freq[x.index(ax)]+"  "+min_freq[x.index(ax)]);
                term2 = ( (y.valueSparse(ay)) - (x.valueSparse(ax) ) ) /(max_freq[x.index(ax)] - min_freq[x.index(ax)]);
                if(Double.isNaN(term2))
                    term2 = 0;
                if(Double.isInfinite(term2))
                    term2=1;
                try {
                    term = term2 * mutualInf(x.index(ax)); // change

                } catch (Exception ex) {
                    Logger.getLogger(TextDis.class.getName()).log(Level.SEVERE, null, ex);
                }
                  //System.out.println("max "+max_freq[ax] +" min "+ min_freq[ax]);
                  //System.out.println("y fre "+y.valueSparse(ay)+ " x fre "+x.valueSparse(ax));
                  //System.out.println( "term2 "+ term2);
                  //System.out.println("mutual inf "+ mutualInf(x.index(ax)));
                  //System.out.println("term "+term);
                    }
                    else
                        term=1; // if attribute not found in y instance
          //  else if(x.isMissing(ax) || y.isMissing(ay)){
          //      System.out.println("missinig");term =1;}
                
                distance += Math.pow(term, 2);
                  // if equal // if missing
                
                }}
            } 
               
        // error----------------------------------------------------------------
                if(Double.isNaN(distance)){//
                  try {
                     // System.out.println(mutualInf( mData,att));
                } catch (Exception ex) {
                    Logger.getLogger(TextDis.class.getName()).log(Level.SEVERE, null, ex);
                }
                System.out.println("NULL");}
        
        //------------------------------------------------------------------------
             
            
      
 		
           if (distance > cutOffValue)
            distance = Double.MAX_VALUE; 
           
        return Math.sqrt(distance);
    } //
    
    
    // search for attribute postion in train instance y whose index equals to test attribute index, if not found return -1.
    public int search(Instance inst,int index){ // correct
        int result = -1;
        for(int i=0;i<inst.numValues();i++){
           if(inst.index(i) == index)
               result = i;// position
    }
     return result;   
    }
        
 
    @Override
    public void postProcessDistances(double distances[]){
    }
    
    @Override
    public void update(Instance ins){
    }

  
    @Override
    public void setAttributeIndices(String value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String getAttributeIndices() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setInvertSelection(boolean value) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean getInvertSelection() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

   public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) {
        return (distance(first, second, Double.MAX_VALUE));
    }
    
     
    public double distance(Instance first, Instance second, PerformanceStats stats) {
        return (distance(first, second, Double.MAX_VALUE)); //To change body of generated methods, choose Tools | Templates.
    }
    
    
    // mutual information of attribute a
    public double mutualInf( int a){
        
        double mInfo=0;
     //  for(int a=0;a<test.numValues();a++) {
       for(int c=0;c<mNumClasses;c++){
         double term = pwc[c][a] * (Math.log(pwc[c][a]/(pw[a]*pc[c]))/Math.log(2))  ; //log2
         mInfo += term; 
               }  
    //   }
     return mInfo;   
    }
    
    
    @Override
    public void add(Instance instance) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void remove(Instance instance) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}