    /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.core.neighboursearch;

import weka.core.Instances;

// Measures
import weka.core.DISCDM;
import weka.core.HVDM;
import weka.core.EuclideanDistance;
import weka.core.TextDis;
/**
 *
 * @author Qu-pc3
 */
public class LinearNNSearch2 extends LinearNNSearch {
    
    public LinearNNSearch2(String metric) {
    super();
    if("Ecluidean".equals(metric))
           m_DistanceFunction = new EuclideanDistance();
    
    else if ("HVDM".equals(metric))
           m_DistanceFunction = new HVDM();
    
    else if("DISCDM".equals(metric))
        m_DistanceFunction = new DISCDM();
    
 
  } //
    
     public LinearNNSearch2(Instances insts,String metric) {
       super(insts);
       // Measures
      if("Ecluidean".equals(metric))
           m_DistanceFunction = new EuclideanDistance();
      
    else if ("HVDM".equals(metric))
           m_DistanceFunction = new HVDM();
    
    else if("DISCDM".equals(metric))
        m_DistanceFunction = new DISCDM();
    
   
      // 
       m_DistanceFunction.setInstances(insts);
       
  }//
   
       public LinearNNSearch2(Instances insts,String metric, double[] max,double[] min,double [] pw,double []wordsPerClass,double [][] wordGivenClass) {
       super(insts);
       
    
        if("TextDis".equals(metric))
        m_DistanceFunction = new TextDis(max,min,pw,wordsPerClass,wordGivenClass);
        else
            
            System.out.println("error");
        
        
      // 
       m_DistanceFunction.setInstances(insts);
       
  }//
    
    
    
    
    
} // class
