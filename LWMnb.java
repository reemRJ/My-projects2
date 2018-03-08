/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.bayes;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch2;
import weka.core.neighboursearch.LinearNNSearch;

import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 *
 * @author Qu-pc3
 */
public class LWMnb extends NaiveBayesMultinomial {
 

    public double [] Max_freq; // maximum frequency for each attribute
    public double [] Min_freq; // minimum frequency for each attribute
    double [] pc;    //p(c)
    double [][] pwc;  //p(c,wi)
    double [] pw;    // p(w)
    
    // p(c): the number of times all words appeared in the documents belonging to class c divided by the total number of all words occurrences
   //  p(wi): the number of times the word wi appears in all the documents divided by the total number of all words occurrences
  //   p(c,wi): the number of times the word wi appeared in the documents belonging to c
              //divided by the total number of all words occurrences
 
    
    
  /** 3
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
       // result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);


    // class
    result.enable(Capabilities.Capability.NOMINAL_CLASS);
    result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
    
    return result;
  }
  
  
  

  // set max and min frequency count for each attribute and mutual information probabilities
  // p(c): the number of times all words appeared in the documents belonging to class c divided by the total number of all words occurrences
  //p(wi): the number of times the word wi appears in all the documents divided by the total number of all words occurrences
 //p(c,wi): the number of times the word wi appeared in the documents belonging to c
          //divided by the total number of all words occurrences

  public void setMaxMinFreq(Instances train) throws Exception{
      
        double numOccurences,totalnumOccurences=0.0;
        double freq;
        Instance instance;
        int classIdx;
        
        //------Max and Min frequency--------------------------------------
         Max_freq = new double[train.numAttributes()];
         Min_freq = new double[train.numAttributes()];

        //----------------------------------------------
       // System.out.println("train"+train.numInstances());
        //System.out.println("attributes"+train.numAttributes());
        pc = new double [train.numClasses()];
        pw = new double[train.numAttributes()];
        pwc = new double [train.numClasses()] [train.numAttributes()];
       // System.out.println(pc.length +"  "+ pw.length );
        //------------------------------------------------------------------
        
        //..........................................
        // initialize P(w,c)/p(c)/pw(wi)/Max_freq[att_index]/Min_freq[att_index]
        //------------------------------------------
        for(int c = 0; c<train.numClasses(); c++){
            
	  for(int att = 0 ; att<train.numAttributes() ; att++){
	  
	       pwc[c][att] = 1; 
               pw[att] = 1; //System.out.println(pwc[c][att] +"  "+ pw[att]);
              Max_freq[att] = 0; Min_freq[att]= 0;
              
	  } // for att
          
               pc[c] = 1;
        
      } // for c
        
        
      //---------------------------------------------------------------------  
        
        
      //PART 2: loop through train data
    java.util.Enumeration<Instance> enumInsts =  train.enumerateInstances();
   
    while (enumInsts.hasMoreElements()) {
        
     
       instance = (Instance) enumInsts.nextElement();
       classIdx = (int) instance.value(instance.classIndex());// class
     //  System.out.println(instance);
            // System.out.println(instance.index(0) +" "+instance.valueSparse(0));
          //System.out.println(instance.classIndex());
      for (int a = 0; a < instance.numValues(); a++) {
          
        if (instance.index(a) != instance.classIndex()) {
          if (!instance.isMissing(a)) {
              numOccurences = instance.valueSparse(a);  
              
              if(numOccurences < 0)
		    throw new Exception("Numeric attribute values must all be greater or equal to zero.");
              
            freq = instance.valueSparse(a);//System.out.println(instance.index(a)+"  "+freq);
            if(Max_freq[instance.index(a)]== 0)
                Max_freq[instance.index(a)] = freq;
            if(Min_freq[instance.index(a)]== 0)
                Min_freq[instance.index(a)] = freq;
            
            if(freq>=Max_freq[instance.index(a)])
                Max_freq[instance.index(a)] = freq;
            if(freq<=Min_freq[instance.index(a)])
                Min_freq[instance.index(a)] = freq;
            
            //-----------------------------------------------------------
            pw[instance.index(a)] += numOccurences; // p(w)
            pc[classIdx] += numOccurences; // p(c)
        //   totalnumOccurences = train.numInstances();
           totalnumOccurences +=  numOccurences;//2
           pwc[classIdx][instance.index(a)] +=numOccurences; // p(w|c)
           //------------------------------------------------------------------------ 
              } // if is missing
        } // if class
    //    System.out.println("pw "+pw[instance.index(a)]+" pc "+pc[classIdx]+" pwc "+pwc[classIdx][instance.index(a)]+" total "+totalnumOccurences);
                        
      } //for a 
   
    }// while
   
   // totalnumOccurences=train.numInstances();
   for(int c=0;c<train.numClasses();c++){
        for(int a=0;a<train.numAttributes();a++){
            pwc[c][a]= pwc[c][a]/totalnumOccurences; //System.out.println(pwc[c][a]);
        //    pw[a]=pw[a]/totalnumOccurences; 
           
        //  System.out.println(pw[a]);
        }
         pc[c] = pc[c]/totalnumOccurences;
         //System.out.println(pc[c]);
               }
  
   for(int i=0;i<train.numAttributes();i++) {
       pw[i]=pw[i]/totalnumOccurences; 
        //System.out.println(Min_freq[i]+"   "+ Max_freq[i]);
   }
   
  }////
  
  
  /** 4
   * Generates the classifier.
   *
     * @param train
   * @param instances set of instances serving as -training data- 
     * @param test  test instance
     * @param k  the number of neighbours
   * @throws Exception if the classifier has not been generated successfully
   */
  public void buildClassifier(Instances train,Instance test,int k) throws Exception 
  {
      //-----------------------------------------------------------------------
    double [] distances = new double [k];
    // weighting
    double []w = new double[k];
    Instance instance;
    int classIndex;
  
    double numOccurences;
    double[] docsPerClass;
    double[] wordsPerClass;
    double sumWeight = 0.0; // correct   
   // Step 1: find similar instances (instances) to test
 
    NearestNeighbourSearch m_NNSearch = new LinearNNSearch2(train,"TextDis",Max_freq,Min_freq,pw,pc,pwc);
     //   NearestNeighbourSearch m_NNSearch = new LinearNNSearch2(train,"DISCDM");
     //NearestNeighbourSearch m_NNSearch = new LinearNNSearch(train);
    Instances instances = m_NNSearch.kNearestNeighbours(test, k);
     double [] d = m_NNSearch.getDistances(); // distances
    if (instances.numInstances() > k ){
         instances = new Instances(instances,0,k);System.arraycopy(d, 0, distances, 0, k);}
    else
    {instances = new Instances(instances,0,instances.numInstances());System.arraycopy(d, 0, distances, 0, d.length);}  
//  System.out.println(distances.length);System.out.println(instances.numInstances());
    
    
   // weighting Step;
    for(int i=0;i<k;i++){
       w[i] =1 / (Math.pow(distances[i],2)+1) ;  sumWeight += w[i];
  //     System.out.println(w[i]);
    //  System.out.println(instances.get(i)+" dis"+distances[i]);
         }
  // Utils.normalize(w);
    
  //  System.out.println(instances.instance(0));
//    System.out.println( instances.instance(0).index(0)+"  "+instances.instance(0).weight());  
      //-----------------------------------------------------------------------
    // can classifier handle the data?
    getCapabilities().testWithFail(instances);

    // remove instances with missing class c
    instances = new Instances(instances);
    instances.deleteWithMissingClass();
    
    m_headerInfo = new Instances(instances, 0);
    m_numClasses = instances.numClasses(); // # of classes in train data
    m_numAttributes = instances.numAttributes(); // |V| # of distinct words.
    m_probOfWordGivenClass = new double[m_numClasses][]; // p(wi | c)
    docsPerClass = new double[m_numClasses];
    wordsPerClass = new double[m_numClasses];
    /*
      initialising the matrix of word counts ( to one )
      NOTE: Laplace estimator introduced in case a word that does not appear for a class in the 
      training set does so for the test set
    */
    for(int c = 0; c<m_numClasses; c++)
      {
	m_probOfWordGivenClass[c] = new double[m_numAttributes];
	for(int att = 0; att<m_numAttributes; att++)
	  {
	    m_probOfWordGivenClass[c][att] = 1;
	  }
       
      }
	
    //enumerate through the instances 
 
    
  int inst =0;
   java.util.Enumeration enumInsts = instances.enumerateInstances();
    while (enumInsts.hasMoreElements()) 
      {
	instance = (Instance) enumInsts.nextElement();
	classIndex = (int)instance.value(instance.classIndex());
        instance.setWeight(w[inst]);//
	docsPerClass[classIndex] += instance.weight(); // p(c)
		
	for(int a = 0; a<instance.numValues(); a++)
	  if(instance.index(a) != instance.classIndex())
	    {
	      if(!instance.isMissing(a))
		{
		  numOccurences = instance.valueSparse(a) * instance.weight();
		  if(numOccurences < 0)
		    throw new Exception("Numeric attribute values must all be greater or equal to zero.");
		  wordsPerClass[classIndex] += numOccurences;
		  m_probOfWordGivenClass[classIndex][instance.index(a)] += numOccurences;
		}
	    }
        inst++;
      }
	//System.out.println(inst);
    /* part 2
      normalising probOfWordGivenClass values
      and saving each value as the log of each value p(wi | ci) = [count(w,c) + 1] / count(c)+|v|
    */
    for(int c = 0; c<m_numClasses; c++)
      for(int v = 0; v<m_numAttributes; v++) 
	m_probOfWordGivenClass[c][v] = Math.log(m_probOfWordGivenClass[c][v] / (wordsPerClass[c] + m_numAttributes - 1));
	
    /* part 3
      calculating Pr(H)
      NOTE: Laplace estimator introduced in case a class does not get mentioned in the set of 
      training instances p(c) = (nc +1) / (n + C) -- nc " # of docs having class c." -- n " # of docs" -- C "# of classes"
    */
    final double numDocs = instances.sumOfWeights() + m_numClasses;
    m_probOfClass = new double[m_numClasses];
    for(int h=0; h<m_numClasses; h++)
      m_probOfClass[h] = (double)(docsPerClass[h] + 1)/numDocs; 
  }
   //-----------------------------------------------------------
  
  
  
  
  /** 5
   * Calculates the class membership probabilities for the given test 
   * instance.
   *
   * @param instance the instance to be classified ** Test instance **
   * @return predicted class probability distribution
   * @throws Exception if there is a problem generating the prediction
   */
  public double [] distributionForInstance(Instance instance) throws Exception 
  {
    double[] probOfClassGivenDoc = new double[m_numClasses];
	
    //calculate the array of log(Pr[D|C])
    double[] logDocGivenClass = new double[m_numClasses];
    for(int h = 0; h<m_numClasses; h++)
      logDocGivenClass[h] = probOfDocGivenClass(instance, h);
	
    double max = logDocGivenClass[Utils.maxIndex(logDocGivenClass)];
    double probOfDoc = 0.0;
	
    for(int i = 0; i<m_numClasses; i++) 
      {
	probOfClassGivenDoc[i] = Math.exp(logDocGivenClass[i] - max) * m_probOfClass[i];
	probOfDoc += probOfClassGivenDoc[i];
      }
	
    //Utils.normalize(probOfClassGivenDoc,probOfDoc);
	
    return probOfClassGivenDoc;
  }
    
  
  
  /** 6
   * log(N!) + (for all the words)(log(Pi^ni) - log(ni!))
   *  
   *  where 
   *      N is the total number of words
   *      Pi is the probability of obtaining word i
   *      ni is the number of times the word at index i occurs in the document
   *
   * @param inst       The instance to be classified
   * @param classIndex The index of the class we are calculating the probability with respect to
   *
   * @return The log of the probability of the document occuring given the class
   */
    
  private double probOfDocGivenClass(Instance inst, int classIndex)
  {
    double answer = 0;
    //double totalWords = 0; //no need as we are not calculating the factorial at all.
	
    double freqOfWordInDoc;  //should be double
    for(int i = 0; i<inst.numValues(); i++) // sum (i=0, number of distinct words in test inst
      if(inst.index(i) != inst.classIndex())
	{
	  freqOfWordInDoc = inst.valueSparse(i) ; // 
	  //totalWords += freqOfWordInDoc;
	  answer += (freqOfWordInDoc * m_probOfWordGivenClass[classIndex][inst.index(i)] // fi * log p(wi|c)
		     ); //- lnFactorial(freqOfWordInDoc));
	}
	
    //answer += lnFactorial(totalWords);//The factorial terms don't make 
    //any difference to the classifier's
    //accuracy, so not needed.
	
    return answer;
  }
    
 
    
 
  
    
    
}//
