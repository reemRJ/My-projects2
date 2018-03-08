
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Reader;


import weka.classifiers.bayes.LWMnb;


import weka.core.Instances;


public class TextClassification_eclu {
   
    public static String[] files ={"tr12.wc.arff","tr11.wc.arff","tr21.wc.arff","tr23.wc.arff","tr31.wc.arff","tr41.wc.arff","tr45.wc.arff","oh0.wc.arff","fbis.wc.arff","wap.wc.arff","la1s.wc.arff","la2s.wc.arff",
    "oh5.wc.arff","oh15.wc.arff","re0.wc.arff","oh10.wc.arff","re1.wc.arff"};
    
    public static String path = "C:\\Users\\Qu-pc3\\Documents\\NetBeansProjects\\M_UCI\\text\\";

     public static  int nFiles = files.length;
     
     
     
     
     
     public static void main(String[] args) throws Exception {
        double totalAccMetric1=0;
        double totalAccMetric2=0;
        double totalAccMetric3 = 0;
        double totalAccMetric4 = 0; 
        
        double noOfBetterFolds1bet3=0;
        double noOfBetterFolds3bet1=0;
        double noOfBetterFolds3bet2=0;
        double noOfBetterFolds2bet3=0;
        
        double noOfBetterFolds4bet3=0;
        double noOfBetterFolds3bet4=0;
         

        boolean discrtized=true;
        String fileName;  
        Accuracy [] accuracy = new Accuracy[nFiles];  
        System.out.println(nFiles);
        
        
        
         for (int f = 0; f < files.length ; f++) {
             fileName = path + files[f];
             Instances insts;
             Reader r = null;
             
             
             try {
                String[] argv = {"-t", fileName};
                r = new BufferedReader(new FileReader(fileName));
                insts = new Instances(r);
                int att = insts.numAttributes() - 1;
                insts.setClass(insts.attribute(att));
                 insts.deleteWithMissingClass();
     
*/
            argv = new String[2]; //{"-t"," ",fileName};
            argv[0] = "-t"; //+fileName;
            argv[1] = fileName; //"c:\\UCIdataSets\\uci-arff\\nominal\\weightedDataSetskNN\\rwkNNglass.arff";
            //end Khalil
            
            //========================
            // evaluation
            //=========================
             accuracy[f] = crossValidateCompare(f,argv,insts); 

                    totalAccMetric1 += accuracy[f].accMetric1;
                    totalAccMetric2 += accuracy[f].accMetric2;
                    totalAccMetric3 += accuracy[f].accMetric3;
                    totalAccMetric4 += accuracy[f].accMetric4;
                    
                    noOfBetterFolds1bet3 +=accuracy[f].noBetterFoldsMethod1bet3;
                    noOfBetterFolds3bet1 +=accuracy[f].noBetterFoldsMethod3bet1;
                    noOfBetterFolds3bet2 +=accuracy[f].noBetterFoldsMethod3bet2;
                    noOfBetterFolds2bet3 +=accuracy[f].noBetterFoldsMethod2bet3;
                    
                    noOfBetterFolds3bet4 +=accuracy[f].noBetterFoldsMethod3bet4;
                    noOfBetterFolds4bet3 +=accuracy[f].noBetterFoldsMethod4bet3;



           
            
             } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
  // end of discritization
         }// files
         
         
         System.out.println(" the average accuracy of LWMNB is " + totalAccMetric1 / (nFiles));
        
    }
             
       public static Accuracy crossValidateCompare(int trial, String [] argv, Instances insts) throws Exception, Exception{
           
        Accuracy acc = new Accuracy();
        
         int numIterations = 0;
        double sumNoIterations = 0.0; 
        double sumNoIterations2 = 0.0; 
        
        float foldCorrectMetric1; 
        float foldCorrectMetric2; 
        float foldCorrectMetric3; 
        float foldCorrectMetric4; 
        
        int correctMetric1 = 0; 
        int correctMetric2 = 0; 
        int correctMetric3 = 0; 
        int correctMetric4 = 0; 
        
       
         System.out.println("f,lwmnb\t\t\t,");
         for (int fold = 0; fold < 5; fold++) {
           
             // train and test sets
            Instances train = insts.trainCV(5 ,fold);
            Instances test = insts.testCV(5, fold);
            //-------------------------------------------
      
            
         // build classifiers using train
         
          LWMnb LMNB = new LWMnb();
          LMNB.setMaxMinFreq(train);
          //-------------------------------------------
            
            foldCorrectMetric1 = 0;
            foldCorrectMetric2 = 0;
            foldCorrectMetric3 = 0; 
            foldCorrectMetric4 = 0;
            
            try {
               

             // System.out.println(test.numInstances());
              
             for (int i = 0; i < test.numInstances(); i++) {
                  
                LMNB.buildClassifier(train, test.instance(i), 30);
             
                     if (LMNB.classifyInstance(test.instance(i)) == test.instance(i).classValue()) {
                        correctMetric1++;
                       foldCorrectMetric1++;}
                
                } //for i
                
                sumNoIterations2 = sumNoIterations2 / test.numInstances();
              System.out.println(fold+","+(double) foldCorrectMetric1 / test.numInstances());


                 


                   
            } catch (Exception ex) {
                ex.printStackTrace();
                System.err.println(ex.getMessage());
            } //catch
            
            
        }  //for fold


        acc.accMetric1=(double) correctMetric1 / insts.numInstances();
        acc.accMetric2=(double) correctMetric2 / insts.numInstances();
        acc.accMetric3 = (double) correctMetric3 / insts.numInstances();
        
        acc.accMetric4 = (double) correctMetric4 / insts.numInstances();

         
        double avgNoOfIterations=sumNoIterations/10.0;
        
        double avgNoOfIterations2=sumNoIterations2/10.0;
        
       // System.out.println(" average ,  "+(double)correctMetric1 / insts.numInstances()+" ,  " +(double) correctMetric2 / insts.numInstances() +" ,  " +(double) correctMetric3 / insts.numInstances()); 
       System.out.println("average,");
       System.out.println(","+(double) correctMetric1 / insts.numInstances() +"\t\t,"+(double) correctMetric2 / insts.numInstances() +"\t\t  ," +(double) correctMetric3 / insts.numInstances() +"\t\t ,"+ (double) correctMetric4 / insts.numInstances() ); 

        return acc;
    }
   
           
       }
        
        
     
     
     
     
     
    

