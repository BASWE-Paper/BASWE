package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

import java.util.ArrayList;
import java.util.Random;

/**
 * BASWE: Balanced Accuracy-based Sliding Window Ensemble
 * @author Douglas Amorim (daoliveirax@gmail.com)
 * Code based in Kappa Updated Ensemble - KUE, from Alberto Cano and Bartosz Krawczyk
 */

public class BASWE extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    /**
     * Type of classifier to use as a component classifier.
     */
    public ClassOption learnerOption = new ClassOption("learner", 'l', "Classifier to train.", Classifier.class, "trees.HoeffdingTree -e 2000000 -g 100 -c 0.01");

    /**
     * Number of component classifiers.
     */
    public IntOption memberCountOption = new IntOption("memberCount", 'n', "The maximum number of classifiers in an ensemble.", 10, 1, Integer.MAX_VALUE);

    /**
     * Number of new component classifiers.
     */
    public IntOption newmemberCountOption = new IntOption("newmemberCount", 'g', "The new number of classifiers generated.", 1, 1, Integer.MAX_VALUE);

    /**
     * Chunk size.
     */
    public IntOption chunkSizeOption = new IntOption("chunkSize", 'c', "The chunk size used for classifier creation and evaluation.", 1000, 1, Integer.MAX_VALUE);

    /**
     * Ensemble classifiers.
     */
    protected Classifier[] learners;

    /**
     * Candidate classifier.
     */
    protected Classifier candidate;

    /**
     * Current chunk of instances.
     */
    protected Instances currentChunk;

    //private Instances windowA;

    public ArrayList<Instance> windowA = new ArrayList<Instance>();


    public ArrayList<Instance> windowB = new ArrayList<Instance>();

    //private Instances windowB;

    private Instances oversampled;

    private int numberComponentsReplaced;

    private int numberAbstentions;

    private double [] balancedAcc;

    private boolean[][] useAttribute;

    public int counter = 0;

    @Override
    public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        this.candidate = (Classifier) getPreparedClassOption(this.learnerOption);
        this.candidate.resetLearning();

        super.prepareForUseImpl(monitor, repository);
    }

    @Override
    public void resetLearningImpl() {
        this.currentChunk = null;
        this.learners = new Classifier[this.memberCountOption.getValue()];
        this.candidate = (Classifier) getPreparedClassOption(this.learnerOption);
        this.candidate.resetLearning();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

        if (this.currentChunk == null)
            this.currentChunk = new Instances(this.getModelContext());

        if(useAttribute == null)
        {
            int numAtts = this.getModelContext().numAttributes() - 1;
            int numClassifiers = this.memberCountOption.getValue();

            useAttribute = new boolean[numClassifiers][numAtts];

            Random r = new Random();

            for(int i = 0; i < numClassifiers; i++)
            {
                int used = 0;
                int numAttsSelected = 1 + r.nextInt(numAtts);

                while(used < numAttsSelected)
                {
                    int index = r.nextInt(numAtts);

                    if(useAttribute[i][index] == false)
                    {
                        useAttribute[i][index] = true;
                        used++;
                    }
                }
            }
        }

        this.currentChunk.add(inst);

        if(inst.classValue() == 0.0) {
            //Add the instance with class 0 to the sliding window windowA
            this.windowA.add(inst);
            //If the sliding window exceeds the maximum size, remove the oldest element at position 0
            if( this.windowA.size() > (this.chunkSizeOption.getValue() / 2)){
                this.windowA.remove(0);
            }
        }else {
            //Add the instance with class 1 to the sliding window windowB
            this.windowB.add(inst);
            //If the sliding window exceeds the maximum size, remove the oldest element at position 0
            if(this.windowB.size() > (this.chunkSizeOption.getValue() / 2)){
                this.windowB.remove(0);
            }
        }

        if ((this.currentChunk.size() % this.chunkSizeOption.getValue() == 0) || (this.currentChunk.size() >= this.chunkSizeOption.getValue())) {

            counter++;
            int classA = 0;
            int classB = 0;



            for (int num = 0; num < this.currentChunk.size(); num++) {

                Instance copy = this.currentChunk.instance(num).copy();

                if(copy.classValue() == 0.0) {
                    classA++;
                }else {
                    classB++;
                }
            }

            //If the sliding window does not have enough elements, perform oversampling
            ArrayList<Instance> windowAtoLearn = new ArrayList<Instance>(windowA);
            ArrayList<Instance> windowBtoLearn = new ArrayList<Instance>(windowB);
            ArrayList<Instance> mergedList = new ArrayList<Instance>();


            int chunkSize = this.currentChunk.size();

            if(windowAtoLearn.size() < (chunkSize / 2)) {
                windowAtoLearn = oversampling_douglas(windowAtoLearn, chunkSize / 2);
            }
            if(windowBtoLearn.size() < (chunkSize / 2)){
                windowBtoLearn = oversampling_douglas(windowBtoLearn, chunkSize / 2);
            }

            oversampled = this.currentChunk;

            for (int i = 0 ; i < chunkSize ; i++){
                oversampled.delete(0);
            }

            mergedList = mergeLists_douglas(windowAtoLearn , windowBtoLearn);

            int overA = 0;
            int overB = 0;
            for(int i = 0; i < mergedList.size(); i++){
                oversampled.add(mergedList.get(i));
                if(mergedList.get(i).classValue() == 0.0){
                    overA++;
                }else{
                    overB++;
                }
            }

            //If you want to follow the evolution:

            /*System.out.println("");
            System.out.println("============================");
            System.out.println("Chunk: " + counter);
            System.out.println("Original proportion " + classA + " - " + classB);
            System.out.println("WindowA's size: "+this.windowA.size());
            System.out.println("WindowB's size: "+this.windowB.size());
            System.out.println("The list of Instances used in learning, after the merge, has a size of: " + oversampled.size() + ", with " + overA + " instances of class 0 and "+ overB + " instances of class 1.");
            System.out.println("============================");*/



            this.processChunk();
        }
    }

    /**
     * Determines whether the classifier is randomizable.
     */
    public boolean isRandomizable() {
        return true;
    }

    /**
     * Predicts a class for an example.
     */
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();

        if (balancedAcc != null)
        {
            for (int i = 0; i < this.learners.length; i++)
            {
                if(balancedAcc[i] < 0) continue;

                Instance copy = inst.copy();

                for(int j = 0; j < useAttribute[i].length; j++)
                    if(useAttribute[i][j] == false)
                        copy.setValue(j, 0);

                DoubleVector vote = new DoubleVector(this.learners[i].getVotesForInstance(copy));

                if (vote.sumOfValues() > 0.0) {
                    vote.normalize();
                    vote.scaleValues(balancedAcc[i]);
                    combinedVote.addValues(vote);
                }
            }
        }

        return combinedVote.getArrayRef();
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.learners.clone();
    }

    /**
     * Processes a chunk of instances.
     * This method is called after collecting a chunk of examples.
     */
    protected void processChunk() {

        numberAbstentions = 0;

        if (this.learners[0] == null)
        {
            for (int i = 0; i < this.learners.length; i++)
            {
                this.learners[i] = this.candidate.copy();
                this.trainOnChunk(this.learners[i], useAttribute[i]);
            }

            computeBalancedAccuracy();
        }
        else
        {
            for (int i = 0; i < this.learners.length; i++)
            {
                this.trainOnChunk(this.learners[i], useAttribute[i]);
            }

            computeBalancedAccuracy();

            Random r = new Random();
            int numAtts = this.getModelContext().numAttributes() - 1;

            numberComponentsReplaced = 0;


            for(int i = 0; i < newmemberCountOption.getValue(); i++)
            {
                int poorestClassifier = this.getPoorestClassifierIndex();
                Classifier addedClassifier = this.candidate.copy();

                int used = 0;
                int numAttsSelected = 1 + r.nextInt(numAtts);
                boolean[] newAttributeArray = new boolean[numAtts];

                while(used < numAttsSelected)
                {
                    int index = r.nextInt(numAtts);

                    if(newAttributeArray[index] == false)
                    {
                        newAttributeArray[index] = true;
                        used++;
                    }
                }

                this.trainOnChunk(addedClassifier, newAttributeArray);

                double newBalancedAcc = computeBalancedAccuracy(addedClassifier);

                if(newBalancedAcc > balancedAcc[poorestClassifier])
                {
                    this.learners[poorestClassifier] = addedClassifier;
                    balancedAcc[poorestClassifier] = newBalancedAcc;
                    useAttribute[poorestClassifier] = newAttributeArray;
                    numberComponentsReplaced++;
                }
            }
        }

        for (int i = 0; i < this.learners.length; i++)
            if(balancedAcc[i] < 0)
                numberAbstentions++;

        this.currentChunk = null;
        this.candidate = (Classifier) getPreparedClassOption(this.learnerOption);
        this.candidate.resetLearning();
    }

    private void computeBalancedAccuracy() {
        balancedAcc = new double[this.learners.length];

        for (int i = 0; i < this.learners.length; i++)
        {
            int[][] confusionMatrix = new int[modelContext.numClasses()][modelContext.numClasses()];

            for(int j = 0; j < currentChunk.size(); j++)
            {
                int predicted = maxIndex(this.learners[i].getVotesForInstance(currentChunk.instance(j)));
                int actual = (int) currentChunk.instance(j).classValue();
                confusionMatrix[predicted][actual]++;
            }

            balancedAcc[i] = balanced_accuracy(confusionMatrix);
        }
    }

    private double computeBalancedAccuracy(Classifier addedClassifier) {
        int[][] confusionMatrix = new int[modelContext.numClasses()][modelContext.numClasses()];

        for(int j = 0; j < currentChunk.size(); j++)
        {
            int predicted = maxIndex(addedClassifier.getVotesForInstance(currentChunk.instance(j)));
            int actual = (int) currentChunk.instance(j).classValue();
            confusionMatrix[predicted][actual]++;
        }

        return balanced_accuracy(confusionMatrix);
    }

    /**
     * Adds ensemble weights to the measurements.
     */
    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement[] measurements = new Measurement[2];
        measurements[0] = new Measurement("components replaced", numberComponentsReplaced);
        measurements[1] = new Measurement("number abstentions", numberAbstentions);
        return measurements;
    }

    /**
     * Adds a classifier to the storage.
     *
     * @param newClassifier
     *            The classifier to add.
     */
    protected Classifier addToStored(Classifier newClassifier) {
        Classifier addedClassifier = null;
        Classifier[] newStored = new Classifier[this.learners.length + 1];

        for (int i = 0; i < newStored.length; i++) {
            if (i < this.learners.length) {
                newStored[i] = this.learners[i];
            } else {
                newStored[i] = addedClassifier = newClassifier.copy();
            }
        }

        this.learners = newStored;

        return addedClassifier;
    }


    private ArrayList<Instance> oversampling_douglas( ArrayList<Instance> originalList, int size){
        //ArrayList<Instance> windowA = new ArrayList<Instance>();
        Random rand = new Random();
        int originalSize = originalList.size();
        ArrayList<Instance> returnList = new ArrayList<Instance>();
        returnList = originalList;
        for(int i = 0; i < (size - originalSize); i++){
            int randomPosition = rand.nextInt(originalSize);
            returnList.add( originalList.get(randomPosition) );
        }

        return returnList;
    }

    private ArrayList<Instance> mergeLists_douglas( ArrayList<Instance> listA, ArrayList<Instance> listB){

        if(listA.size() != listB.size()){
            System.out.println("You are trying to merge two lists of different sizes. This doesn't make sense within the original scope of the study.");
            return null;
        }
        int listSize = listA.size();
        ArrayList<Instance> returnList = new ArrayList<Instance>();
        Random rand = new Random();

        for(int i = 0; i < listSize; i++){
            int randomPosition = rand.nextInt(listSize - i); //Choosing a random position
            returnList.add(listA.get(randomPosition));
            returnList.add(listB.get(randomPosition));

            listA.remove(randomPosition);
            listB.remove(randomPosition);

        }
        return returnList;
    }

    /**
     * Finds the index of the classifier with the smallest weight.
     * @return
     */
    private int getPoorestClassifierIndex() {
        return minIndex(balancedAcc);
    }

    /**
     * Trains a component classifier on the most recent chunk of data.
     *
     * @param classifierToTrain
     *            Classifier being trained.
     */


    private void trainOnChunk(Classifier classifierToTrain, boolean[] useAttribute) {
        Random r = new Random();

        for (int num = 0; num < this.oversampled.size(); num++) {
            //for (int num = 0; num < this.chunkSizeOption.getValue(); num++) {

            Instance copy = this.oversampled.instance(num).copy();


            for(int i = 0; i < useAttribute.length; i++)
                if(useAttribute[i] == false)
                    copy.setValue(i, 0);

            int k = MiscUtils.poisson(1.0, r);
            if (k > 0) {
                copy.setWeight(copy.weight() * k);
                classifierToTrain.trainOnInstance(copy);
            }
        }
    }

    private int minIndex(double[] doubles) {

        double minimum = Double.MAX_VALUE;
        int minIndex = -1;

        for (int i = 0; i < doubles.length; i++) {
            if ((i == 0) || (doubles[i] < minimum)) {
                minIndex = i;
                minimum = doubles[i];
            }
        }

        return minIndex;
    }

    private int maxIndex(double[] doubles) {

        double maximum = -Double.MAX_VALUE;
        int maxIndex = -1;

        for (int i = 0; i < doubles.length; i++) {
            if ((i == 0) || (doubles[i] > maximum)) {
                maxIndex = i;
                maximum = doubles[i];
            }
        }

        return maxIndex;
    }

    private double balanced_accuracy(int[][] confusionMatrix) {
        int TP = confusionMatrix[0][0];
        int FN = confusionMatrix[0][1];
        int FP = confusionMatrix[1][0];
        int TN = confusionMatrix[1][1];

        double TPR = TP / (double)(TP + FN);
        double TNR = TN / (double)(TN + FP);

        return (TPR + TNR) / 2.0;
    }
}