package neural_network;

import data.reader.MnistMatrix;
import util.Hasher;

import java.sql.Timestamp;
import java.util.Collection;
import java.util.*;

public class NeuralNetwork {
    private final double[][] activations;
    private final double[][][] weights;
    private final double[][] biases;
    private final int numOutputNodes;
    private final int numInputNodes;
    private final int outputIdx;
    private final String[] outputNames;

    public NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int nodesPerHiddenLayer) {

        numInputNodes = numInputs;
        numOutputNodes = numOutputs;
        outputIdx = numHiddenLayers + 1;

        activations = new double[numHiddenLayers + 2][];
        declareActivationArrays(numInputs, numOutputs, nodesPerHiddenLayer);

        weights = new double[numHiddenLayers + 1][][];
        declareWeightArrays(numInputs, numOutputs, nodesPerHiddenLayer);

        biases = new double[numHiddenLayers + 1][];
        declareBiasArrays(numOutputs, nodesPerHiddenLayer);

        outputNames = new String[numOutputs];
        declareOutputNames();

        initArrays();
    }

    private void declareActivationArrays(int numInputs, int numOutputs, int nodesPerHiddenLayer) {
        activations[0] = new double[numInputs];
        for (int i = 1; i < activations.length - 1; i++) {
            activations[i] = new double[nodesPerHiddenLayer];
        }
        activations[outputIdx] = new double[numOutputs];
    }

    private void declareWeightArrays(int numInputs, int numOutputs, int nodesPerHiddenLayer) {
        weights[0] = new double[nodesPerHiddenLayer][numInputs];
        for (int i = 1; i < weights.length - 1; i++) {
            weights[i] = new double[nodesPerHiddenLayer][nodesPerHiddenLayer];
        }
        weights[weights.length - 1] = new double[numOutputs][nodesPerHiddenLayer];
    }

    private void declareBiasArrays(int numOutputs, int nodesPerHiddenLayer) {
        for (int i = 0; i < biases.length - 1; i++) {
            biases[i] = new double[nodesPerHiddenLayer];
        }
        biases[biases.length - 1] = new double[numOutputs];
    }

    private void declareOutputNames() {
        for (int i = 0; i < outputNames.length; i++) {
            outputNames[i] = String.valueOf(i);
        }
    }

    private void initArrays() {
        final double initVal = 0;
        for (double[] activation : activations) {
            Arrays.fill(activation, initVal);
        }
        for (double[][] weight : weights) {
            for (double[] doubles : weight) {
                Arrays.fill(doubles, initVal);
            }
        }
        for (double[] bias : biases) {
            Arrays.fill(bias, initVal);
        }
    }

    public void backprop(Collection<MnistMatrix> images) {
        //random weights and biases
        //the gradient has to be set to 0 before backprop
        //for loop over every image
        //set input to current image and feed forward.

        //for every output layer node
        //factor 1 = diffence to the expected value of the output layer
        //factor 2 = activation
        //
    }

    public double avgCostOfSet(Collection<AiData> dataset) {
        double totalCost = 0;
        for (AiData image : dataset) {
            setInputs(image);
            feedForward();
            totalCost += getCost(image.getLabel());
        }
        return totalCost / dataset.size();
    }

    public void testOnData(Collection<AiData> dataSet) {
        double numberOfCorrectAnswers = 0;
        for (AiData image : dataSet) {
            setInputs(image);
            feedForward();
        }
        //output percentage correct to 2 decimal places
        System.out.println("Percent correct: " + Math.round(numberOfCorrectAnswers / dataSet.size() * 10000.0) / 100.0 + "%");
    }

    /*
    public void cheapSolve(List<AiData> images) {
        feedForward();
        int sublistSize = 400;
        int iterationLimit = 20;

        double initialStepSizeWeight = .05;
        double initialStepSizeBias = .05;
        double lastCost;
        double currCost = Double.MAX_VALUE;

        double stepMult = .25;
        double minCostChange = .005;

        int weightsNotChanged = 0;
        int weightsChangedAdded = 0;
        int weightsChangedSubtracted = 0;
        int biasNotChanged = 0;
        int biasChangedAdded = 0;
        int biasChangedSubtracted = 0;

        int iterationIdxOnLastStepsizeChange = 0;


        Timestamp start = new Timestamp(System.currentTimeMillis());

        for (int iteration = 0; iteration < iterationLimit; iteration++) {
            System.out.println("iteration " + iteration);

            //random subset of images to train on
            int imagesSize = images.size() - sublistSize;
            int randomIdx = (int) (Math.random() * imagesSize);
            List<AiData> imageSubset = images.subList(randomIdx, randomIdx + sublistSize);
            //iterate over layers
            double costBefore = avgCostOfSet(imageSubset);

            for (int layerIdx = 0; layerIdx < weights.length; layerIdx++) {
                System.out.println("layer " + layerIdx);
                for (int nodeIdx = 0; nodeIdx < weights[layerIdx].length; nodeIdx++) {
                    for (int weightIdx = 0; weightIdx < weights[layerIdx][nodeIdx].length; weightIdx++) {
                        double weightBefore = weights[layerIdx][nodeIdx][weightIdx];
                        double costBeforeActual = avgCostOfSet(imageSubset);
                        if (costBeforeActual != costBefore) {
                            throw new RuntimeException("cost before actual is not equal to cost before");
                        }
                        double proportionalStepSize =
                                //initialStepSizeWeight;
                                (initialStepSizeWeight +
                                        (costLastWeightChange[layerIdx][nodeIdx][weightIdx]));

                        weights[layerIdx][nodeIdx][weightIdx] =
                                weightBefore + proportionalStepSize;
                        double costAddingStep = avgCostOfSet(imageSubset);

                        weights[layerIdx][nodeIdx][weightIdx] =
                                weightBefore - proportionalStepSize;
                        double costSubtractingStep = avgCostOfSet(imageSubset);

                        if (costBefore < costAddingStep && costBefore < costSubtractingStep) {
                            //if the cost is lower without changing the weight, don't change the weight
                            weights[layerIdx][nodeIdx][weightIdx] = weightBefore;
                            costLastWeightChange[layerIdx][nodeIdx][weightIdx] = 0;
                            weightsNotChanged++;
                        } else if (costAddingStep <= costSubtractingStep) {
                            //if the cost is lower with the weight increased, increase the weight
                            weights[layerIdx][nodeIdx][weightIdx] = weightBefore + proportionalStepSize;
                            costLastWeightChange[layerIdx][nodeIdx][weightIdx] = (costBefore - costAddingStep);
                            costBefore = costAddingStep;
                            weightsChangedAdded++;
                        } else {
                            //if the cost is lower with the weight decreased, decrease the weight
                            weights[layerIdx][nodeIdx][weightIdx] = weightBefore - proportionalStepSize;
                            costLastWeightChange[layerIdx][nodeIdx][weightIdx] = (costBefore - costSubtractingStep);
                            costBefore = costSubtractingStep;
                            weightsChangedSubtracted++;
                        }
                    }
                }
            }

            for (int layerIdx = 0; layerIdx < biases.length; layerIdx++) {
                for (int nodeIdx = 0; nodeIdx < biases[layerIdx].length; nodeIdx++) {
                    double biasBefore = biases[layerIdx][nodeIdx];
                    double proportionalStepSize =
                            //initialStepSizeBias;
                            (initialStepSizeBias +
                                    (costLastBiasChange[layerIdx][nodeIdx]));

                    biases[layerIdx][nodeIdx] =
                            biasBefore + proportionalStepSize;
                    double costAddingStep = avgCostOfSet(imageSubset);

                    biases[layerIdx][nodeIdx] =
                            biasBefore - proportionalStepSize;
                    double costSubtractingStep = avgCostOfSet(imageSubset);

                    //if the cost is lower without changing the bias, don't change the bias
                    if (costBefore < costAddingStep && costBefore < costSubtractingStep) {
                        //if the cost is lower without changing the bias, don't change the bias
                        biases[layerIdx][nodeIdx] = biasBefore;
                        costLastBiasChange[layerIdx][nodeIdx] = 0;
                        biasNotChanged++;
                    }
                    //if the cost is lower with the bias increased, increase the bias
                    else if (costAddingStep <= costSubtractingStep) {
                        //if the cost is lower with the bias increased, increase the bias
                        biases[layerIdx][nodeIdx] = biasBefore + proportionalStepSize;
                        costLastBiasChange[layerIdx][nodeIdx] = (costBefore - costAddingStep);
                        costBefore = costAddingStep;
                        biasChangedAdded++;
                    }
                    //if the cost is lower with the bias decreased, decrease the bias
                    else {
                        //if the cost is lower with the bias decreased, decrease the bias
                        biases[layerIdx][nodeIdx] = biasBefore - proportionalStepSize;
                        costLastBiasChange[layerIdx][nodeIdx] = (costBefore - costSubtractingStep);
                        costBefore = costSubtractingStep;
                        biasChangedSubtracted++;
                    }
                }
            }
            trainOnChangedSetAgain(imageSubset);

            lastCost = currCost;
            currCost = avgCostOfSet(imageSubset);

            System.out.println("currcost: " + currCost);

            System.out.println("weights not changed " + weightsNotChanged);
            System.out.println("weights changed added " + weightsChangedAdded);
            System.out.println("weights changed subtracted " + weightsChangedSubtracted);
            System.out.println("bias not changed " + biasNotChanged);
            System.out.println("bias changed added " + biasChangedAdded);
            System.out.println("bias changed subtracted " + biasChangedSubtracted);
            System.out.println("last changed " + (lastCost - currCost));

            printAvgAbsCostOfWeights();
            printAvgAbsCostOfBiases();

            if (lastCost - currCost < minCostChange &&
                    //you can only half the step size every 5 iterations
                    iterationIdxOnLastStepsizeChange > 5) {
                System.out.println("changing step size");
                System.out.println("last cost: " + lastCost + " curr cost: " + currCost);
                initialStepSizeWeight *= stepMult;
                initialStepSizeBias *= stepMult;
                minCostChange /= 10;
                iterationIdxOnLastStepsizeChange = 0;
            }
            iterationIdxOnLastStepsizeChange++;

            weightsNotChanged = 0;
            weightsChangedAdded = 0;
            weightsChangedSubtracted = 0;
            biasNotChanged = 0;
            biasChangedAdded = 0;
            biasChangedSubtracted = 0;

            //print estimate of time remaining
            Timestamp endOfLearning = new Timestamp(System.currentTimeMillis());

            long timeSinceStart = (endOfLearning.getTime() - start.getTime());
            int tasksToDo = iterationLimit - iteration;
            long timePerTask = timeSinceStart / (iteration + 1);
            long timeRemaining = timePerTask * tasksToDo;
            System.out.println("time elapsed: " + TimeHelper.getTimeString(timeSinceStart));
            System.out.println("time remaining: " + TimeHelper.getTimeString(timeRemaining));

            System.out.println("-------------------------------");
        }
    }
    private void printAvgAbsCostOfWeights() {
        System.out.println("avg abs cost of weights:");
        String output = "";
        for (int layerIdx = 0; layerIdx < costLastWeightChange.length; layerIdx++) {
            double totalCost = 0;
            int totalWeights = 0;
            for (int nodeIdx = 0; nodeIdx < costLastWeightChange[layerIdx].length; nodeIdx++) {
                for (int weightIdx = 0; weightIdx < costLastWeightChange[layerIdx][nodeIdx].length; weightIdx++) {
                    totalCost += Math.abs(weights[layerIdx][nodeIdx][weightIdx]);
                    totalWeights++;
                }
            }
            output += "layer" + layerIdx + ": " + (totalCost / totalWeights) + ", \n";
        }
        System.out.println(output);
    }

    private void printAvgAbsCostOfBiases() {
        System.out.println("avg abs cost of biases");
        String output = "";
        for (int layerIdx = 0; layerIdx < costLastBiasChange.length; layerIdx++) {
            double totalCost = 0;
            int totalBiases = 0;
            for (int nodeIdx = 0; nodeIdx < costLastBiasChange[layerIdx].length; nodeIdx++) {
                totalCost += Math.abs(biases[layerIdx][nodeIdx]);
                totalBiases++;
            }
            output += "layer" + layerIdx + ": " + (totalCost / totalBiases) + ", \n";
        }
        System.out.println(output);
    }

    private void trainOnChangedSetAgain(Collection<MnistMatrix> images) {
        double lastCost = Double.MAX_VALUE;

        //curr cost is max double
        double currCost = avgCostOfSet(images);

        System.out.println("trying to repeat changes");
        int counter = 0;
        while (currCost < lastCost) {
            lastCost = currCost;
            applyChangesToWeighsAndBiasesAgain();
            currCost = avgCostOfSet(images);
            counter++;
        }
        System.out.println("did it " + counter + " times");

        revertChangesToWeighsAndBiases();

    }

    private void applyChangesToWeighsAndBiasesAgain() {
        for (int layerIdx = 0; layerIdx < costLastWeightChange.length; layerIdx++) {
            for (int nodeIdx = 0; nodeIdx < costLastWeightChange[layerIdx].length; nodeIdx++) {
                for (int weightIdx = 0; weightIdx < costLastWeightChange[layerIdx][nodeIdx].length; weightIdx++) {
                    weights[layerIdx][nodeIdx][weightIdx] += costLastWeightChange[layerIdx][nodeIdx][weightIdx];
                }
            }
        }
        for (int layerIdx = 0; layerIdx < costLastBiasChange.length; layerIdx++) {
            for (int nodeIdx = 0; nodeIdx < costLastBiasChange[layerIdx].length; nodeIdx++) {
                biases[layerIdx][nodeIdx] += costLastBiasChange[layerIdx][nodeIdx];
            }
        }
    }

    private void revertChangesToWeighsAndBiases() {
        for (int layerIdx = 0; layerIdx < costLastWeightChange.length; layerIdx++) {
            for (int nodeIdx = 0; nodeIdx < costLastWeightChange[layerIdx].length; nodeIdx++) {
                for (int weightIdx = 0; weightIdx < costLastWeightChange[layerIdx][nodeIdx].length; weightIdx++) {
                    weights[layerIdx][nodeIdx][weightIdx] -= costLastWeightChange[layerIdx][nodeIdx][weightIdx];
                }
            }
        }
        for (int layerIdx = 0; layerIdx < costLastBiasChange.length; layerIdx++) {
            for (int nodeIdx = 0; nodeIdx < costLastBiasChange[layerIdx].length; nodeIdx++) {
                biases[layerIdx][nodeIdx] -= costLastBiasChange[layerIdx][nodeIdx];
            }
        }
    }
    */
    public void setInputs(AiData image) {
        double[] inputData = image.getInputs();
        if (inputData.length != numInputNodes) {
            throw new IllegalArgumentException("the input picture is not the right size");
        }
        activations[0] = inputData;
    }

    public void feedForward() {
        for (int i = 1; i < activations.length; i++) {
            calculateLayer(i);
        }
    }

    public void calculateLayer(int layerIdx) {
        double[] activationPrevLayer = this.activations[layerIdx - 1];
        double[][] currWeights = this.weights[layerIdx-1];
        double[] currBiases = this.biases[layerIdx-1];

        if(currBiases.length != currWeights.length) {
            throw new IllegalArgumentException("the number of biases does not match the number of nodes in the layer");
        }

        for (int currNodeIdx = 0; currNodeIdx < currBiases.length; currNodeIdx++) {
            double value = 0;
            for (int prevNodeIdx = 0; prevNodeIdx < activationPrevLayer.length; prevNodeIdx++) {
                double activation = activationPrevLayer[prevNodeIdx];
                double weight = currWeights[currNodeIdx][prevNodeIdx];
                value += activation * weight;
            }
            value += currBiases[currNodeIdx];
            activations[layerIdx][currNodeIdx] = Sigmoid(value);
        }
    }

    public String getWeightsAndBiases() {
        StringBuilder sb = new StringBuilder();

        sb.append("Weights and Biases:\n");
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    sb.append("layer: ")
                            .append(i)
                            .append(" node: ")
                            .append(j)
                            .append(" weight: ")
                            .append(k)
                            .append(" = weight ")
                            .append(weights[i][j][k])
                            .append("\n");
                }
                sb.append("layer: ")
                        .append(i)
                        .append(" node: ")
                        .append(j)
                        .append(" = bias ")
                        .append(biases[i][j])
                        .append("\n")
                        .append("----------------\n");
            }
            sb.append("----------------\n");
        }
        return sb.toString();
    }

    public double getCost(String expectedLabel) {
        double cost = 0d;
        for (int i = 0; i < numOutputNodes; i++) {
            double expectedValue =
                    outputNames[i].equals(expectedLabel)
                            ? 1d
                            : 0d;
            cost += Math.pow(activations[outputIdx][i] - expectedValue, 2);
        }
        return cost;
    }

    public void randomiseWeightsAndBiases() {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] = randomInclusive(-1, 1);
                }
            }
        }
        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                biases[i][j] = randomInclusive(-1, 1);
            }
        }
    }

    public String getResult(boolean print) {
        double max = 0d;
        int index = 0;
        for (int i = 0; i < numOutputNodes; i++) {
            if (activations[outputIdx][i] > max) {
                max = activations[outputIdx][i];
                index = i;
            }
            if (print) {
                System.out.println("Output " + outputNames[i] + ": " + activations[outputIdx][i]);
            }
        }
        return outputNames[index];
    }

    public void benchmarkCalculate() {
        benchmarkCalculate(10000);
    }

    public void benchmarkCalculate(int iterations) {
        System.out.println("Benchmarking calculate()...");
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());
        for (int i = 0; i < iterations; i++) {
            feedForward();
        }
        Timestamp timestamp2 = new Timestamp(System.currentTimeMillis());
        double time = (timestamp2.getTime() - timestamp.getTime());
        System.out.println("Time per calculate: " + time / iterations + "ms");
    }

    private static double Sigmoid(double value) {
        return 1d / (1d + Math.exp(-value));
    }

    private double reLu(double value) {
        return Math.max(0, value);
    }

    private double randomInclusive(double min, double max) {
        return Math.random() * (max - min) + min;
    }

    public String getOutputHash(String expectedLabel) {
        StringBuilder sb = new StringBuilder();
        sb.append(getCost(expectedLabel));
        for (int i = 0; i < activations[outputIdx].length; i++) {
            sb.append(activations[outputIdx][i]);
        }
        return Hasher.getHash(sb.toString());
    }
    public double[] getCopyOfOutput() {
        return Arrays.copyOf(activations[outputIdx], activations[outputIdx].length);
    }
}
