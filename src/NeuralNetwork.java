import time.TimeHelper;
import data.reader.MnistMatrix;

import java.sql.Timestamp;
import java.util.Collection;
import java.util.*;

public class NeuralNetwork {
    private final int numHiddenLayers;
    private final int nodesPerHiddenLayer;
    private final double[] inputLayer;
    private final double[][] hiddenLayer;
    private double[] outputLayer;
    //first index is layer, second is node, third is weight
    private final double[][][] weights;
    private final double[][][] costLastWeightChange;
    private final double[][] biases;
    private final double[][] costLastBiasChange;
    private final double maxCost = 10;

    public NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int nodesPerHiddenLayer) {
        this.numHiddenLayers = numHiddenLayers;
        this.nodesPerHiddenLayer = nodesPerHiddenLayer;
        inputLayer = new double[numInputs];
        hiddenLayer = new double[numHiddenLayers][nodesPerHiddenLayer];
        outputLayer = new double[numOutputs];

        costLastWeightChange = new double[numHiddenLayers + 1][][];
        weights = new double[numHiddenLayers + 1][][];

        weights[0] = new double[nodesPerHiddenLayer][numInputs];
        costLastWeightChange[0] = new double[nodesPerHiddenLayer][numInputs];

        for (int i = 1; i < numHiddenLayers; i++) {
            weights[i] = new double[nodesPerHiddenLayer][nodesPerHiddenLayer];
            costLastWeightChange[i] = new double[nodesPerHiddenLayer][nodesPerHiddenLayer];
        }
        weights[numHiddenLayers] = new double[numOutputs][nodesPerHiddenLayer];
        costLastWeightChange[numHiddenLayers] = new double[numOutputs][nodesPerHiddenLayer];

        biases = new double[numHiddenLayers + 1][];
        costLastBiasChange = new double[numHiddenLayers + 1][];

        biases[numHiddenLayers] = new double[outputLayer.length];
        costLastBiasChange[numHiddenLayers] = new double[outputLayer.length];
        for (int i = 0; i < numHiddenLayers; i++) {
            biases[i] = new double[nodesPerHiddenLayer];
            costLastBiasChange[i] = new double[nodesPerHiddenLayer];
        }
        initBeforeArrays();
        initArrays();
    }

    private void initArrays() {
        double initWeight = 0.0;
        double initBias = 0.0;
        for (double[][] layer : weights) {
            for (double[] node : layer) {
                for (int i = 0; i < node.length; i++) {
                    node[i] = initWeight;
                }
            }
        }
        for (double[] layer : biases) {
            for (int i = 0; i < layer.length; i++) {
                layer[i] = initBias;
            }
        }
    }

    private void initBeforeArrays() {

        double initLastWeight = maxCost;
        //last weight change init to initLastWeight
        for (double[][] layer : costLastWeightChange) {
            for (double[] node : layer) {
                Arrays.fill(node, initLastWeight);
            }
        }
        double initLastBias = 1.0;
        //last bias change init to initLastBias
        for (double[] layer : costLastBiasChange) {
            Arrays.fill(layer, initLastBias);
        }
    }

    public void trainOnData(Collection<MnistMatrix> images, boolean detail) {
        int amountDone = 0;
        int total = images.size();
        double percentCorrect = 0;
        Timestamp start = new Timestamp(System.currentTimeMillis());
        for (MnistMatrix image : images) {
            amountDone++;
            Timestamp startOfLearning = new Timestamp(System.currentTimeMillis());
            setInputs(image);
            //cheapSolve(image.getLabel());
            percentCorrect += image.getLabel() == getResult(false) ? 1 : 0;
            System.out.println("result : " + getResult(detail));
            System.out.println("expected value : " + image.getLabel());
            System.out.println("cost : " + getCost(image.getLabel()));
            System.out.println("percent correct : " + percentCorrect / amountDone * 100 + "%");
            Timestamp endOfLearning = new Timestamp(System.currentTimeMillis());

            System.out.println("time taken: " + (endOfLearning.getTime() - startOfLearning.getTime()) + "ms");

            // time remaining till finish
            long timeRemaining = (endOfLearning.getTime() - startOfLearning.getTime()) * (total - amountDone);
            System.out.println("time remaining: " + TimeHelper.getTimeString(timeRemaining));
            long timePerPicture = (endOfLearning.getTime() - startOfLearning.getTime());
            System.out.println("done " + amountDone + "/" + total);
            System.out.println("-------------------------------");
        }
    }

    public double avgCostOfSet(Collection<MnistMatrix> images) {
        double totalCost = 0;
        for (MnistMatrix image : images) {
            setInputs(image);
            feedForward();
            totalCost += getCost(image.getLabel());
        }
        return totalCost / images.size();
    }

    public void testOnData(Collection<MnistMatrix> images, boolean detail) {
        double numberOfCorrectAnswers = 0;
        for (MnistMatrix image : images) {
            setInputs(image);
            feedForward();
            if (getResult(false) == image.getLabel()) {
                numberOfCorrectAnswers++;
            }
            if (detail) {
                System.out.println("result : " + getResult(false));
                System.out.println("expected value : " + image.getLabel());
                System.out.println("cost : " + getCost(image.getLabel()));
                System.out.println("-------------------------------");
            }
        }
        //output percentage correct to 2 decimal places
        System.out.println("Percent correct: " + Math.round(numberOfCorrectAnswers / images.size() * 10000.0) / 100.0 + "%");
    }

    public void cheapSolve(Collection<MnistMatrix> images) {
        feedForward();

        double initialStepSizeWeight = .05;
        double initialStepSizeBias = .05;
        int iterationLimit = 20;
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

            //iterate over layers
            double costBefore = avgCostOfSet(images);

            for (int layerIdx = 0; layerIdx < weights.length; layerIdx++) {
                System.out.println("layer " + layerIdx);
                for (int nodeIdx = 0; nodeIdx < weights[layerIdx].length; nodeIdx++) {
                    for (int weightIdx = 0; weightIdx < weights[layerIdx][nodeIdx].length; weightIdx++) {
                        double weightBefore = weights[layerIdx][nodeIdx][weightIdx];

                        double proportionalStepSize =
                                initialStepSizeWeight;
                        //(initialStepSizeWeight +
                        //(costLastWeightChange[layerIdx][nodeIdx][weightIdx]));

                        weights[layerIdx][nodeIdx][weightIdx] =
                                weightBefore + proportionalStepSize;
                        double costAddingStep = avgCostOfSet(images);

                        weights[layerIdx][nodeIdx][weightIdx] =
                                weightBefore - proportionalStepSize;
                        double costSubtractingStep = avgCostOfSet(images);

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
                            initialStepSizeBias;
                    //(initialStepSizeBias +
                    //(costLastBiasChange[layerIdx][nodeIdx]));

                    biases[layerIdx][nodeIdx] =
                            biasBefore + proportionalStepSize;
                    double costAddingStep = avgCostOfSet(images);

                    biases[layerIdx][nodeIdx] =
                            biasBefore - proportionalStepSize;
                    double costSubtractingStep = avgCostOfSet(images);

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
            trainOnChangedSetAgain(images);

            lastCost = currCost;
            currCost = avgCostOfSet(images);

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

            if(lastCost - currCost < minCostChange &&
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
            long timeRemaining = (endOfLearning.getTime() - start.getTime()) *
                    ((long)iterationLimit - (long)iteration);
            System.out.println("time remaining: " + TimeHelper.getTimeString(timeRemaining));

            System.out.println("-------------------------------");
        }
    }
    private void printAvgAbsCostOfWeights()
    {
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
            output += "layer"+layerIdx+": " + (totalCost / totalWeights)+", \n";
        }
        System.out.println(output);
    }
    private void printAvgAbsCostOfBiases()
    {
        System.out.println("avg abs cost of biases");
        String output = "";
        for (int layerIdx = 0; layerIdx < costLastBiasChange.length; layerIdx++) {
            double totalCost = 0;
            int totalBiases = 0;
            for (int nodeIdx = 0; nodeIdx < costLastBiasChange[layerIdx].length; nodeIdx++) {
                totalCost += Math.abs(biases[layerIdx][nodeIdx]);
                totalBiases++;
            }
            output += "layer"+layerIdx+": " + (totalCost / totalBiases)+", \n";
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

    public void setInputs(MnistMatrix image) {
        for (int i = 0; i < image.getNumberOfRows(); i++) {
            for (int j = 0; j < image.getNumberOfColumns(); j++) {
                inputLayer[i * image.getNumberOfColumns() + j] = ((double) (image.getValue(i, j))) / 255d;
            }
        }
    }

    public void feedForward() {

        //input calculation to first hidden layer
        hiddenLayer[0] = calculateLayer(inputLayer, weights[0], biases[0]);

        //hidden layer calculations
        for (int i = 1; i < numHiddenLayers; i++) {
            hiddenLayer[i] =
                    calculateLayer(
                            hiddenLayer[i - 1],
                            weights[i],
                            biases[i]);
        }
        //calculate last hidden layer to output layer
        outputLayer =
                calculateLayer(
                        hiddenLayer[numHiddenLayers - 1],
                        weights[numHiddenLayers],
                        biases[numHiddenLayers]);
    }

    /*
    public void calculateThreaded() {

        //input calculation to first hidden layer
        calculateLayerThreaded(inputLayer, hiddenLayer[0], weights[0], biases[0]);

        //hidden layer calculations
        for (int i = 1; i < numHiddenLayers; i++) {
            calculateLayerThreaded(
                    hiddenLayer[i - 1],
                    hiddenLayer[i],
                    weights[i],
                    biases[i]);
        }
        //calculate last hidden layer to output layer
        calculateLayerThreaded(
                hiddenLayer[numHiddenLayers - 1],
                outputLayer,
                weights[numHiddenLayers],
                biases[numHiddenLayers]);
    }
    */
    public double[] calculateLayer(double[] layerBefore, double[][] weights, double[] biases) {
        double[] result = new double[biases.length];
        for (int i = 0; i < weights.length; i++) {
            double value = 0;
            for (int j = 0; j < layerBefore.length; j++) {
                value = value + (layerBefore[j] * weights[i][j]);
            }
            value += biases[i];
            result[i] = Sigmoid(value + biases[i]);
        }
        return result;
    }

    public void printHiddenLayerValues() {
        System.out.println("Hidden Layer Values:");
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[i].length; j++) {
                System.out.print("layer: " + i + " node: " + j + " = " + hiddenLayer[i][j] + "\n");
            }
            System.out.println("----------------");
        }
    }

    public String getWeightsAndBiases() {
        StringBuilder sb = new StringBuilder();

        sb.append("Weights and Biases:\n");
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    sb.append("layer: " + i + " node: " + j + " weight: " + k + " = weight " + weights[i][j][k] + "\n");
                }
                sb.append("layer: " + i + " node: " + j + " = bias " + biases[i][j] + "\n");
                sb.append("----------------\n");
            }
            sb.append("----------------\n");
        }
        return sb.toString();
    }

    public String getLastCostChange() {
        StringBuilder sb = new StringBuilder();

        sb.append("Last Cost Change:\n");
        for (int i = 0; i < costLastWeightChange.length; i++) {
            for (int j = 0; j < costLastWeightChange[i].length; j++) {
                for (int k = 0; k < costLastWeightChange[i][j].length; k++) {
                    sb.append("layer: " + i + " node: " + j + " weight: " + k + " = weight change " + costLastWeightChange[i][j][k] + "\n");
                }
                sb.append("layer: " + i + " node: " + j + " = bias change " + costLastBiasChange[i][j] + "\n");
                sb.append("----------------\n");
            }
            sb.append("----------------\n");
        }

        return sb.toString();
    }

    public double getCost(double expectedIndex) {
        double cost = 0d;
        for (int i = 0; i < outputLayer.length; i++) {
            double expectedValue = i == expectedIndex ? 1d : 0d;
            cost += Math.pow(outputLayer[i] - expectedValue, 2);
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

    public int getResult(boolean print) {
        double max = 0d;
        int index = 0;
        for (int i = 0; i < outputLayer.length; i++) {
            if (outputLayer[i] > max) {
                max = outputLayer[i];
                index = i;
            }
            if (print) {
                System.out.println("Output " + i + ": " + outputLayer[i]);
            }
        }
        return index;
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

    private double randomInclusive(double min, double max) {
        return Math.random() * (max - min) + min;
    }
}
