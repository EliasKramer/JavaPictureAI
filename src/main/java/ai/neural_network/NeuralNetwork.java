package ai.neural_network;

import ai.time.TimeHelper;
import ai.util.Hasher;

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
    public void learn(List<AiData> dataset)
    {
        TrainingBatchHandler<AiData> batchHandler = new TrainingBatchHandler<>(dataset);
        int batchSize = 100;

        List<AiData> currentBatch = batchHandler.getNewRandomBatch(batchSize);

        AiData currentData = currentBatch.get(0);
        setInputs(currentData);
        feedForward();
        double[] unhappiness = new double[numOutputNodes];
        for(int i = 0; i < numOutputNodes; i++)
        {
            unhappiness[i] = currentData.getLabel().equals(outputNames[i]) ? 1 : 0 -
                                activations[outputIdx][i];
        }
        for(int i = 0; i < numOutputNodes; i++)
        {
            for(int j = 0; j < weights[outputIdx].length; j++)
            {
                //weights[outputIdx][i][j] += unhappiness[i] * activations[outputIdx - 1][j];
                double deltaWeight = unhappiness[i] * activations[outputIdx - 1][j];
                double deltaBias = unhappiness[i];
            }
        }
    }
    public void backprop(int idx, double[] unhappiness) {
        if(idx == -1)
        {
            return;
        }


    }

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
    public void benchmarkCalculate(int iterations) {
        System.out.println("Benchmarking calculate()...");
        TimeHelper.start("benchmark");
        for (int i = 0; i < iterations; i++) {
            feedForward();
        }
        double time = TimeHelper.stop("benchmark");
        System.out.println("Average time for feeding forward once: " + time / iterations + "ms");
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
