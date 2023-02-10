package ai.neural_network;

import ai.fx.util.FxNeuralNetwork;
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
    private final int outputWeightIdx;
    private final int outputBiasIdx;
    private final int batchSize = 1000;
    private final int numEpochs = 2500;
    private final double weightMultiplier = .1;
    private final double biasMultiplier = .1;
    private double[][][][] deltaWeights;
    private double[][][] deltaBiases;
    private final String[] outputNames;
    private final FxNeuralNetwork fx;

    public NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int nodesPerHiddenLayer) {
        this(numInputs, numOutputs, numHiddenLayers, nodesPerHiddenLayer, null);
    }

    public NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int nodesPerHiddenLayer, FxNeuralNetwork fx) {
        this.fx = fx;

        numInputNodes = numInputs;
        numOutputNodes = numOutputs;
        outputIdx = numHiddenLayers + 1;
        outputWeightIdx = outputIdx - 1;
        outputBiasIdx = outputIdx - 1;

        activations = new double[numHiddenLayers + 2][];
        declareActivationArrays(numInputs, numOutputs, nodesPerHiddenLayer);

        weights = new double[numHiddenLayers + 1][][];
        deltaWeights = new double[numHiddenLayers + 1][][][];
        declareWeightArrays(numInputs, numOutputs, nodesPerHiddenLayer);

        biases = new double[numHiddenLayers + 1][];
        deltaBiases = new double[numHiddenLayers + 1][][];
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
        deltaWeights[0] = new double[nodesPerHiddenLayer][numInputs][batchSize];
        for (int i = 1; i < weights.length - 1; i++) {
            weights[i] = new double[nodesPerHiddenLayer][nodesPerHiddenLayer];
            deltaWeights[i] = new double[nodesPerHiddenLayer][nodesPerHiddenLayer][batchSize];
        }
        weights[outputWeightIdx] = new double[numOutputs][nodesPerHiddenLayer];
        deltaWeights[outputWeightIdx] = new double[numOutputs][nodesPerHiddenLayer][batchSize];
    }

    private void declareBiasArrays(int numOutputs, int nodesPerHiddenLayer) {
        for (int i = 0; i < biases.length - 1; i++) {
            biases[i] = new double[nodesPerHiddenLayer];
            deltaBiases[i] = new double[nodesPerHiddenLayer][batchSize];
        }
        biases[outputBiasIdx] = new double[numOutputs];
        deltaBiases[outputBiasIdx] = new double[numOutputs][batchSize];
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
        double totalCost = 0;
        for (AiData image : dataSet) {
            setInputs(image);
            feedForward();
            if(getResult(false).equals(image.getLabel()))
            {
                numberOfCorrectAnswers++;
            }
            totalCost += getCost(image.getLabel());
        }
        //output percentage correct to 2 decimal places
        System.out.println("Percent correct: " + Math.round(numberOfCorrectAnswers / dataSet.size() * 10000.0) / 100.0 + "%");
        System.out.println("Average cost: " + totalCost / dataSet.size());
    }

    public void learn(Collection<AiData> dataset) {
        //making a copy of the dataset so that it can be shuffled without affecting the original
        randomiseWeightsAndBiases();
        TrainingBatchHandler<AiData> batchHandler = new TrainingBatchHandler<>(new ArrayList<>(dataset));

        long startTime = System.currentTimeMillis();

        int[] i = new int[1];

        Thread thread = new Thread(() -> {
            long start = System.currentTimeMillis();
            long intervalInMs = 1000;
            while (i[0] < numEpochs) {
                long currTime = System.currentTimeMillis();
                if(currTime - start > intervalInMs) {
                    double percentDone = (double) i[0] / (double)numEpochs * 100;

                    System.out.println(
                            "Batch Number: " +
                                    String.format("%.2f", percentDone) + "%" +
                                    " | Time passed: " +
                                    TimeHelper.getTimeString(((System.currentTimeMillis() - startTime))) +
                                    " | Time remaining: " +
                                    TimeHelper.getTimeString((long) ((System.currentTimeMillis() - startTime) / percentDone * (100 - percentDone)))
                    );
                    start = currTime;
                }
            }
        });
        thread.start();

        for (; i[0] < numEpochs; i[0]++) {
            List<AiData> currentBatch = batchHandler.getNewRandomBatch(batchSize);

            int batchNum = 0;
            for (AiData currentData : currentBatch) {
                setInputs(currentData);
                feedForward();

                double[] unhappiness = new double[numOutputNodes];
                for (int j = 0; j < numOutputNodes; j++) {
                    unhappiness[j] =
                            currentData.getLabel().equals(outputNames[j]) ? 1 : 0 -
                                    activations[outputIdx][j];
                }
                backprop(outputIdx, unhappiness, batchNum);
                batchNum++;
            }
            updateWeightsAndBiases();
        }
    }

    public void backprop(int idx, double[] unhappiness, int batchNum) {
        if (idx <= 0) {
            return;
        }

        int currLayerIdx = idx;
        int layerBeforeIdx = idx - 1;
        int weightAndBiasIdx = idx - 1;
        double[] prevLayer = activations[layerBeforeIdx];
        double[][] currWeights = weights[weightAndBiasIdx];
        double[] currBiases = biases[weightAndBiasIdx];
        double[] currLayer = activations[currLayerIdx];

        //set next unhappiness array for the next recursion
        double[] unhappinessPrevLayer = new double[prevLayer.length];
        for (int prev = 0; prev < prevLayer.length; prev++) {
            double sumForUnhappiness = 0;
            for (int curr = 0; curr < currLayer.length; curr++) {
                sumForUnhappiness += currWeights[curr][prev] * unhappiness[curr];
            }
            unhappinessPrevLayer[prev] = sumForUnhappiness;
        }

        //apply weight and bias changes
        for (int curr = 0; curr < currLayer.length; curr++) {
            for (int prev = 0; prev < prevLayer.length; prev++) {
                double deltaWeight = unhappiness[curr] * activations[layerBeforeIdx][prev];
                deltaWeights[weightAndBiasIdx][curr][prev][batchNum] = deltaWeight;
                //currWeights[curr][prev] += deltaWeight;
            }
            double deltaBias = unhappiness[curr];
            deltaBiases[weightAndBiasIdx][curr][batchNum] = deltaBias;
            //currBiases[curr] += deltaBias;
        }

        backprop(idx - 1, unhappinessPrevLayer, batchNum);
    }
    private void updateWeightsAndBiases() {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    double sum = 0;
                    for (int l = 0; l < batchSize; l++) {
                        sum += deltaWeights[i][j][k][l];
                    }
                    weights[i][j][k] += sum / batchSize * weightMultiplier;
                }
            }
        }
        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                double sum = 0;
                for (int k = 0; k < batchSize; k++) {
                    sum += deltaBiases[i][j][k];
                }
                biases[i][j] += sum / batchSize * biasMultiplier;
            }
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
        double[][] currWeights = this.weights[layerIdx - 1];
        double[] currBiases = this.biases[layerIdx - 1];

        if (currBiases.length != currWeights.length) {
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

    public void updateFx() {
        for (int i = 0; i < activations.length; i++) {
            for (int j = 0; j < activations[i].length; j++) {
                fx.setActivationAt(i, j, activations[i][j]);
            }
        }
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    fx.setWeightAt(i, j, k, weights[i][j][k]);
                }
            }
        }
        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                fx.setBiasAt(i, j, biases[i][j]);
            }
        }
    }

    public FxNeuralNetwork getFx() {
        return fx;
    }
}