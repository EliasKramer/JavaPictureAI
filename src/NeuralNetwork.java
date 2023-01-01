public class NeuralNetwork {
    private final int numHiddenLayers;
    private final int nodesPerHiddenLayer;
    private final double[] inputLayer;
    private final double[][] hiddenLayer;
    private final double[] outputLayer;
    //first index is layer, second is node, third is weight
    private final double[][][] weights;
    private final double[][] biases;

    public NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int nodesPerHiddenLayer) {
        this.numHiddenLayers = numHiddenLayers;
        this.nodesPerHiddenLayer = nodesPerHiddenLayer;
        inputLayer = new double[numInputs];
        hiddenLayer = new double[numHiddenLayers][nodesPerHiddenLayer];
        outputLayer = new double[numOutputs];
        weights = new double[numHiddenLayers + 1][][];
        weights[0] = new double[nodesPerHiddenLayer][numInputs];
        for (int i = 1; i < numHiddenLayers; i++) {
            weights[i] = new double[nodesPerHiddenLayer][nodesPerHiddenLayer];
        }
        weights[numHiddenLayers] = new double[numOutputs][nodesPerHiddenLayer];

        biases = new double[numHiddenLayers+1][];
        biases[numHiddenLayers] = new double[outputLayer.length];
        for(int i = 0; i < numHiddenLayers; i++)
        {
            biases[i] = new double[nodesPerHiddenLayer];
        }
    }
    public void setInputs(GrayImage image) {
        int[][] pixels = image.GetPixels();
        for (int i = 0; i < pixels.length; i++) {
            for (int j = 0; j < pixels[i].length; j++) {
                inputLayer[i * pixels[i].length + j] = pixels[i][j];
            }
        }
    }
    public void calculate()
    {
       //input calculation
       for(int i = 0; i < nodesPerHiddenLayer; i++)
       {
           double value = 0;

           for(int j = 0; j < inputLayer.length; j++)
           {
                value = value + (inputLayer[j] * weights[0][i][j]);
           }

           hiddenLayer[0][i] = Sigmoid(value + biases[0][i]);
       }

        for(int i = 1; i < numHiddenLayers; i++)
        {
            for(int j = 0; j < nodesPerHiddenLayer; j++)
            {
                double value = 0d;
                for(int k = 0; k < nodesPerHiddenLayer; k++)
                {
                    value = value + (weights[i][j][k] * hiddenLayer[i][j]);
                }
                hiddenLayer[i][j] = Sigmoid(value + biases[i][j]);
            }
        }
        //output calculation
        for(int i = 0; i < outputLayer.length; i++)
        {
            double value = 0d;

            for(int j = 0; j < nodesPerHiddenLayer; j++)
            {
                value = value + (weights[numHiddenLayers][i][j] * hiddenLayer[numHiddenLayers-1][j]);
            }

            outputLayer[i] = Sigmoid(value + biases[numHiddenLayers][i]);
        }
    }
    public double getCost(double expectedIndex)
    {
        double cost = 0d;
        for(int i = 0; i < outputLayer.length; i++)
        {
            double expectedValue = i == expectedIndex ? 1d : 0d;
            cost += Math.pow(outputLayer[i] - expectedValue,2);
        }
        return cost;
    }
    public void randomiseWeightsAndBiases()
    {
        for(int i = 0; i < weights.length; i++)
        {
            for(int j = 0; j < weights[i].length; j++)
            {
                for(int k = 0; k < weights[i][j].length; k++)
                {
                    weights[i][j][k] = randomInclusive(-1,1);
                }
            }
        }
        for(int i = 0; i < biases.length; i++)
        {
            for(int j = 0; j < biases[i].length; j++)
            {
                biases[i][j] = randomInclusive(-1,1);
            }
        }
    }
    public int getResult(boolean print)
    {
        double max = 0d;
        int index = 0;
        for(int i = 0; i < outputLayer.length; i++)
        {
            if(outputLayer[i] > max)
            {
                max = outputLayer[i];
                index = i;
            }
            if(print)
            {
                System.out.println("Output " + i + ": " + outputLayer[i]);
            }
        }
        return index;
    }
    private static double Sigmoid(double value)
    {
        return 1d / (1d + Math.exp(-value));
    }
    private double randomInclusive(double min, double max)
    {
        return Math.random() * (max - min) + min;
    }
}
