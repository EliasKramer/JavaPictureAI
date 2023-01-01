public class NeuralNetwork {
    private int numHiddenLayers;
    private int nodesPerHiddenLayer;
    private double[] inputLayer;
    private double[][] hiddenLayer;
    private double[] outputLayer;
    //first index is layer, second is node, third is weight
    private double[][][] weights;

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
    }
    public void SetInputs(GrayImage image) {
        int[][] pixels = image.GetPixels();
        for (int i = 0; i < pixels.length; i++) {
            for (int j = 0; j < pixels[i].length; j++) {
                inputLayer[i * pixels[i].length + j] = pixels[i][j];
            }
        }
    }
    public void calculate()
    {
        for(int i = 0; i < numHiddenLayers; i++)
        {
            for(int j = 0; j < nodesPerHiddenLayer; j++)
            {
                double value = 0d;
                int numOfPreviousLayerNodes = i == 0 ? inputLayer.length : hiddenLayer.length;
                for(int k = 0; k < numOfPreviousLayerNodes; k++)
                {
                    value = value + (weights[i][j][k] * hiddenLayer[i][j]);
                }
                hiddenLayer[i][j] = Sigmoid(value);
            }
        }
        //output calculation
    }
    private static double Sigmoid(double value)
    {
        return 1d / (1d + Math.exp(-value));
    }
}
