
import ai.neural_network.NeuralNetwork;
import ai.reader.ImageManager;
import ai.reader.MnistMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

class TestNeuralNetwork {
    private List<MnistMatrix> getUnitTestImages() {
        return ImageManager.getImages(
                "data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte")
                .subList(0, 1000);
    }
    @Test
    //idempotent means that the result of a function is the same if it is called multiple times
    void testIdempotenceOfFeedForward() {

        NeuralNetwork nn = new NeuralNetwork(getUnitTestImages().get(0).getSizeOfInputs(), 10, 2, 16);


        for (MnistMatrix image : getUnitTestImages()) {
            for (int j = 0; j < 2; j++) {
                nn.randomiseWeightsAndBiases();
                String currHash = null;
                for (int i = 0; i < 5; i++) {
                    nn.setInputs(image);
                    nn.feedForward();
                    nn.getOutputHash(image.getLabel());
                    if (currHash == null) {
                        currHash = nn.getOutputHash(image.getLabel());
                    } else {
                        Assertions.assertEquals(currHash, nn.getOutputHash(image.getLabel()));
                    }
                }
            }
        }
    }
    @Test
    void testIfNewNetworkOutputsNeutralValues() {
        NeuralNetwork nn = new NeuralNetwork(getUnitTestImages().get(0).getSizeOfInputs(), 10, 2, 16);

        nn.feedForward();
        double[] output = nn.getCopyOfOutput();
        for (double value : output) {
            Assertions.assertEquals(0.5, value);
        }
    }
}