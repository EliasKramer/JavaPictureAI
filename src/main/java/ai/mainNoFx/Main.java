package ai.mainNoFx;

import ai.neural_network.NeuralNetwork;
import ai.reader.ImageManager;
import ai.reader.MnistMatrix;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");

        List<MnistMatrix> trainingImages = ImageManager.getImages(
                "data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte");
        List<MnistMatrix> testImages = ImageManager.getImages(
                "data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte");


        NeuralNetwork nn = new NeuralNetwork(trainingImages.get(0).getSizeOfInputs(), 10, 2, 16);
        nn.setInputs(trainingImages.get(0));
        nn.benchmarkCalculate(1000000);
    }
}