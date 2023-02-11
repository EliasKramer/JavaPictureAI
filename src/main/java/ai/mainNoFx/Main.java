package ai.mainNoFx;

import ai.neural_network.AiData;
import ai.neural_network.SuperNetwork;
import ai.reader.ImageManager;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");

        List<AiData> trainingImages = ImageManager.getImages(
                "data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte");
        List<AiData> testImages = ImageManager.getImages(
                "data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte");

        SuperNetwork superNetwork = new SuperNetwork(2, 8.0, 4.0, 28*28, 10, 2, 4);

        superNetwork.learn(trainingImages, 3, 1, 300, 300);

        superNetwork.getBestNetwork().testOnData(testImages, true);
    }
}