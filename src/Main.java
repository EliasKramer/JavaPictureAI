import data.reader.ImageManager;
import data.reader.MnistMatrix;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");

        List<MnistMatrix> trainingImages = ImageManager.getImages(
                "data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte");
        List<MnistMatrix> testImages = ImageManager.getImages(
                "data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte");


        //data.reader.ImageManager.printImage(images.get(0));
        NeuralNetwork nn = new NeuralNetwork(Constants.INPUT_SIZE, 10, 2, 16);
        //nn.randomiseWeightsAndBiases();

        //ImageManager.printImages(trainingImages.subList(0, 10));
        nn.randomiseWeightsAndBiases();
        nn.trainOnData(trainingImages.subList(0, 1000), false);

        nn.testOnData(testImages, false);
       // nn.printHiddenLayerValues();
        //nn.printWeightsAndBiases();
        //nn.randomiseWeightsAndBiases();
        //nn.setInputs(images.get(0));

        //nn.benchmarkCalculate(10000);
         /*

        System.out.println("calculate");
        nn.calculate();
        System.out.println(nn.getResult(true));
        System.out.println("calculate threaded");
        nn.calculate();
        System.out.println(nn.getResult(true));*/
    }
}