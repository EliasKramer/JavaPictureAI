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
        //nn.setInputs(trainingImages.get(0));

        //nn.trainOnData(trainingImages.subList(0, 1000), false);
        //ImageManager.printImages(trainingImages.subList(0, 10));
        //nn.randomiseWeightsAndBiases();
        //nn.trainOnData(trainingImages.subList(18, 20), false);
        //nn.trainOnData(trainingImages.subList(18, 19), false);
        //nn.trainOnData(trainingImages.subList(19, 20), false);
        //AnalyseData.saveStringInTxt(nn.getWeightsAndBiases(), "schlecht");

        //nn.testOnData(testImages, false);
        //nn.printHiddenLayerValues();
        //nn.printLastCostChange();
        //nn.randomiseWeightsAndBiases();
        nn.setInputs(testImages.get(0));
        nn.benchmarkCalculate(100000);
        /*

        System.out.println("calculate");
        nn.calculate();
        System.out.println(nn.getResult(true));
        System.out.println("calculate threaded");
        nn.calculate();
        System.out.println(nn.getResult(true));*/
    }
}