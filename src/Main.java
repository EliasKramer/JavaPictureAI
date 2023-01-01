import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");

        ArrayList<GrayImage> images = ImageManager.getImages("data/digit_0");
        ImageManager.printImage(images.get(0));
        NeuralNetwork nn = new NeuralNetwork(Constants.INPUT_SIZE, 10, 2, 16);
        nn.randomiseWeightsAndBiases();
        nn.setInputs(images.get(0));
        nn.calculate();

        System.out.println("Done!");
        System.out.println("The result is " + nn.getResult(true));
        System.out.println("The cost is " + nn.getCost(0));
    }
}