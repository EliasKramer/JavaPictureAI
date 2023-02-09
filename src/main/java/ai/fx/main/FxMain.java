package ai.fx.main;

import ai.fx.util.FxNeuralNetwork;
import ai.fx.util.NeuralNode;
import ai.neural_network.NeuralNetwork;
import ai.reader.ImageManager;
import ai.reader.MnistMatrix;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.util.List;

public class FxMain extends Application {
    private int dimX = 1000;
    private int dimY = 600;
    NeuralNetwork network;
    @Override
    public void start(Stage stage) {
        network = new NeuralNetwork(28*28, 10, 2, 16, new FxNeuralNetwork(20, 10, 2, 16, dimX, dimY));

        Group root = network.getFx().getNodesAsGroup();

        Scene scene = new Scene(root, dimX, dimY);
        stage.setTitle("Hello!");
        stage.setScene(scene);
        stage.show();
        afterFxStart();
    }

    public static void main(String[] args) {
        launch();
    }

    public void afterFxStart()
    {
        System.out.println("FX started");

        //get ai images
        List<MnistMatrix> trainingImages = ImageManager.getImages(
                "data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte");

        network.setInputs(trainingImages.get(0));
        network.randomiseWeightsAndBiases();
        network.feedForward();
        network.updateFx();
    }
}