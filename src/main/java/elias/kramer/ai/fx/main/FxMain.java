package elias.kramer.ai.fx.main;

import elias.kramer.ai.fx.util.FxNeuralNetwork;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.stage.Stage;
public class FxMain extends Application {

    @Override
    public void start(Stage stage) {
        FxNeuralNetwork network = new FxNeuralNetwork(5, 2, 16, 10, 1000, 600);

        Group root = network.getNodesAsGroup();

        Scene scene = new Scene(root, 1000, 600);
        stage.setTitle("Hello!");
        stage.setScene(scene);
        stage.show();

        network.setActivationAt(0, 0, 0.5f);
        network.setActivationAt(0, 1, 0.5f);

        network.setBiasAt(1, 0, 0.5f);
        network.setWeightAt(0, 0, 1, 100f);
    }

    public static void main(String[] args) {
        launch();
    }
}