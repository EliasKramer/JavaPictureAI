module elias.kramer.ai.demo {
    requires javafx.controls;
    requires javafx.fxml;

    exports elias.kramer.ai.fx.util;
    opens elias.kramer.ai.fx.util to javafx.fxml;
    exports elias.kramer.ai.fx.main;
    opens elias.kramer.ai.fx.main to javafx.fxml;
}