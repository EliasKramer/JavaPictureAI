module elias.kramer.ai.demo {
    requires javafx.controls;
    requires javafx.fxml;

    exports ai.fx.util;
    opens ai.fx.util to javafx.fxml;
    exports ai.fx.main;
    opens ai.fx.main to javafx.fxml;
}