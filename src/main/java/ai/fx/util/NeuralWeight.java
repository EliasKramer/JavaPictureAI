package ai.fx.util;

import javafx.scene.Node;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;

public class NeuralWeight {
    private final NeuralNode from;
    private final NeuralNode to;
    private double weight;
    private Rectangle rectangleLine;
    public NeuralWeight(NeuralNode from, NeuralNode to, float weight) {
        this.from = from;
        this.to = to;
        this.weight = weight;
        initFx();
    }
    private void initFx()
    {
        rectangleLine = new Rectangle();
        updateLine();
    }

    private void updateLine() {
        double x1 = from.getX() + from.getRadius();
        double y1 = from.getY();
        double x2 = to.getX() - to.getRadius();
        double y2 = to.getY();

        double length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
        double midX = (x1 + x2) / 2;
        double midY = (y1 + y2) / 2;

        rectangleLine.setX(midX - length / 2);
        rectangleLine.setY(midY - 5);
        rectangleLine.setWidth(length);

        rectangleLine.setHeight(Math.abs(weight*1.5));
        rectangleLine.setFill(weight == 0 ? Color.BLACK : weight > 0 ? Color.BLUE : Color.RED);

        rectangleLine.setRotate(Math.toDegrees(Math.atan2(y2 - y1, x2 - x1)));
    }
    public Node getLine() {
        updateLine();
        return rectangleLine;
    }
    public NeuralNode getFrom() {
        return from;
    }

    public NeuralNode getTo() {
        return to;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
        updateLine();
    }
}
