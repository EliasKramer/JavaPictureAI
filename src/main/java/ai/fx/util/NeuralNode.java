package ai.fx.util;

import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.text.Text;

import java.text.DecimalFormat;

public class NeuralNode {
    private int x;
    private int y;
    private int radius;
    private double number;
    private double bias;
    private Color circleColor;
    private Color textColor;
    private Text text;
    private Text biasText;
    private Circle circle;
    public NeuralNode(int x, int y, int radius, double number, Color color, Color textColor, double bias) {
        this.x = x;
        this.y = y;
        this.radius = radius;
        this.number = number;
        this.circleColor = color;
        this.textColor = textColor;
        this.bias = bias;
        initFx();
    }
    public NeuralNode()
    {
        this(0,0,0,0,Color.DARKGRAY, Color.WHITE, 0);
    }
    public void initFx()
    {
        circle = new Circle();
        //convert node number to text
        text = new Text();
        biasText = new Text();
        //set text white;

        updateText();
        updatePosition();
        updateCircleColor();
        updateTextColor();
    }
    //setter getter
    public Circle getCircle() {
        return circle;
    }
    public Text getText() {
        return text;
    }
    public Text getBiasText() {
        return biasText;
    }
    public int getX() {
        return x;
    }
    public void setX(int x) {
        this.x = x;
        updatePosition();
    }
    public int getY() {
        return y;
    }
    public void setY(int y) {
        this.y = y;
        updatePosition();
    }
    public double getNumber() {
        return number;
    }
    public void setNumber(double number) {
        this.number = number;
        updateText();
    }
    public int getRadius() {
        return radius;
    }
    public void setRadius(int nodeRadius) {
        this.radius = nodeRadius;
        updatePosition();
    }
    public Color getTextColor() {
        return textColor;
    }
    public void setTextColor(Color textColor) {
        this.textColor = textColor;
        updateTextColor();
    }
    public Color getCircleColor() {
        return circleColor;
    }
    public void setCircleColor(Color circleColor) {
        this.circleColor = circleColor;
        updateCircleColor();
    }
    public double getBias() {
        return bias;
    }
    public void setBias(double bias) {
        this.bias = bias;
        updateText();
        updateTextColor();
    }

    //update
    private void updateTextColor() {
        text.setFill(textColor);
        biasText.setFill(bias > 0 ? Color.BLUE : Color.RED);
    }

    private void updateCircleColor() {
        circle.setFill(circleColor);
    }
    private void updatePosition() {
        circle.setCenterX(x);
        circle.setCenterY(y);

        circle.setRadius(Math.round(radius * 100.0) / 100.0);

        // font-size of text
        int fontSize = (int) Math.round(radius / 2);
        text.setStyle("-fx-font-size: " + fontSize + "px;");
        biasText.setStyle("-fx-font-size: " + fontSize + "px;");

        // center text
        double textX = x - text.getLayoutBounds().getWidth() / 2;
        double biasTextX = x - biasText.getLayoutBounds().getWidth() / 2;
        text.setX(Math.round(textX * 100.0) / 100.0);
        biasText.setX(Math.round(biasTextX * 100.0) / 100.0);

        double textY = y - fontSize / 3;
        double biasTextY = y + fontSize / 1.5;  // increase the offset
        text.setY(Math.round(textY * 100.0) / 100.0);
        biasText.setY(Math.round(biasTextY * 100.0) / 100.0);
    }


    private void updateText()
    {
        DecimalFormat df = new DecimalFormat("#.##");
        String formattedNumber = df.format(number);
        String formattedBias = df.format(bias);
        text.setText(formattedNumber);
        biasText.setText(formattedBias);
    }
}
