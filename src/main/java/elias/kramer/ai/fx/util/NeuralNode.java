package elias.kramer.ai.fx.util;

import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.text.Text;

public class NeuralNode {
    private int x;
    private int y;
    private int radius;
    private float number;
    private float bias;
    private Color circleColor;
    private Color textColor;
    private Text text;
    private Text biasText;
    private Circle circle;
    public NeuralNode(int x, int y, int radius, float number, Color color, Color textColor, float bias) {
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
    public float getNumber() {
        return number;
    }
    public void setNumber(float number) {
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
    public float getBias() {
        return bias;
    }
    public void setBias(float bias) {
        this.bias = bias;
        updateText();
        updateTextColor();
    }

    //update
    private void updateTextColor() {
        text.setFill(textColor);
        biasText.setFill(bias > 0 ? Color.GREEN : Color.RED);
    }

    private void updateCircleColor() {
        circle.setFill(circleColor);
    }
    private void updatePosition()
    {
        circle.setCenterX(x);
        circle.setCenterY(y);

        circle.setRadius(radius);
        //font-size of text
        text.setStyle("-fx-font-size: "+ radius/2 +"px;");
        biasText.setStyle("-fx-font-size: "+ radius/2 +"px;");
        //center text
        text.setX(x - text.getLayoutBounds().getWidth() / 6);
        biasText.setX(x - biasText.getLayoutBounds().getWidth() / 6);

        text.setY(y - radius/3);
        biasText.setY(y + radius/3);
    }
    private void updateText()
    {
        text.setText(String.valueOf(number));
        biasText.setText(String.valueOf(bias));
    }
}
