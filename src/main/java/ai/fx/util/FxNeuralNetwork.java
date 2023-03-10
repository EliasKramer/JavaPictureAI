package ai.fx.util;

import javafx.scene.Group;

public class FxNeuralNetwork {
    private final NeuralNode[][] nodes;
    private final NeuralWeight[][][] weights;
    private final int boundX;
    private final int boundY;

    public FxNeuralNetwork(
            int inputSize,
            int outputSize,
            int numHiddenLayers,
            int sizeHiddenLayer,
            int boundX,
            int boundY) {
        nodes = new NeuralNode[numHiddenLayers + 2][];
        nodes[0] = new NeuralNode[inputSize];
        for (int i = 1; i < numHiddenLayers + 1; i++) {
            nodes[i] = new NeuralNode[sizeHiddenLayer];
        }
        nodes[numHiddenLayers + 1] = new NeuralNode[outputSize];

        this.boundX = boundX;
        this.boundY = boundY;

        initNodes();
        calculateNodePositions(boundX, boundY, 1, 3, -1);

        weights = new NeuralWeight[nodes.length - 1][][];
        initWeights();
    }

    private void initNodes() {
        for (int i = 0; i < nodes.length; i++) {
            for (int j = 0; j < nodes[i].length; j++) {
                nodes[i][j] = new NeuralNode();
            }
        }
    }

    public void calculateNodePositions(
            int availableWidth,
            int availableHeight,
            int distanceBetween,
            int distanceToBound,
            int maxRadius
    ) {
        int biggestLayerSize = getBiggestLayerSize();
        int radius = (availableHeight - (biggestLayerSize - 1) * 2 - distanceToBound * 2) / (biggestLayerSize * 2);
        if (maxRadius != -1 && radius > maxRadius)
            radius = maxRadius;

        int distanceBetweenHorizontal = (availableWidth - nodes.length * radius * 2 - distanceToBound * 2) / (nodes.length - 1);

        int currX = distanceToBound + radius;

        for (NeuralNode[] neuralNodes : nodes) {
            int currY = distanceToBound + radius;
            for (NeuralNode node : neuralNodes) {
                node.setX(currX);
                node.setY(currY);
                node.setRadius(radius);
                currY += distanceBetween + radius * 2;
            }
            currX += distanceBetweenHorizontal + radius * 2;
        }
    }

    private int getBiggestLayerSize() {
        int biggestLayerSize = 0;
        for (NeuralNode[] neuralNodes : nodes) {
            if (neuralNodes.length > biggestLayerSize)
                biggestLayerSize = neuralNodes.length;
        }
        return biggestLayerSize;
    }

    private void initWeights() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = new NeuralWeight[nodes[i + 1].length][nodes[i].length];

            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] = new NeuralWeight(nodes[i][k],nodes[i + 1][j],  0);
                }
            }
        }
    }


    public Group getNodesAsGroup() {
        Group group = new Group();
        for (NeuralNode[] layer : nodes) {
            for (NeuralNode n : layer) {
                if (!isNodeInBounds(n))
                    throw new IllegalArgumentException("Node is not in bounds");
                group.getChildren().add(n.getCircle());
                group.getChildren().add(n.getText());
                group.getChildren().add(n.getBiasText());
            }
        }
        for (NeuralWeight[][] layer : weights) {
            for (NeuralWeight[] n : layer) {
                for (NeuralWeight w : n) {
                    if (w != null) {
                        group.getChildren().add(w.getLine());
                    }
                }
            }
        }
        return group;
    }

    private boolean isNodeInBounds(NeuralNode node) {
        return isPositionInBounds(node.getX(), node.getY());
    }

    private boolean isPositionInBounds(int x, int y) {
        return x < boundX && y < boundY;
    }

    public void setActivationAt(int x, int y, double value) {
        if (x >= nodes.length || y >= nodes[x].length)
            return;

        nodes[x][y].setNumber(value);
    }

    public void setBiasAt(int x, int y, double value) {
        //x++ is because the first layer is the input layer
        x++;
        if(x >= nodes.length || y >= nodes[x].length)
            return;

        nodes[x][y].setBias(value);
    }

    public void setWeightAt(int x, int y, int z, double value) {
        if(x >= weights.length || y >= weights[x].length || z >= weights[x][y].length)
            return;
        weights[x][y][z].setWeight(value);
    }
}