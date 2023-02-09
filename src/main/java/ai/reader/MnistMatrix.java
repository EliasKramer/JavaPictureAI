package ai.reader;

import ai.neural_network.AiData;

public class MnistMatrix implements AiData {
    private double [] data;
    private int rowCount;
    private int colCount;
    private String label;

    public MnistMatrix(int givenRows, int givenCols) {
        this.rowCount = givenRows;
        this.colCount = givenCols;

        data = new double[rowCount * colCount];
    }

    public double getValue(int r, int c) {
        return data[getIdx(r,c)];
    }

    public void setValue(int row, int col, double value) {
        data[getIdx(row,col)] = value;
    }


    @Override
    public int getSizeOfInputs() {
        return rowCount * colCount;
    }

    @Override
    public double[] getInputs() {
        return data;
    }

    @Override
    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public int getNumberOfRows() {
        return rowCount;
    }

    public int getNumberOfColumns() {
        return colCount;
    }
    public int getIdx(int r, int c)
    {
        return r * colCount + c;
    }

}
