package ai.reader;

import ai.neural_network.AiData;

import java.io.*;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class ImageManager {
    public static List<AiData> getImages(String dataFilePath, String labelFilePath) {
        try {
            DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
            int magicNumber = dataInputStream.readInt();
            int numberOfItems = dataInputStream.readInt();
            int nRows = dataInputStream.readInt();
            int nCols = dataInputStream.readInt();
            /*
            System.out.println("magic number is " + magicNumber);
            System.out.println("number of items is " + numberOfItems);
            System.out.println("number of rows is: " + nRows);
            System.out.println("number of cols is: " + nCols);
            */
            DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
            int labelMagicNumber = labelInputStream.readInt();
            int numberOfLabels = labelInputStream.readInt();

            //System.out.println("labels magic number is: " + labelMagicNumber);
            //System.out.println("number of labels is: " + numberOfLabels);

            AiData[] data = new MnistMatrix[numberOfItems];

            assert numberOfItems == numberOfLabels;

            for (int i = 0; i < numberOfItems; i++) {
                MnistMatrix mnistMatrix = new MnistMatrix(nRows, nCols);

                mnistMatrix.setLabel(String.valueOf(labelInputStream.readUnsignedByte()));
                for (int r = 0; r < nRows; r++) {
                    for (int c = 0; c < nCols; c++) {
                        mnistMatrix.setValue(r, c, dataInputStream.readUnsignedByte());
                    }
                }
                data[i] = mnistMatrix;
            }
            dataInputStream.close();
            labelInputStream.close();
            return Arrays.stream(data).toList();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new LinkedList<>();
    }

    public static void printImages(List<MnistMatrix> images) {
        for (MnistMatrix image : images) {
            printImage(image);
        }
    }

    public static void printImage(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());

        for (int r = 0; r < matrix.getNumberOfRows(); r++) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                //print values in the color of their grayscale value. 0 is black, 255 is white
                System.out.print("\u001B[48;2;" +
                        (int) matrix.getValue(r, c) + ";" +
                        (int) matrix.getValue(r, c) + ";" +
                        (int) matrix.getValue(r, c) +
                        "m  \u001B[0m");
            }
            System.out.println();
        }
    }
}
