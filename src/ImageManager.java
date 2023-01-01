import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class ImageManager {
    public static ArrayList<GrayImage> getImages(String path) {
        ArrayList<GrayImage> images = new ArrayList<GrayImage>();

        // Read all image files in the given path
        File folder = new File(path);
        File[] imageFiles = folder.listFiles((dir, name) -> name.endsWith(".jpg") || name.endsWith(".png"));

        for (File imageFile : imageFiles) {
            try {
                // Read in the image file
                BufferedImage image = ImageIO.read(imageFile);

                // Convert the image to grayscale
                BufferedImage grayscaleImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
                grayscaleImage.getGraphics().drawImage(image, 0, 0, null);

                // Extract the pixel data from the grayscale image
                int[][] pixels = new int[grayscaleImage.getWidth()][grayscaleImage.getHeight()];
                for (int x = 0; x < grayscaleImage.getWidth(); x++) {
                    for (int y = 0; y < grayscaleImage.getHeight(); y++) {
                        pixels[x][y] = grayscaleImage.getRGB(x, y) & 0xff;
                    }
                }

                // Create a GrayImage object with the pixel data and add it to the list
                GrayImage grayImage = new GrayImage(pixels);
                images.add(grayImage);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return images;
    }
    public static void printImage(GrayImage image)
    {
        int[][] pixels = image.GetPixels();
        for (int[] pixel : pixels) {
            for (int grayValue : pixel) {
                // Set the text color based on the grayscale value of the pixel
                String colorCode = String.format("\033[38;2;%d;%d;%dm", grayValue, grayValue, grayValue);
                System.out.print(colorCode + "**" + "\033[0m");
            }
            System.out.println();
        }
    }
}
