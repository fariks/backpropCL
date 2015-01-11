
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTReader {

    public static void readNumbersImages(String labelsFile, String imagesFile,
                                         float[] x, float[] t, int n) {
        try {
            DataInputStream labels = new DataInputStream(
                    new FileInputStream(labelsFile));
            DataInputStream images = new DataInputStream(
                    new FileInputStream(imagesFile));
            int magicNumber = labels.readInt();
            if (magicNumber != 2049) {
                System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
                System.exit(0);
            }
            magicNumber = images.readInt();
            if (magicNumber != 2051) {
                System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
                System.exit(0);
            }
            int numLabels = labels.readInt();
            int numImages = images.readInt();
            int numRows = images.readInt();
            int numCols = images.readInt();
            if (numLabels != numImages) {
                System.err.println("Image file and label file do not contain the same number of entries.");
                System.err.println("  Label file contains: " + numLabels);
                System.err.println("  Image file contains: " + numImages);
                System.exit(0);
            }
            System.out.println("numImages: " + numImages);
            System.out.println("numRows: " + numRows + "  numCols: " + numCols);
            int numLabelsRead = 0;
            int numImagesRead = 0;
            while (labels.available() > 0 && numLabelsRead < numLabels && numImagesRead < n) {
                byte label = labels.readByte();
                for (byte i = 0; i < 10; i++) {
                    t[numLabelsRead * 10 + i] = (label == i ? 1f : 0f);
                }
                numLabelsRead++;

                int[][] image = new int[numCols][numRows];
                for (int colIdx = 0; colIdx < numCols; colIdx++) {
                    for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                        image[colIdx][rowIdx] = images.readUnsignedByte();
                        x[numImagesRead * numRows * numCols + numRows * colIdx + rowIdx] = image[colIdx][rowIdx] / 255.0f;
                    }
                }
                numImagesRead++;
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}