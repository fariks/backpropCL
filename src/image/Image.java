package image;

import com.sun.media.jai.codec.FileSeekableStream;
import com.sun.media.jai.codec.ImageCodec;
import com.sun.media.jai.codec.ImageDecoder;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.RenderedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * @author: Алексей
 * Date: 20.10.13
 * Time: 19:34
 */

public class Image {

    private float[] source;

    private int height;

    private int width;

    /**
     * Read image by file name
     */
    public Image(String fileName) {
        try {
            String extension = "";

            int i = fileName.lastIndexOf('.');
            if (i > 0) {
                extension = fileName.substring(i + 1);
            }
            if ("tiff".equals(extension)) {
                File file = new File(fileName);
                ImageDecoder dec = ImageCodec.createImageDecoder("tiff", new FileSeekableStream(file), null);
                RenderedImage sourceImage = dec.decodeAsRenderedImage();
                width = sourceImage.getWidth();
                height = sourceImage.getHeight();
                source = new float[height * width];
                Raster srcRaster = sourceImage.getData();
                srcRaster.getSamples(0, 0, width, height, 0, source);
            } else if ("jpg".equals(extension)) {
                BufferedImage image = ImageIO.read(new FileInputStream(fileName));
                //BufferedImage grayScaleImage = convertColorImageToGrayScale(image);
                width = image.getWidth();
                height = image.getHeight();
                source = new float[height * width];
                Raster srcRaster = image.getData();
                srcRaster.getSamples(0, 0, width, height, 0, source);
            } else {
                BufferedImage image = ImageIO.read(new FileInputStream(fileName));
                width = image.getWidth();
                height = image.getHeight();
                source = new float[height * width];
                Raster srcRaster = image.getData();
                srcRaster.getSamples(0, 0, width, height, 0, source);
            }
        } catch (IOException ioe) {
            System.out.println(ioe);
        }
    }

    private BufferedImage convertColorImageToGrayScale(BufferedImage image) {
        BufferedImage destImage = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
        WritableRaster destRaster = destImage.getRaster();
        Raster srcRaster = image.getData();
        int width = srcRaster.getWidth();
        int height = srcRaster.getHeight();
        double[] red = new double[height * width];
        double[] green = new double[height * width];
        double[] blue = new double[height * width];
        double[] result = new double[height * width];
        srcRaster.getSamples(0, 0, width, height, 0, red);
        srcRaster.getSamples(0, 0, width, height, 1, green);
        srcRaster.getSamples(0, 0, width, height, 2, blue);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result[i * width + j] = (red[i * width + j] + green[i * width + j] + blue[i * width + j]) / 3;
            }
        }
        destRaster.setSamples(0, 0, width, height, 0, result);
        destRaster.setSamples(0, 0, width, height, 1, result);
        destRaster.setSamples(0, 0, width, height, 2, result);
        return destImage;
    }

    /*public static void main(String... args) throws Exception {
        BufferedImage image = ImageIO.read(new FileInputStream("C:\\Users\\Алексей\\Desktop\\ms\\src\\resources\\cristallograms\\saliva\\1 40.jpg"));
        ImageIO.write(convertColorImageToGrayScale(image), "jpg", new FileOutputStream("C:\\Users\\Алексей\\Desktop\\ms\\src\\resources\\test.jpg"));
    }*/

    /**
     * Read image by source matrix
     */
    public Image(float[] source, int height, int width) {
        this.source = source;
        this.height = height;
        this.width = width;
    }

    public float[] getSource() {
        return source;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }
}
