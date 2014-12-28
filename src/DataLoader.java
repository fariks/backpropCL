import image.Image;
import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.color.ColorSpace;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.*;
import java.util.Scanner;

/**
 * Created by Алексей on 15.06.14.
 */
public class DataLoader {

    public static void readSample(String fileName, float[] x, float[] t, int n, int v, int m)
    {
        try {
            Scanner in = new Scanner(new FileInputStream(fileName));
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < v; j++)
                {
                    x[i * v + j] = Float.parseFloat(in.next());
                }
                int classLabel = Integer.parseInt(in.next());
                for (int j = 0; j < m; j++)
                {
                    if (classLabel == j)
                    {
                        t[i * m + j] = 1f;
                    }
                    else {
                        t[i * m + j] = 0f;
                    }
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }


    public static void readSampleImages(String dirName, float[] x, float[] t, int[] dist, int m)
    {
        try {
            File dir = new File(dirName);
            File[] subDirs = dir.listFiles();
            int offset = 0;
            int classCount = m;
            for (int k = 0; k < classCount; k++)
            {
                File[] files = subDirs[k].listFiles(new FileFilter() {
                    private final FileNameExtensionFilter filter =
                            new FileNameExtensionFilter("Image files",
                                    "tiff", "jpg", "BMP");
                    public boolean accept(File file) {
                        return filter.accept(file);
                    }
                });
                dist[k] = files.length;
                for (int i = 0; i < files.length; i++)
                {
                    float[] tmp = new Image(files[i].getAbsolutePath()).getSource();
                    for (int j = 0; j < tmp.length; j++)
                    {
                        x[offset * tmp.length + i * tmp.length + j] = tmp[j] / 255.0f;
                    }
                    for (int j = 0; j < classCount; j++)
                    {
                        t[offset * classCount + i * classCount + j] = j == k ? 1f : 0f;
                    }
                }
                offset += files.length;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void readSampleImages(String dirName, String outDirName, float[] x, float[] xOut, float[] t, int[] dist, int m)
    {
        try {
            File dir = new File(dirName);
            File[] subDirs = dir.listFiles();
            File outDir = new File(outDirName);
            File[] outSubDirs = dir.listFiles();
            int offset = 0;
            int classCount = m;
            for (int k = 0; k < classCount; k++)
            {
                System.out.println(subDirs[k].getName());
                File[] files = subDirs[k].listFiles(new FileFilter() {
                    private final FileNameExtensionFilter filter =
                            new FileNameExtensionFilter("Image files",
                                    "tiff", "jpg", "BMP");
                    public boolean accept(File file) {
                        return filter.accept(file);
                    }
                });
                File[] outFiles = outSubDirs[k].listFiles(new FileFilter() {
                    private final FileNameExtensionFilter filter =
                            new FileNameExtensionFilter("Image files",
                                    "tiff", "jpg", "BMP");
                    public boolean accept(File file) {
                        return filter.accept(file);
                    }
                });
                dist[k] = files.length;
                for (int i = 0; i < files.length; i++)
                {
                    float[] tmp = new Image(files[i].getAbsolutePath()).getSource();
                    float[] outTmp = new Image(outFiles[i / 3].getAbsolutePath()).getSource();
                    for (int j = 0; j < tmp.length; j++)
                    {
                        x[offset * tmp.length + i * tmp.length + j] = tmp[j] / 255.0f;
                        xOut[offset * tmp.length + i * tmp.length + j] = outTmp[j] / 255.0f;
                    }
                    for (int j = 0; j < classCount; j++)
                    {
                        t[offset * classCount + i * classCount + j] = j == k ? 1f : 0f;
                    }
                }
                offset += files.length;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void writeSampleImages(float[] out, int n, String outputDir)
    {
        try {
            int iWeight = 28;
            int iHeight = 28;
            BufferedImage destImage = new BufferedImage(iWeight, iHeight, ColorSpace.TYPE_RGB);
            WritableRaster destRaster = destImage.getRaster();
            int imageSize = out.length / n;
            float[] temp = new float[imageSize];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < imageSize; j++)
                {
                    temp[j] = out[i * imageSize + j] * 255f;
                }
                destRaster.setSamples(0, 0, iWeight, iHeight, 0, temp);
                destRaster.setSamples(0, 0, iWeight, iHeight, 1, temp);
                destRaster.setSamples(0, 0, iWeight, iHeight, 2, temp);
                ImageIO.write(destImage, "png", new FileOutputStream(outputDir + "\\out" + i + ".png"));
                temp = new float[imageSize];
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void resizeImages(String inputFile, String outputDir)
    {
        for (File sourceImageFile : new File(inputFile).listFiles()) {
            resizeImage(sourceImageFile, outputDir);
        }
    }

    private static void resizeImage(File inputFile, String outputDir)
    {
        try {
            BufferedImage img = ImageIO.read(inputFile);
            BufferedImage outImage = Scalr.resize(img, 64, 64);
            int crop = Math.min(outImage.getHeight(), outImage.getWidth());
            outImage = Scalr.crop(outImage, 42, 42);

            ImageIO.write(outImage, "jpg", new File(outputDir + "\\" + inputFile.getName()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]) throws IOException{
        /*DataLoader.resizeImages("E:\\ms\\ms\\ms\\src\\resources\\cristallograms\\tear",
                "E:\\ms\\backprop\\src\\resourses\\resized\\cristallograms\\tear");*/
        /*BufferedImage image = readImage(new File("E:\\ms\\backprop\\src\\resourses\\resized\\cristallograms\\tear\\test\\eight\\8.BMP"));
        image = rotateImage(image, 90);
        writeImage(image, new File("E:\\ms\\backprop\\src\\resourses\\out\\test.jpg"));*/

        File dir = new File("E:\\ms\\backprop\\src\\resourses\\resized\\cristallograms\\tear\\validation");
        File[] subDirs = dir.listFiles();
        for (File subDir : subDirs)
        {
            File[] files = subDir.listFiles(new FileFilter() {
                private final FileNameExtensionFilter filter =
                        new FileNameExtensionFilter("Image files",
                                "tiff", "jpg", "BMP");
                public boolean accept(File file) {
                    return filter.accept(file);
                }
            });
            for (File file : files)
            {
                BufferedImage image = readImage(file);
                BufferedImage image90 = rotateImage(image, 90);
                BufferedImage image180 = rotateImage(image, 180);
                BufferedImage image270 = rotateImage(image, 270);
                String path = file.getAbsolutePath();
                File dirTmp = new File(path.substring(0, path.indexOf("validation")) + "training\\" + file.getParentFile().getName());
                if (!dirTmp.exists())
                {
                    dirTmp.mkdir();
                }
                writeImage(image90, new File(dirTmp.getAbsolutePath() + "\\" +
                        file.getName().substring(0, file.getName().indexOf(".")) + "_90.jpg"));
                writeImage(image180, new File(dirTmp.getAbsolutePath() + "\\" +
                        file.getName().substring(0, file.getName().indexOf(".")) + "_180.jpg"));
                writeImage(image270, new File(dirTmp.getAbsolutePath() + "\\" +
                        file.getName().substring(0, file.getName().indexOf(".")) + "_270.jpg"));
            }
        }
    }

    private static BufferedImage readImage(File file) throws IOException
    {
        return ImageIO.read(file);
    }

    private static void writeImage(BufferedImage image, File file) throws IOException
    {
        ImageIO.write(image, "jpg", file);
    }

    private static BufferedImage rotateImage(BufferedImage image, int angle)
    {
        double rotationRequired = Math.toRadians(angle);
        double locationX = image.getWidth() / 2;
        double locationY = image.getHeight() / 2;
        AffineTransform tx = AffineTransform.getRotateInstance(rotationRequired, locationX, locationY);
        AffineTransformOp op = new AffineTransformOp(tx, AffineTransformOp.TYPE_BILINEAR);

        return op.filter(image, null);
    }

}
