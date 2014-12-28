import memobjects.MemObject;
import org.jocl.Sizeof;

import java.io.IOException;
import java.util.Date;

import static util.CLWrapper.readBuffer;

/**
 * Created by Алексей on 15.06.14.
 */
public class Main {

    public static void main(String args[]) {

        int n = 500;//21
        int v = 28 * 28; //32 * 32;
        int h1 = 200;
        int h2 = 70;
        int h3 = 30;
        int m = 10;//3
        int batch = n;
        float[] w_res = new float[(v + 1) * h1 + (h1 + 1) * h2 + (h2 + 1) * h3 + (h3 + 1) * m];

        long before = new Date().getTime();

        String sampleFile = "E:\\ms\\backprop\\src\\resourses\\rotate\\training";
        String sampleFileTest = "E:\\ms\\backprop\\src\\resourses\\rotate\\test";
        String validationFile = "E:\\ms\\backprop\\src\\resourses\\resized\\cristallograms\\tear\\validation";
        String labelsFile = "E:\\ms\\backprop\\src\\resourses\\mnist\\train-labels.idx1-ubyte";
        String imagesFile = "E:\\ms\\backprop\\src\\resourses\\mnist\\train-images.idx3-ubyte";

        MLPNet net1 = new MLPNet(v, h1, v);

        float[] x = new float[n * v];
        float[] xOut = new float[n * v];
        float[] t = new float[n * m];
        int[] dist = new int[m];

        int nTestTmp = 50;
        float[] xTestTmp = new float[(nTestTmp + n) * v];
        float[] tTestTmp = new float[(nTestTmp + n) * m];

        //DataLoader.readSampleImages(sampleFile, validationFile, x, xOut, t, dist, m);
        try {
            MNISTReader.readNumbersImages(labelsFile, imagesFile, xTestTmp, tTestTmp, nTestTmp + n);
            System.out.println("Read MNIST done. Number of used images: " + x.length / v);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.arraycopy(xTestTmp, 0, x, 0, n * v);
        System.arraycopy(tTestTmp, 0, t, 0, n * m);

        BackPropCL autoEncoder1 = new BackPropCL(net1, x, x, n, batch, 0.01f); //dist);
        autoEncoder1.init();

        System.out.println(autoEncoder1.train(10, 100));

        readBuffer(autoEncoder1.memObjects.get(MemObject.OUT), n * net1.getHiddenOutputSize() * Sizeof.cl_float, autoEncoder1.out);
        readBuffer(autoEncoder1.memObjects.get(MemObject.WEIGHTS), net1.getWeightsSize() * Sizeof.cl_float, autoEncoder1.net.weights);

        System.arraycopy(autoEncoder1.net.weights, 0, w_res, 0, (v + 1) * h1);

        float[] dest = new float[n * net1.getOutputSize()];
        System.arraycopy(autoEncoder1.out, n * net1.getHiddenSize(), dest, 0, n * net1.getOutputSize());

        autoEncoder1.release();

        float[] x2 = new float[n * net1.getHiddenSize()];
        System.arraycopy(autoEncoder1.out, 0, x2, 0, n * net1.getHiddenSize());

        MLPNet net2 = new MLPNet(h1, h2, h1);

        BackPropCL autoEncoder2 = new BackPropCL(net2, x2, x2, n, batch, 0.01f); //dist);
        autoEncoder2.init();

        System.out.println(autoEncoder2.train(10, 40000));

        readBuffer(autoEncoder2.memObjects.get(MemObject.OUT), n * net2.getHiddenOutputSize() * Sizeof.cl_float, autoEncoder2.out);
        readBuffer(autoEncoder2.memObjects.get(MemObject.WEIGHTS), net2.getWeightsSize() * Sizeof.cl_float, autoEncoder2.net.weights);

        System.arraycopy(autoEncoder2.net.weights, 0, w_res, (v + 1) * h1, (h1 + 1) * h2);

        float[] x3 = new float[n * net2.getHiddenSize()];
        System.arraycopy(autoEncoder2.out, 0, x3, 0, n * net2.getHiddenSize());

        autoEncoder2.release();

        MLPNet net3 = new MLPNet(h2, h3, m);

        BackPropCL backPropCL = new BackPropCL(net3, x3, t, n, batch, 0.3f); //dist);
        backPropCL.init();
        System.out.println(backPropCL.train(10, 40000));

        readBuffer(backPropCL.memObjects.get(MemObject.OUT), n * net3.getHiddenOutputSize() * Sizeof.cl_float, backPropCL.out);
        readBuffer(backPropCL.memObjects.get(MemObject.WEIGHTS), net3.getWeightsSize() * Sizeof.cl_float, backPropCL.net.weights);

        System.arraycopy(backPropCL.net.weights, 0, w_res, (v + 1) * h1 + (h1 + 1) * h2, (h2 + 1) * h3 + (h3 + 1) * m);

 /*       int errorCount = 0;

        for (int i = 0; i < n; i++) {
            float max = 0f;
            float out = 0f;
            int maxIndxOut = -1;
            int maxIndxSrc = -1;
            boolean isError = false;
            for (int j = 0; j < net3.layerSizes[2]; j++) {
                //System.out.print(backPropCL.out[n * net3.layerSizes[1] + i * net3.layerSizes[2] + j] + " ");
                out = backPropCL.out[n * net3.layerSizes[1] + i * net3.layerSizes[2] + j];
                if (max < out) {
                    max = out;
                    maxIndxOut = j;
                }
            }
            for (int j = 0; j < 10; j++) {
                out = t[i * 10 + j];
                if (max < out) {
                    max = out;
                    maxIndxSrc = j;
                }
            }
            if (maxIndxOut != maxIndxSrc) {
                isError = true;
                errorCount++;
            }
            System.out.println(i + ":  " + maxIndxOut + " (source is " + maxIndxSrc + ")"
                    + (isError ? "  ERROR" : ""));
        }*/

        backPropCL.release();

        MLPNet net4 = new MLPNet(w_res, v, h1, h2, h3, m);

        BackPropCL deepNet = new BackPropCL(net4, x, t, n, n, 0.3f); //dist);
        deepNet.init();
        System.out.println(deepNet.train(1, 100));
        readBuffer(deepNet.memObjects.get(MemObject.WEIGHTS), net4.getWeightsSize() * Sizeof.cl_float, deepNet.net.weights);
        deepNet.release();

        //Test net
        int nTest = 50;
        float[] xTest = new float[nTest * v];
        float[] tTest = new float[nTest * m];

        System.arraycopy(xTestTmp, n * v, xTest, 0, nTest * v);
        System.arraycopy(tTestTmp, n * m, tTest, 0, nTest * m);

        DataLoader.writeSampleImages(xTest, nTest, "E:\\ms\\backprop\\src\\resourses\\out1");

        //DataLoader.readSampleImages(sampleFileTest, xTest, tTest, dist, m);

        MLPNet net5 = new MLPNet(deepNet.net.weights, v, h1, h2, h3, m);

        BackPropCL testNet = new BackPropCL(net5, xTest, tTest, nTest, nTest, 0.3f); //dist);
        testNet.init();
        System.out.println(testNet.train(1, 1));
        readBuffer(testNet.memObjects.get(MemObject.OUT), nTest * net5.getHiddenOutputSize() * Sizeof.cl_float, testNet.out);

       // System.out.println("Total error count: " + errorCount + " from " + n + " images");

        /*float[] dest = new float[n * net.getOutputSize()];
        System.arraycopy(backPropCL.out, n * net.getHiddenSize(), dest, 0, n * net.getOutputSize());
        backPropCL.writeSampleImages(dest, "E:\\ms\\backprop\\src\\resourses\\out");*/

        /*System.out.println("Result: ");
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < net.layerSizes[1]; j++)
            {
                System.out.print(backPropCL.out[i * net.layerSizes[1] + j] + " ");
            }
            System.out.println();
        }*/
        System.out.println();
        boolean error = false;
        for (int i = 0; i < nTest; i++)
        {
            error = false;
            for (int j = 0; j < net4.layerSizes[4]; j++)
            {
                float tmp = tTest[i * net4.layerSizes[4] + j] - testNet.out[nTest * (net4.layerSizes[1] + net4.layerSizes[2] + net4.layerSizes[3])
                        + i * net4.layerSizes[4] + j];
                System.out.print(tmp + " ");
                if (tmp > 0.5f)
                {
                    error = true;
                }
            }
            if (error)
                System.out.print(" ERROR!!!");
            System.out.println();
        }
        System.out.println("Training time = " + (new Date().getTime() - before) / 1000.0 + " sec");
        /*for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < net3.layerSizes[3]; j++)
            {
                System.out.print(backPropCL.out[n * (net3.layerSizes[1] + net3.layerSizes[2]) + i * net3.layerSizes[3] + j] + " ");
            }
            System.out.println();
        }*/
        /*System.out.println("w: ");
        for (int i = 0; i < net.getWeightsSize(); i++)
        {
            System.out.print(backPropCL.net.weights[i] + " ");
        }*/


    }
}
