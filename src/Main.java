import memobjects.MemObject;
import org.jocl.Sizeof;

import java.util.Date;

import static util.CLWrapper.readBuffer;

/**
 * Created by Алексей on 15.06.14.
 */
public class Main {

    public static void main(String args[]) {

        int n = 1000;//21
        int v = 28 * 28; //32 * 32;
        int h1 = 196;
        int h2 = 32;
        int h3 = 20;
        int m = 10;//3
        int batch = 50;
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

        int nTest = 500;
        int nVal = 50;
        float[] xVal = new float[nVal * v];
        float[] tVal = new float[nVal * m];

        float[] xTestTmp = new float[(nTest + nVal + n) * v];
        float[] tTestTmp = new float[(nTest + nVal + n) * m];

        //DataLoader.readSampleImages(sampleFile, validationFile, x, xOut, t, dist, m);
        MNISTReader.readNumbersImages(labelsFile, imagesFile, xTestTmp, tTestTmp, nTest + n);
        System.out.println("Read MNIST done. Number of used images: " + x.length / v);

        System.arraycopy(xTestTmp, 0, x, 0, n * v);
        System.arraycopy(tTestTmp, 0, t, 0, n * m);

        System.arraycopy(xTestTmp, n * v, xVal, 0, nVal * v);
        System.arraycopy(tTestTmp, n * m, tVal, 0, nVal * m);

        BackPropCL autoEncoder1 = new BackPropCLVal(net1, x, x, n, batch, 0.01f, xVal, xVal, nVal);
        autoEncoder1.init();

        System.out.println(autoEncoder1.train(10, 60000));

        readBuffer(autoEncoder1.memObjects.get(MemObject.OUT), n * net1.getHiddenOutputSize() * Sizeof.cl_float, autoEncoder1.out);
        readBuffer(autoEncoder1.memObjects.get(MemObject.WEIGHTS), net1.getWeightsSize() * Sizeof.cl_float, autoEncoder1.net.weights);

        System.arraycopy(autoEncoder1.net.weights, 0, w_res, 0, (v + 1) * h1);

        autoEncoder1.release();

        float[] x2 = new float[n * net1.getHiddenSize()];
        System.arraycopy(autoEncoder1.out, 0, x2, 0, n * net1.getHiddenSize());

        MLPNet net2 = new MLPNet(h1, h2, h1);

        BackPropCL autoEncoder2 = new BackPropCL(net2, x2, x2, n, batch, 0.01f); //dist);
        autoEncoder2.init();

        System.out.println(autoEncoder2.train(20, 100000));

        readBuffer(autoEncoder2.memObjects.get(MemObject.OUT), n * net2.getHiddenOutputSize() * Sizeof.cl_float, autoEncoder2.out);
        readBuffer(autoEncoder2.memObjects.get(MemObject.WEIGHTS), net2.getWeightsSize() * Sizeof.cl_float, autoEncoder2.net.weights);

        System.arraycopy(autoEncoder2.net.weights, 0, w_res, (v + 1) * h1, (h1 + 1) * h2);

        float[] x3 = new float[n * net2.getHiddenSize()];
        System.arraycopy(autoEncoder2.out, 0, x3, 0, n * net2.getHiddenSize());

        autoEncoder2.release();

        MLPNet net3 = new MLPNet(h2, h3, m);

        BackPropCL backPropCL = new BackPropCL(net3, x3, t, n, batch, 0.3f); //dist);
        backPropCL.init();
        System.out.println(backPropCL.train(20, 100000));

        readBuffer(backPropCL.memObjects.get(MemObject.OUT), n * net3.getHiddenOutputSize() * Sizeof.cl_float, backPropCL.out);
        readBuffer(backPropCL.memObjects.get(MemObject.WEIGHTS), net3.getWeightsSize() * Sizeof.cl_float, backPropCL.net.weights);

        System.arraycopy(backPropCL.net.weights, 0, w_res, (v + 1) * h1 + (h1 + 1) * h2, (h2 + 1) * h3 + (h3 + 1) * m);

        backPropCL.release();

        MLPNet net4 = new MLPNet(w_res, v, h1, h2, h3, m);

        BackPropCL deepNet = new BackPropCL(net4, x, t, n, batch, 0.3f); //dist);
        deepNet.init();
        System.out.println(deepNet.train(10, 100));
        readBuffer(deepNet.memObjects.get(MemObject.WEIGHTS), net4.getWeightsSize() * Sizeof.cl_float, deepNet.net.weights);
        deepNet.release();

        //Test net
        float[] xTest = new float[nTest * v];
        float[] tTest = new float[nTest * m];

        System.arraycopy(xTestTmp, (n + nVal) * v, xTest, 0, nTest * v);
        System.arraycopy(tTestTmp, (n + nVal) * m, tTest, 0, nTest * m);

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
        int numberOfError = 0;
        boolean error = false;
        for (int i = 0; i < nTest; i++)
        {
            error = false;
            for (int j = 0; j < m; j++)
            {
                float tmp = tTest[i * m + j] - testNet.out[nTest * (net4.layerSizes[1] + net4.layerSizes[2] + net4.layerSizes[3])
                        + i * m + j];
                System.out.print(tmp + " ");
                if (tmp > 0.5f)
                {
                    error = true;
                    numberOfError++;
                }
            }
            if (error) System.out.println(" ERROR!!!");
            else System.out.println();
        }
        System.out.println("\nnumberOfError = " + numberOfError);
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
