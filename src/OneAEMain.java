import memobjects.MemObject;
import org.jocl.Sizeof;

import static util.CLWrapper.readBuffer;

public class OneAEMain {

    public static void main(String[] args) {
        int n = 1000;
        int nTest = 1000;
        int nVal = 50;
        int batch = 50;
        int m = 10;

        int v = 784; //28 * 28;
        int h1 = 196;
        int h2 = 100;
        float[] w_res = new float[(v + 1) * h1 + (h1 + 1) * h2 + (h2 + 1) * m];

        float[] x = new float[n * v];
        float[] t = new float[n * m];

        float[] xVal = new float[nVal * v];
        float[] tVal = new float[nVal * m];

        float[] xTest = new float[nTest * v];
        float[] tTest = new float[nTest * m];

        float[] xAll = new float[(nTest + nVal + n) * v];
        float[] tAll = new float[(nTest + nVal + n) * m];

        String labelsFile = "E:\\ms\\backprop\\src\\resourses\\mnist\\train-labels.idx1-ubyte";
        String imagesFile = "E:\\ms\\backprop\\src\\resourses\\mnist\\train-images.idx3-ubyte";
        MNISTReader.readNumbersImages(labelsFile, imagesFile, xAll, tAll, nTest + n);
        System.out.println("Read MNIST done. Number of used images: " + x.length / v);

        System.arraycopy(xAll, 0, x, 0, n * v);
        System.arraycopy(tAll, 0, t, 0, n * m);
        System.arraycopy(xAll, n * v, xVal, 0, nVal * v);
        System.arraycopy(tAll, n * m, tVal, 0, nVal * m);
        System.arraycopy(xAll, (n + nVal) * v, xTest, 0, nTest * v);
        System.arraycopy(tAll, (n + nVal) * m, tTest, 0, nTest * m);

        xAll = null;
        tAll = null;

        MLPNet aeNet = new MLPNet(v, h1, v);
        BackPropCL aeNetTrainer = new BackPropCLVal(aeNet, x, x, n, batch, 0.01f, xVal, xVal, nVal);
        aeNetTrainer.init();
        System.out.println(aeNetTrainer.train(10, 20000));
        readBuffer(aeNetTrainer.memObjects.get(MemObject.OUT), n * aeNet.getHiddenOutputSize() * Sizeof.cl_float, aeNetTrainer.out);
        readBuffer(aeNetTrainer.memObjects.get(MemObject.WEIGHTS), aeNet.getWeightsSize() * Sizeof.cl_float, aeNetTrainer.net.weights);
        System.arraycopy(aeNetTrainer.net.weights, 0, w_res, 0, (v + 1) * h1);
        float[] x2 = new float[n * aeNet.getHiddenSize()];
        System.arraycopy(aeNetTrainer.out, 0, x2, 0, n * aeNet.getHiddenSize());
        aeNetTrainer.release();

        MLPNet classifierNet = new MLPNet(h1, h2, m);
        BackPropCL classifierNetTrainer = new BackPropCL(classifierNet, x2, t, n, batch, 0.01f);
        classifierNetTrainer.init();
        System.out.println(classifierNetTrainer.train(10, 100000));
        readBuffer(classifierNetTrainer.memObjects.get(MemObject.OUT), n * classifierNet.getHiddenOutputSize() * Sizeof.cl_float, classifierNetTrainer.out);
        readBuffer(classifierNetTrainer.memObjects.get(MemObject.WEIGHTS), classifierNet.getWeightsSize() * Sizeof.cl_float, classifierNetTrainer.net.weights);
        System.arraycopy(classifierNetTrainer.net.weights, 0, w_res, (v + 1) * h1, (h1 + 1) * h2);
        classifierNetTrainer.release();

        MLPNet deepNet = new MLPNet(w_res, v, h1, h2, m);
        BackPropCL deepNetTrainer = new BackPropCL(deepNet, x, t, n, batch, 0.01f);
        deepNetTrainer.init();
        System.out.println(deepNetTrainer.train(10, 1000));
        readBuffer(deepNetTrainer.memObjects.get(MemObject.WEIGHTS), deepNet.getWeightsSize() * Sizeof.cl_float, deepNetTrainer.net.weights);
        deepNetTrainer.release();

        MLPNet testNet = new MLPNet(deepNetTrainer.net.weights, v, h1, h2, m);
        BackPropCL testNetTrainer = new BackPropCL(testNet, xTest, tTest, nTest, nTest, 0.3f);
        testNetTrainer.init();
        testNetTrainer.feedForward();
        readBuffer(testNetTrainer.memObjects.get(MemObject.OUT), nTest * testNet.getHiddenOutputSize() * Sizeof.cl_float, testNetTrainer.out);
        testNetTrainer.release();

        float[] actual = new float[nTest * m];
        System.arraycopy(testNetTrainer.out, nTest * testNet.getHiddenSize(), actual, 0, nTest * m);

        printInfo(nTest, m, tTest, actual);
    }

    private static void printInfo(int nTest, int m, float[] expected, float[] actual) {
        int numberOfError = 0;
        boolean error = false;
        for (int i = 0; i < nTest; i++)
        {
            error = false;
            for (int j = 0; j < m; j++)
            {
                System.out.print(actual[i * m + j] + " ");
                if (Math.abs(expected[i * m + j] - actual[i * m + j]) > 0.5f)
                {
                    error = true;
                    numberOfError++;
                }
            }
            if (error) System.out.println(" ERROR!!!");
            else System.out.println();
        }
        System.out.println("numberOfError = " + numberOfError);
    }
}
