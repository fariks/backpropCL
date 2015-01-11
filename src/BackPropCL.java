import com.google.common.collect.Maps;
import memobjects.MemObject;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

import java.util.*;

import static org.jocl.CL.*;
import static util.CLWrapper.*;

public class BackPropCL {

    public MLPNet net;

    private float[] x;
    protected float[] x_cur;

    private float[] t;
    private float[] t_cur;

    private int n;
    private boolean fullBatch;

    private static final String program = "E:\\ms\\backprop\\src\\kernel\\backprop.cl";

    public Map<MemObject, cl_mem> memObjects = Maps.newHashMap();

    private cl_kernel forwardKernel;

    private cl_kernel outputErrorKernel;

    private cl_kernel hiddenErrorKernel;

    private cl_kernel adjustWeightsKernel;

    private float nu = 0.01f;
    private float momentum = 0.9f;
    protected int batch;
    private int[] dist;

    public float sum[];
    public float out[];
    public float sigma[];

    public float prev_weights[];

    public BackPropCL(MLPNet net, float[] x, float[] t, int n, int batch, float nu) {
        this.net = net;
        this.n = n;
        this.x = x;
        this.t = t;
        this.batch = batch;
        this.nu = nu;
        this.fullBatch = n == batch;
        prev_weights = new float[net.getWeightsSize()];
        sum = new float[batch * net.getHiddenOutputSize()];
        out = new float[batch * net.getHiddenOutputSize()];
        sigma = new float[batch * net.getHiddenOutputSize()];
        if (!fullBatch) {
            x_cur = new float[batch * net.getInputSize()];
            t_cur = new float[batch * net.getOutputSize()];
        } else {
            x_cur = x;
            t_cur = t;
        }
    }

    public void init() {
        cl_init(program);
        forwardKernel = createKernel("forward");
        outputErrorKernel = createKernel("output_error");
        hiddenErrorKernel = createKernel("hidden_error");
        adjustWeightsKernel = createKernel("adjust_weights");

        memObjects.put(MemObject.SUM, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getHiddenOutputSize(), sum));
        memObjects.put(MemObject.OUT, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getHiddenOutputSize(), out));
        memObjects.put(MemObject.SIGMA, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getHiddenOutputSize(), sigma));
        memObjects.put(MemObject.SOURCE, createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getInputSize(), x_cur));
        memObjects.put(MemObject.TARGET, createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getOutputSize(), t_cur));
        memObjects.put(MemObject.WEIGHTS, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * net.getWeightsSize(), net.weights));
        memObjects.put(MemObject.WEIGHTS_PREV, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * net.getWeightsSize(), net.weights));
        setBatch();
    }

    private void setBatch() {
        if (!fullBatch) {
            Random r = new Random();
            int c;
            for (int i = 0, m = 0; i < x_cur.length; i += net.getInputSize(), m += net.getOutputSize()) {
                c = r.nextInt(n);

                System.arraycopy(x, c * net.getInputSize(), x_cur, i, net.getInputSize());
                System.arraycopy(t, c * net.getOutputSize(), t_cur, m, net.getOutputSize());
            }
            writeBuffer(memObjects.get(MemObject.SOURCE), Sizeof.cl_float * batch * net.getInputSize(), x_cur);
            writeBuffer(memObjects.get(MemObject.TARGET), Sizeof.cl_float * batch * net.getOutputSize(), t_cur);
        }
    }

    public void release() {
        cl_release(memObjects, forwardKernel, outputErrorKernel, hiddenErrorKernel, adjustWeightsKernel);
    }

    public void feedForward() {
        int outOffset = 0;
        int inputOffset = 0;
        for (int i = 0; i < net.layerSizes.length - 1; i++) {
            if (i == 0) {
                setKernelArg(forwardKernel, 0, Sizeof.cl_mem, memObjects.get(MemObject.SOURCE));
            } else {
                setKernelArg(forwardKernel, 0, Sizeof.cl_mem, memObjects.get(MemObject.OUT));
            }
            setKernelArg(forwardKernel, 1, Sizeof.cl_mem, memObjects.get(MemObject.WEIGHTS));
            setKernelArg(forwardKernel, 2, Sizeof.cl_int, new int[]{net.layerSizes[i + 1]});
            setKernelArg(forwardKernel, 3, Sizeof.cl_int, new int[]{net.layerSizes[i] + 1});
            setKernelArg(forwardKernel, 4, Sizeof.cl_int, new int[]{net.getWeightMatrixSizes(i)});
            setKernelArg(forwardKernel, 5, Sizeof.cl_int, new int[]{inputOffset});
            setKernelArg(forwardKernel, 6, Sizeof.cl_int, new int[]{outOffset});
            setKernelArg(forwardKernel, 7, Sizeof.cl_mem, memObjects.get(MemObject.SUM));
            setKernelArg(forwardKernel, 8, Sizeof.cl_mem, memObjects.get(MemObject.OUT));

            runKernel(forwardKernel, 2, new long[]{batch, net.layerSizes[i + 1]}, null);

            if (i != 0) {
                inputOffset += batch * net.layerSizes[i];
            }

            outOffset += batch * net.layerSizes[i + 1];
        }
    }

    public void outputError() {
        setKernelArg(outputErrorKernel, 0, Sizeof.cl_mem, memObjects.get(MemObject.TARGET));
        setKernelArg(outputErrorKernel, 1, Sizeof.cl_mem, memObjects.get(MemObject.OUT));
        setKernelArg(outputErrorKernel, 2, Sizeof.cl_mem, memObjects.get(MemObject.SUM));
        setKernelArg(outputErrorKernel, 3, Sizeof.cl_int, new int[]{batch * net.getHiddenSize()});
        setKernelArg(outputErrorKernel, 4, Sizeof.cl_int, new int[]{net.getOutputSize()});
        setKernelArg(outputErrorKernel, 5, Sizeof.cl_mem, memObjects.get(MemObject.SIGMA));

        runKernel(outputErrorKernel, 2, new long[]{batch, net.getOutputSize()}, null);
    }

    public void hiddenError() {
        for (int i = net.layerSizes.length - 2; i > 0; i--) {
            setKernelArg(hiddenErrorKernel, 0, Sizeof.cl_mem, memObjects.get(MemObject.WEIGHTS));
            setKernelArg(hiddenErrorKernel, 1, Sizeof.cl_mem, memObjects.get(MemObject.SIGMA));
            setKernelArg(hiddenErrorKernel, 2, Sizeof.cl_mem, memObjects.get(MemObject.SUM));
            setKernelArg(hiddenErrorKernel, 3, Sizeof.cl_int, new int[]{net.layerSizes[i]});
            setKernelArg(hiddenErrorKernel, 4, Sizeof.cl_int, new int[]{net.layerSizes[i + 1]});
            setKernelArg(hiddenErrorKernel, 5, Sizeof.cl_int, new int[]{net.getWeightMatrixSizes(i)});
            setKernelArg(hiddenErrorKernel, 6, Sizeof.cl_int, new int[]{net.getHiddenSize(i + 1) * batch});
            setKernelArg(hiddenErrorKernel, 7, Sizeof.cl_int, new int[]{net.getHiddenSize(i) * batch});

            runKernel(hiddenErrorKernel, 2, new long[]{batch, net.layerSizes[i]}, null);
        }
    }

    public void adjustWeights() {
        int outOffset = 0;
        for (int i = 0; i < net.layerSizes.length - 1; i++) {
            if (i == 0) {
                setKernelArg(adjustWeightsKernel, 0, Sizeof.cl_mem, memObjects.get(MemObject.SOURCE));
            } else {
                setKernelArg(adjustWeightsKernel, 0, Sizeof.cl_mem, memObjects.get(MemObject.OUT));
            }
            setKernelArg(adjustWeightsKernel, 1, Sizeof.cl_mem, memObjects.get(MemObject.WEIGHTS));
            setKernelArg(adjustWeightsKernel, 2, Sizeof.cl_mem, memObjects.get(MemObject.SIGMA));
            setKernelArg(adjustWeightsKernel, 3, Sizeof.cl_int, new float[]{nu});
            setKernelArg(adjustWeightsKernel, 4, Sizeof.cl_int, new int[]{net.layerSizes[i + 1]});
            setKernelArg(adjustWeightsKernel, 5, Sizeof.cl_int, new int[]{net.getHiddenSizeWithBias(i)});
            setKernelArg(adjustWeightsKernel, 6, Sizeof.cl_int, new int[]{batch});
            setKernelArg(adjustWeightsKernel, 7, Sizeof.cl_int, new int[]{net.getWeightMatrixSizes(i)});
            setKernelArg(adjustWeightsKernel, 8, Sizeof.cl_int, new int[]{outOffset});
            setKernelArg(adjustWeightsKernel, 9, Sizeof.cl_int, new int[]{net.getHiddenSize(i + 1) * batch});
            setKernelArg(adjustWeightsKernel, 10, Sizeof.cl_mem, memObjects.get(MemObject.WEIGHTS_PREV));
            setKernelArg(adjustWeightsKernel, 11, Sizeof.cl_float, new float[]{momentum});

            runKernel(adjustWeightsKernel, 2, new long[]{net.layerSizes[i + 1], net.getHiddenSizeWithBias(i)}, null);

            if (i != 0) {
                outOffset += batch * net.layerSizes[i];
            }
        }
    }

    public int train(int s, int maxEpoch) {
        int k = 0;
        boolean stop = false;
        float delta = 0.0001f;
        int epoch = 0;
        for (epoch = 0; epoch < maxEpoch && !stop; epoch++) {
            feedForward();
            outputError();
            hiddenError();
            adjustWeights();
            k++;
            if (k >= s) {
                readBuffer(memObjects.get(MemObject.WEIGHTS), net.getWeightsSize() * Sizeof.cl_float, net.weights);
                /*readBuffer(memObjects.get(MemObject.OUT), batch * net.getHiddenOutputSize() * Sizeof.cl_float, out);
                float[] dest = new float[batch * net.getOutputSize()];
                System.arraycopy(out, batch * net.getHiddenSize(), dest, 0, batch * net.getOutputSize());
                float errorSum = 0.f;
                for (int i = 0; i < dest.length; i ++)
                {
                    errorSum += Math.abs(dest[i] - x_cur[i]);
                }
                System.out.println(errorSum);*/
                setBatch();
                stop = checkDifference(net.weights, prev_weights, delta);
                validate();
                k = 0;
            }
        }
        //Final forward for full training set
        sum = new float[n * net.getHiddenOutputSize()];
        out = new float[n * net.getHiddenOutputSize()];
        sigma = new float[n * net.getHiddenOutputSize()];
        memObjects.put(MemObject.SUM, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getHiddenOutputSize(), sum));
        memObjects.put(MemObject.OUT, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getHiddenOutputSize(), out));
        memObjects.put(MemObject.SIGMA, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getHiddenOutputSize(), sigma));
        memObjects.put(MemObject.SOURCE, createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getInputSize(), x));
        memObjects.put(MemObject.TARGET, createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getOutputSize(), t));
        feedForward();
        //end
        return epoch;
    }

    protected boolean validate() {return false;}

    protected float[] feedForwardInternal(float[] x, float[] t, int n)
    {
        sum = new float[n * net.getHiddenOutputSize()];
        out = new float[n * net.getHiddenOutputSize()];
        sigma = new float[n * net.getHiddenOutputSize()];
        clReleaseMemObject(memObjects.get(MemObject.SUM));
        clReleaseMemObject(memObjects.get(MemObject.OUT));
        clReleaseMemObject(memObjects.get(MemObject.SIGMA));
        clReleaseMemObject(memObjects.get(MemObject.SOURCE));
        clReleaseMemObject(memObjects.get(MemObject.TARGET));
        memObjects.put(MemObject.SUM, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getHiddenOutputSize(), sum));
        memObjects.put(MemObject.OUT, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getHiddenOutputSize(), out));
        memObjects.put(MemObject.SIGMA, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getHiddenOutputSize(), sigma));
        memObjects.put(MemObject.SOURCE, createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getInputSize(), x));
        memObjects.put(MemObject.TARGET, createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n * net.getOutputSize(), t));
        feedForward();
        float[] tmp = new float[n * net.getHiddenOutputSize()];
        readBuffer(memObjects.get(MemObject.OUT), n * net.getHiddenOutputSize() * Sizeof.cl_float, tmp);
        float[] out = new float[n * net.getOutputSize()];
        System.arraycopy(tmp, n * net.getHiddenSize(), out, 0, n * net.getOutputSize());

        //Return all back
        clReleaseMemObject(memObjects.get(MemObject.SUM));
        clReleaseMemObject(memObjects.get(MemObject.OUT));
        clReleaseMemObject(memObjects.get(MemObject.SIGMA));
        clReleaseMemObject(memObjects.get(MemObject.SOURCE));
        clReleaseMemObject(memObjects.get(MemObject.TARGET));
        memObjects.put(MemObject.SUM, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getHiddenOutputSize(), sum));
        memObjects.put(MemObject.OUT, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getHiddenOutputSize(), out));
        memObjects.put(MemObject.SIGMA, createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getHiddenOutputSize(), sigma));
        memObjects.put(MemObject.SOURCE, createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getInputSize(), x_cur));
        memObjects.put(MemObject.TARGET, createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * batch * net.getOutputSize(), t_cur));
        return out;
    }

    private boolean checkDifference(float[] x, float[] y, float delta) {
        for (int i = 0; i < x.length; i++) {
            if (Math.abs(x[i] - y[i]) > delta) {
                prev_weights = x;
                return false;
            }
        }
        return true;
    }

    /*public void test(float[] x, int n)
    {
        readBuffer(memObjects.get(MemObject.WEIGHTS), Sizeof.cl_float * net.getWeightsSize(), net.weights);
        MLPNet testNet = new MLPNet(net.weights, net.layerSizes);
        BackPropCL backPropCL = new BackPropCL(testNet, x, x, n, n, dist);
        backPropCL.feedForward();
    }*/

    public static void main(String args[]) {
        int n = 91;
        int v = 13;
        int m = 13;
        int h1 = 1000;
        int h2 = 1000;
        int batch = 1;

        String sampleFile = "E:\\ms\\backprop\\src\\resourses\\rotate.txt";

        MLPNet net = new MLPNet(v, h1, h2, m);

        float[] x = new float[n * v];
        float[] t = new float[n * m];

        DataLoader.readSample(sampleFile, x, t, n, v, m);

        BackPropCL backPropCL = new BackPropCL(net, x, t, n, batch, 0.01f);
        backPropCL.init();
        long before = new Date().getTime();
        System.out.println(backPropCL.train(50000, 40000));
        System.out.println(new Date().getTime() - before);

        //readBuffer(backPropCL.memObjects.get(MemObject.OUT), batch * net.getHiddenOutputSize() * Sizeof.cl_float, backPropCL.out);
        //readBuffer(backPropCL.memObjects.get(MemObject.WEIGHTS), net.getWeightsSize() * Sizeof.cl_float, backPropCL.net.weights);

        //System.out.println("Result: ");
        /*for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < net.layerSizes[1]; j++)
            {
                System.out.print(backPropCL.out[i * net.layerSizes[1] + j] + " ");
            }
            System.out.println();
        }
        System.out.println();*/
        /*for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < net.layerSizes[2]; j++)
            {
                System.out.print(backPropCL.out[batch * net.layerSizes[1] + i * net.layerSizes[2] + j] + " ");
            }
            System.out.println();
        }*/
        /*for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < net.layerSizes[3]; j++)
            {
                System.out.print(backPropCL.out[n * (net.layerSizes[1] + net.layerSizes[2]) + i * net.layerSizes[3] + j] + " ");
            }
            System.out.println();
        }
        System.out.println("w: ");
        for (int i = 0; i < net.getWeightsSize(); i++)
        {
            System.out.print(backPropCL.net.weights[i] + " ");
        }*/

        backPropCL.release();
    }
}
