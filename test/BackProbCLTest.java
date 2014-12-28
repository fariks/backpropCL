/*
import org.jocl.Sizeof;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;

import static util.CLWrapper.readBuffer;
*/
/**
 * Created by Алексей on 01.05.14.
 *//*


public class BackProbCLTest {

    private BackPropCL backPropCL;

    private MLPNet net;

    private int n = 4;

    @Before
    public void setUp()
    {

        int v = 4;
        int m = 2;
        int h = 3;

        net = Mockito.spy(new MLPNet(v, h, m));
        Mockito.doNothing().when(net).initWeights();

        int size = net.getWeightMatrixSizes(net.layerSizes.length - 1);
        net.weights = new float[size];
        net.weightsPrev = new float[size];

        for (int i = 0; i < size; i++)
        {
            net.weights[i] = 0.5f;
            net.weightsPrev[i] = net.weights[i];
        }

        float x[] = new float[] {
                1, 0, 0, 0,
                0 ,1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        };
        int[] t = new int[] {
                1, 0,
                1, 0,
                0, 1,
                0, 1
        };
        backPropCL = new BackPropCL(net, x, t, n);
        backPropCL.init();
    }

    @After
    public void tearDown()
    {
        backPropCL.release();
    }

    @Test
    public void testFeedForward() throws Exception
    {

        float[] expectedRes = new float[] {
                1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f,
                1.5965879f, 1.5965879f, 1.5965879f, 1.5965879f,
                1.5965879f, 1.5965879f, 1.5965879f, 1.5965879f};

        float[] expectedResOut = new float[] {
                0.7310586f, 0.7310586f, 0.7310586f, 0.7310586f,
                0.7310586f,0.7310586f, 0.7310586f, 0.7310586f,
                0.7310586f, 0.7310586f, 0.7310586f, 0.7310586f,
                0.8315409f, 0.8315409f, 0.8315409f, 0.8315409f,
                0.8315409f, 0.8315409f, 0.8315409f, 0.8315409f
        };
        float delta = 0.0001f;

        backPropCL.feedForward();

        readBuffer(backPropCL.memObjects.get("sum"), n * net.getHiddenOutputSize() * Sizeof.cl_float, backPropCL.sum);
        readBuffer(backPropCL.memObjects.get("out"), n * net.getHiddenOutputSize() * Sizeof.cl_float, backPropCL.out);

        Assert.assertArrayEquals(expectedRes, backPropCL.sum, delta);
        Assert.assertArrayEquals(expectedResOut, backPropCL.out, delta);
        */
/*System.out.println("Result: ");
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < net.getHiddenOutputSize(); j++)
            {
                System.out.print(backPropCL.sum[i * net.getHiddenOutputSize() + j] + " ");
            }
            System.out.println();
        }*//*


        */
/*for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < net.getHiddenOutputSize(); j++)
            {
                System.out.print(backPropCL.out[i * net.getHiddenOutputSize() + j] + " ");
            }
            System.out.println();
        }*//*

    }

    @Test
    public void testOutputError() throws Exception
    {
        float[] expectedRes = new float[] {
                0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f,
                0.023597863f, -0.11648279f,
                0.023597863f, -0.11648279f,
                -0.11648279f, 0.023597863f,
                -0.11648279f, 0.023597863f };
        float delta = 0.0001f;

        backPropCL.feedForward();

        backPropCL.outputError();

        readBuffer(backPropCL.memObjects.get("sigma"), n * net.getHiddenOutputSize() * Sizeof.cl_float, backPropCL.sigma);

        Assert.assertArrayEquals(expectedRes, backPropCL.sigma, delta);

        */
/*for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < net.getHiddenOutputSize(); j++)
            {
                System.out.print(backPropCL.sigma[i * net.getHiddenOutputSize() + j] + " ");
            }
            System.out.println();
        }*//*

    }

    @Test
    public void testHiddenError() throws Exception
    {
        float[] expectedRes = new float[] {
                -0.009131142f, -0.009131142f, -0.009131142f,
                -0.009131142f, -0.009131142f, -0.009131142f,
                -0.009131142f, -0.009131142f, -0.009131142f,
                -0.009131142f, -0.009131142f, -0.009131142f,
                0.023597863f, -0.11648279f,
                0.023597863f, -0.11648279f,
                -0.11648279f, 0.023597863f,
                -0.11648279f, 0.023597863f };
        float delta = 0.0001f;

        backPropCL.feedForward();

        backPropCL.outputError();

        backPropCL.hiddenError();

        readBuffer(backPropCL.memObjects.get("sigma"), n * net.getHiddenOutputSize() * Sizeof.cl_float, backPropCL.sigma);

        Assert.assertArrayEquals(expectedRes, backPropCL.sigma, delta);

        */
/*for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < net.getHiddenOutputSize(); j++)
            {
                System.out.print(backPropCL.sigma[i * net.getHiddenOutputSize() + j] + " ");
            }
            System.out.println();
        }*//*

    }

    @Test
    public void testAdjustWeights()
    {
        float[] expectedRes = new float[] {
                0.49726066f, 0.49931517f, 0.49931517f, 0.49931517f, 0.49931517f,
                0.49726066f, 0.49931517f, 0.49931517f, 0.49931517f, 0.49931517f,
                0.49726066f, 0.49931517f, 0.49931517f, 0.49931517f, 0.49931517f,

                0.48606727f, 0.48981434f, 0.48981434f, 0.48981434f,
                0.48606727f, 0.48981434f, 0.48981434f, 0.48981434f};
        float delta = 0.0001f;

        backPropCL.feedForward();

        backPropCL.outputError();

        backPropCL.hiddenError();

        backPropCL.adjustWeights();

        readBuffer(backPropCL.memObjects.get("w"), net.getWeightsSize() * Sizeof.cl_float, backPropCL.net.weights);

        Assert.assertArrayEquals(expectedRes, backPropCL.net.weights, delta);

        */
/*int h_prev = 0;
        int v_prev = 0;
        for (int i = 1; i < backPropCL.net.layerSizes.length; i++)
        {
            int h = backPropCL.net.layerSizes[i];
            int v = backPropCL.net.layerSizes[i - 1] + 1;
            for (int j = 0; j < h; j++)
            {
                for (int k = 0; k < v; k++)
                {
                    System.out.print(backPropCL.net.weights[(i - 1) * h_prev * v_prev + j * v + k] + " ");
                }
                System.out.println();
            }
            h_prev = h;
            v_prev = v;
            System.out.println();
        }*//*

    }

}
*/
