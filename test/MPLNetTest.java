import org.junit.Assert;
import org.junit.Test;

/**
 * Created by Алексей on 01.05.14.
 */
public class MPLNetTest {

    @Test
    public void testInitWeightsRandomly()
    {
        int[] layerSizes = new int[] {4, 6, 4, 2};
        MLPNet net = new MLPNet(layerSizes);

        Assert.assertEquals(net.weights.length, 5 * 6 + 7 * 4 + 5 * 2);
    }

    @Test
    public void testInitWeights()
    {
        int[] layerSizes = new int[] {2, 3, 1};
        float[] weights = new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f};
        float delta = 0.0001f;

        MLPNet net = new MLPNet(weights, layerSizes);

        Assert.assertArrayEquals(net.weights, weights, delta);
        Assert.assertArrayEquals(net.weightsPrev, weights, delta);
    }
}
