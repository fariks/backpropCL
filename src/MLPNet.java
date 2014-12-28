import java.util.Random;

public class MLPNet {

    public int[] layerSizes;

    public float[] weights;

    public float[] weightsPrev;

    public MLPNet(int... layerSizes) {
        this.layerSizes = layerSizes;
        initWeightsRandomly();
    }

    public MLPNet(float[] weights, int... layerSizes) {
        this.layerSizes = layerSizes;
        initWeights(weights);
    }

    private void initWeights(float[] weights)
    {
        int size = 0;
        for (int i = 0; i < layerSizes.length - 1; i++)
        {
            size += (layerSizes[i] + 1) * layerSizes[i + 1];
        }
        this.weights = new float[size];
        this.weightsPrev = new float[size];
        System.arraycopy(weights, 0, this.weights, 0, size);
        System.arraycopy(weights, 0, this.weightsPrev, 0, size);
    }

    private void initWeightsRandomly()
    {
        int prev_size = 0;
        int size = 0;
        for (int i = 0; i < layerSizes.length - 1; i++)
        {
            size += (layerSizes[i] + 1) * layerSizes[i + 1];
        }
        weights = new float[size];
        weightsPrev = new float[size];

        Random r = new Random();
        size = 0;
        for (int i = 1; i < layerSizes.length; i++)
        {
            size += (layerSizes[i - 1] + 1) * layerSizes[i];
            for (int j = prev_size; j < size; j++)
            {
                weights[j] = (r.nextFloat() - 0.5f) / 10f;//(r.nextFloat() - 0.5f) * 2 * (1f / (float) Math.sqrt(layerSizes[i - 1]));
                weightsPrev[j] = weights[j];
            }
            prev_size += size;
        }
    }

    public int getInputSize() {
        return layerSizes[0];
    }

    public int getOutputSize() {
        return layerSizes[layerSizes.length - 1];
    }

    public int getHiddenOutputSize() {
        int size = 0;
        for (int i = 1; i < layerSizes.length; i++)
        {
            size += layerSizes[i];
        }
        return size;
    }

    public int getHiddenSize() {
        int size = 0;
        for (int i = 1; i < layerSizes.length - 1; i++)
        {
            size += layerSizes[i];
        }
        return size;
    }

    public int getHiddenSize(int k) {
        int size = 0;
        for (int i = 1; i < k; i++)
        {
            size += layerSizes[i];
        }
        return size;
    }

    public int getHiddenSizeWithBias(int k) {
        if (k != layerSizes.length - 1)
        {
            return layerSizes[k] + 1;
        }
        return layerSizes[k];
    }

    public int getWeightsSize() {
        int size = 0;
        for (int i = 0; i < layerSizes.length - 1; i++)
        {
            size += (layerSizes[i] + 1) * layerSizes[i + 1];
        }
        return size;
    }

    public int getWeightMatrixSizes(int k) {
        int size = 0;
        for (int i = 0; i < k; i++)
        {
            size += (layerSizes[i] + 1) * layerSizes[i + 1];
        }
        return size;
    }
}
