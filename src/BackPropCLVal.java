import memobjects.MemObject;
import org.jocl.Sizeof;

import static util.CLWrapper.readBuffer;

public class BackPropCLVal extends BackPropCL {

    private float[] xVal;
    private float[] tVal;
    private int nVal;

    public BackPropCLVal(MLPNet net, float[] x, float[] t, int n, int batch, float nu, float[] xVal, float[] tVal, int nVal) {
        super(net, x, t, n, batch, nu);
        this.xVal = xVal;
        this.tVal = tVal;
        this.nVal = nVal;
    }

    @Override
    protected boolean validate() {
        float errorSum = 0.f;
        readBuffer(memObjects.get(MemObject.OUT), batch * net.getHiddenOutputSize() * Sizeof.cl_float, out);
        float[] dest = new float[batch * net.getOutputSize()];
        System.arraycopy(out, batch * net.getHiddenSize(), dest, 0, batch * net.getOutputSize());
        errorSum = 0.f;
        for (int i = 0; i < dest.length; i++) {
            errorSum += Math.sqrt((dest[i] - x_cur[i]) * (dest[i] - x_cur[i]));
        }
        System.out.print(errorSum / dest.length + " ");

        float[] out = feedForwardInternal(xVal, tVal, nVal);
        errorSum = 0.f;
        for (int i = 0; i < out.length; i ++)
        {
            errorSum += Math.sqrt((out[i] - tVal[i]) * (out[i] - tVal[i]));
        }
        System.out.println(errorSum / out.length);


        return false;
    }
}
