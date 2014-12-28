/*float f(float x)
{
    //float tmp = exp(-2 * x);
	return tanh(x);//(1.0 - tmp) / (1.0 + tmp);
}

float df(float x)
{
	float tanh = f(x);
	return 1.0 - tanh * tanh;
}*/

float f(float x)
{
	return 1.0 / (1.0 + exp(-x));
}

float df(float x)
{
	float sigm = f(x);
	return sigm * (1 - sigm);
}

__kernel void forward(
	__global float* input,
	__global float* w,
	int n,
	int m,
	int w_offset,
	int input_offset,
	int out_offset,
	__global float* osum,
	__global float* out
)
{
	int i = get_global_id(0); //batch
	int j = get_global_id(1); //n

	float sum = w[w_offset + j * m];
	for (int k = 1; k < m; k++)
	{
		sum += input[input_offset + i * (m - 1)+ k - 1] * w[w_offset + j * m + k];
	}
	osum[out_offset + i * n + j] = sum;
	out[out_offset + i * n + j] = f(sum);
}

__kernel void output_error(
	__global float* t,
	__global float* y,
	__global float* osum,
	int offset,
	int output,
	__global float* osigma
)
{
	int i = get_global_id(0); //batch
	int j = get_global_id(1); //output

	osigma[offset + i * output + j] = (t[i * output + j] - y[offset + i * output + j]) * df(osum[offset + i * output + j]);
}

__kernel void hidden_error(
	__global float* w,
	__global float* sigma,
	__global float* sum,
	int hidden,
	int output,
	int w_offset,
	int osigma_offset,
	int hsigma_offset
)
{
	int i = get_global_id(0); //batch
	int j = get_global_id(1); //hidden

	float tsum = 0.0;
	for (int k = 0; k < output; k++) // output without bias
	{
		tsum += sigma[osigma_offset + i * output + k] * w[w_offset + k * (hidden + 1) + j + 1];
	}
	sigma[hsigma_offset + i * hidden + j] = tsum * df(sum[hsigma_offset + i * hidden + j]);
}

__kernel void adjust_weights(
	__global float* input,
	__global float* w,
	__global float* sigma,
	float nu,
	int n,
	int m,
	int batch,
	int w_offset,
	int input_offset,
	int sigma_offset,
	__global float* w_prev,
	float momentum
)
{
	int j = get_global_id(0); //n
	int k = get_global_id(1); //m
	if (k == 0)
	{
		float sum = 0.0;
		for (int i = 0; i < batch; i++)
		{
			sum += sigma[sigma_offset + i * n + j];
		}
		float delta = w[w_offset + j * m] - w_prev[w_offset + j * m];
		w_prev[w_offset + j * m] = w[w_offset + j * m];
		w[w_offset + j * m] += (nu * sum) / batch + momentum * delta;
	}
	else
	{
		float sum = 0.0;
		for (int i = 0; i < batch; i++)
		{
			sum += sigma[sigma_offset + i * n + j] * input[input_offset + i * (m - 1) + k - 1];
		}
		float delta = w[w_offset + j * m + k] - w_prev[w_offset + j * m + k];
		w_prev[w_offset + j * m + k] = w[w_offset + j * m + k];
		w[w_offset + j * m + k] += (nu * sum) / batch + momentum * delta;
	}
}


