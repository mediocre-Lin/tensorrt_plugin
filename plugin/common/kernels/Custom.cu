#include "common/kernels/kernel.h"
namespace nvinfer1
{
namespace plugin
{
template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void pCustomKernel(const int n, const float negativeSlope, const float* input, float* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = input[i] > 0 ? input[i] : input[i] * negativeSlope;
    }
}

pluginStatus_t lCustomGPU(cudaStream_t stream, const int n, const float negativeSlope, const void* input, void* output)
{
    const int BS = 512;
    const int GS = (n + BS - 1) / BS;
    pCustomKernel<BS><<<GS, BS, 0, stream>>>(n, negativeSlope,
                                           (const float*) input,
                                           (float*) output);
    return STATUS_SUCCESS;
}

pluginStatus_t CustomInference(
    cudaStream_t stream, const int n, const float negativeSlope, const void* input, void* output)
{
    return lCustomGPU(stream, n, negativeSlope, (const float*) input, (float*) output);
}
} // namespace plugin
} // namespace nvinfer1
