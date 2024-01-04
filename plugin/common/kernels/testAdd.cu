#include "common/kernels/kernel.h"
namespace nvinfer1
{
namespace plugin
{
template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void testAddKernel(const int n, const float bias, const float* input_1,const float* input_2, float* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = input_1[i] +  bias + input_2[i] ;
        // printf("%f, %f, %f, %f,\n",output[i],input_1[i],bias, input_2[i]);

    }
}

pluginStatus_t testAddGPU(cudaStream_t stream, const int n, const float bias, const void* input_1,const void* input_2,void* output)
{
    const int BS = 512;
    const int GS = (n + BS - 1) / BS;

    testAddKernel<BS><<<GS, BS, 0, stream>>>(n, bias, (const float*) input_1,(const float*) input_2, (float*) output);
    return STATUS_SUCCESS;
}

pluginStatus_t testAddInference(
    cudaStream_t stream, int32_t n, float bias, void const* input_1, void const* input_2, void* output)
{
    // std::cout << "start testAddGPU " << std::endl;
    // std::cout << "n : " << n << " , bias : " << bias << std::endl;
    // return STATUS_SUCCESS;
    return testAddGPU(stream, n, bias, (const float*) input_1, (const float*) input_2, (float*) output);
}
} // namespace plugin
} // namespace nvinfer1
