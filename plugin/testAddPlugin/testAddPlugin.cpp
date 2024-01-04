#include "testAddPlugin.h"
#include "common/checkMacrosPlugin.h"
#include "common/kernels/kernel.h"
namespace nvinfer1
{
namespace plugin
{
static char const* const KTEST_ADD_PLUGIN_VERSION{"1"};
static char const* const KTEST_ADD_PLUGIN_NAME{"Test_Add_TRT"};
PluginFieldCollection TestAddPluginCreator::mFC{};
std::vector<PluginField> TestAddPluginCreator::mPluginAttributes;
TestAddPluginCreator::TestAddPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* TestAddPluginCreator::getPluginName() const noexcept
{
    return KTEST_ADD_PLUGIN_NAME;
}

const char* TestAddPluginCreator::getPluginVersion() const noexcept
{
    return KTEST_ADD_PLUGIN_VERSION;
}

const PluginFieldCollection* TestAddPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* TestAddPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    PLUGIN_ASSERT(fc->nbFields == 1);
    PLUGIN_ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    float bias = *(static_cast<const float*>(fields[0].data));
    TestAddPlugin* obj = new TestAddPlugin{bias};
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

IPluginV2DynamicExt* TestAddPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    TestAddPlugin* obj = new TestAddPlugin{serialData, serialLength};
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}



TestAddPlugin::TestAddPlugin(float bias)
    : m_bias(bias)
{
    PLUGIN_VALIDATE(bias >= 0.0F);
}
TestAddPlugin::TestAddPlugin(const void* buffer, size_t length)
{
    char const *d = reinterpret_cast<char const*>(buffer), *a = d;
    m_bias = read<float>(d);
    PLUGIN_VALIDATE(d == a + length);
}

int32_t TestAddPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs TestAddPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(outputIndex == 0);
    return nvinfer1::DimsExprs(inputs[0]);
}

int TestAddPlugin::initialize() noexcept
{
    return 0;
}

void TestAddPlugin::terminate() noexcept {}

size_t TestAddPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int TestAddPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    PLUGIN_VALIDATE(inputDesc[0].dims.nbDims == inputDesc[1].dims.nbDims)
    int n1 = 1, n2 = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        n1 *= inputDesc[0].dims.d[i];
        n2 *= inputDesc[1].dims.d[i];
    }
    PLUGIN_ASSERT(n1 == n2);
    const void* inputData1 = inputs[0];
    const void* inputData2 = inputs[1];
    void* outputData = outputs[0];
    pluginStatus_t status = plugin::testAddInference(stream, n1, m_bias, inputData1, inputData2, outputData);
    return status;
}

size_t TestAddPlugin::getSerializationSize() const noexcept
{
    return sizeof(float);
}

void TestAddPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, m_bias);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

bool TestAddPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(0 <= pos && pos < 3);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    switch (pos)
    {
    case 0: return in[0].type == DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR;
    case 1: return in[1].type == DataType::kFLOAT && in[1].format == nvinfer1::TensorFormat::kLINEAR;
    case 2: return out[0].type == DataType::kFLOAT && out[0].format == nvinfer1::TensorFormat::kLINEAR;
    }
}

const char* TestAddPlugin::getPluginType() const noexcept
{
    return KTEST_ADD_PLUGIN_NAME;
}

const char* TestAddPlugin::getPluginVersion() const noexcept
{
    return KTEST_ADD_PLUGIN_VERSION;
}

void TestAddPlugin::destroy() noexcept
{
    delete this;
}

nvinfer1::IPluginV2DynamicExt* TestAddPlugin::clone() const noexcept
{
    auto* plugin = new TestAddPlugin(m_bias);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
}

void TestAddPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* TestAddPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

nvinfer1::DataType TestAddPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

void TestAddPlugin::attachToContext(
    cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept
{
}

void TestAddPlugin::detachFromContext() noexcept {}

void TestAddPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2 && in[0].desc.dims.d[1] != -1);
}


} // namespace plugin
} // namespace nvinfer1

// using namespace nvinfer1;
// using nvinfer1::plugin::CustomPluginCreator;
// using nvinfer1::plugin::Custom;

// static const char* Custom_PLUGIN_VERSION{"1"};
// static const char* Custom_PLUGIN_NAME{"Custom_TRT"};
// PluginFieldCollection CustomPluginCreator::mFC{};
// std::vector<PluginField> CustomPluginCreator::mPluginAttributes;

// // LeakyReLU {{{
// Custom::Custom(float negSlope)
//     : mNegSlope(negSlope)
//     , mBatchDim(1)
// {
// }

// Custom::Custom(const void* buffer, size_t length)
// {
//     const char *d = reinterpret_cast<const char *>(buffer), *a = d;
//     mNegSlope = read<float>(d);
//     mBatchDim = read<int>(d);
//     PLUGIN_VALIDATE(d == a + length);
// }

// int Custom::getNbOutputs() const noexcept
// {
//     return 1;
// }

// DimsExprs Custom::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
//                 int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
// {
//     nvinfer1::DimsExprs output(inputs[0]);
//     return output;
// }

// int Custom::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
//     const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
//     cudaStream_t stream) noexcept
// {
//     const void* inputData = inputs[0];
//     void* outputData = outputs[0];
//     //在common/kernel.h文件里按照IReluInference实现CustomInference
//     pluginStatus_t status = plugin::CustomInference(stream, mBatchDim, mNegSlope, inputData, outputData);
//     return status;
// }

// size_t Custom::getSerializationSize() const noexcept
// {
//     // mNegSlope, mBatchDim
//     return sizeof(float) + sizeof(int);
// }

// void Custom::serialize(void* buffer) const noexcept
// {
//     char *d = reinterpret_cast<char *>(buffer), *a = d;
//     write(d, mNegSlope);
//     write(d, mBatchDim);
//     PLUGIN_VALIDATE(d == a + getSerializationSize());
// }

// bool Custom::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
//         int nbInputs, int nbOutputs) noexcept
// {
//     assert(0 <= pos && pos < 2);
//     const auto *in = inOut;
//     const auto *out = inOut + nbInputs;
//     switch (pos) {
//         case 0:
//         return in[0].type == DataType::kFLOAT &&
//                 in[0].format == nvinfer1::TensorFormat::kLINEAR;
//         case 1:
//         return out[0].type == in[0].type &&
//                 out[0].format == nvinfer1::TensorFormat::kLINEAR;
//     }
// }

// int Custom::initialize() noexcept
// {
//     return 0;
// }

// void Custom::terminate() noexcept {}

// size_t Custom::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
//             int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
// {
//     return 0;
// }

// const char* Custom::getPluginType() const noexcept
// {
//     return Custom_PLUGIN_NAME;
// }

// const char* Custom::getPluginVersion() const noexcept
// {
//     return Custom_PLUGIN_VERSION;
// }

// void Custom::destroy() noexcept
// {
//     delete this;
// }

// IPluginV2DynamicExt* Custom::clone() const noexcept
// {
//     auto* plugin = new Custom(mNegSlope);
//     plugin->setPluginNamespace(mPluginNamespace.c_str());
//     plugin->initialize();
//     return plugin;
// }

// void Custom::setPluginNamespace(const char* pluginNamespace) noexcept
// {
//     mPluginNamespace = pluginNamespace;
// }

// const char* Custom::getPluginNamespace() const noexcept
// {
//     return mPluginNamespace.c_str();
// }

// nvinfer1::DataType Custom::getOutputDataType(
//     int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
// {
//     PLUGIN_VALIDATE(inputTypes && nbInputs > 0 && index == 0);
//     return inputTypes[0];
// }

// // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
// void Custom::attachToContext(
//     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
// {
// }

// // Detach the plugin object from its execution context.
// void Custom::detachFromContext() noexcept {}

// void Custom::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
//     const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
// {
//     // Not support dynamic shape in C dimension
//     PLUGIN_VALIDATE(nbInputs == 1 && in[0].desc.dims.d[1] != -1);
// }

// CustomPluginCreator::CustomPluginCreator()
// {
//     mPluginAttributes.clear();
//     mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));

//     mFC.nbFields = mPluginAttributes.size();
//     mFC.fields = mPluginAttributes.data();
// }

// const char* CustomPluginCreator::getPluginName() const noexcept
// {
//     return Custom_PLUGIN_NAME;
// }

// const char* CustomPluginCreator::getPluginVersion() const noexcept
// {
//     return Custom_PLUGIN_VERSION;
// }

// const PluginFieldCollection* CustomPluginCreator::getFieldNames() noexcept
// {
//     return &mFC;
// }

// IPluginV2DynamicExt* CustomPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
// {
//     const PluginField* fields = fc->fields;
//     PLUGIN_VALIDATE(fc->nbFields == 1);
//     PLUGIN_VALIDATE(fields[0].type == PluginFieldType::kFLOAT32);
//     float negSlope = *(static_cast<const float*>(fields[0].data));
//     Custom* obj = new Custom{negSlope};

//     obj->setPluginNamespace(mNamespace.c_str());
//     obj->initialize();
//     return obj;
// }

// IPluginV2DynamicExt* CustomPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t
// serialLength) noexcept
// {
//     // This object will be deleted when the network is destroyed, which will
//     // call CustomPlugin::destroy()
//     Custom* obj = new Custom{serialData, serialLength};
//     obj->setPluginNamespace(mNamespace.c_str());
//     obj->initialize();
//     return obj;
// }
