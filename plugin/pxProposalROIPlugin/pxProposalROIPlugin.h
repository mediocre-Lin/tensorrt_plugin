#ifndef TRT_PX_PROPOSAL_ROI_PLUGIN_H
#define TRT_PX_PROPOSAL_ROI_PLUGIN_H
#include "NvInferPlugin.h"
#include "common/kernels/kernel.h"
#include "common/plugin.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class PxProposalROIPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    PxProposalROIPlugin(int pre_nms_num, int roi_num, int img_h, int img_w, float nms_score, float nms_iou);

    PxProposalROIPlugin(const void* buffer, size_t length);

    PxProposalROIPlugin() = delete;

    ~PxProposalROIPlugin() override = default;

    int32_t getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    void attachToContext(
        cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;
    void detachFromContext() noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
private:
    int mpre_nms_num;
    int mroi_num;
    int mimg_h;
    int mimg_w;
    float mnms_score;
    float mnms_iou;
    std::string mPluginNamespace;
    std::string mNamespace;
};

class PxProposalROIPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    PxProposalROIPluginCreator();

    ~PxProposalROIPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_L_TEST_ADD_PLUGIN_H
