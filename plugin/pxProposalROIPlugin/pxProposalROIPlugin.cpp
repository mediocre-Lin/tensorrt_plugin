#include "pxProposalROIPlugin.h"
#include "common/checkMacrosPlugin.h"
#include "common/kernels/kernel.h"
namespace nvinfer1
{
namespace plugin
{
static char const* const KPX_PROPOSAL_PLUGIN_VERSION{"1"};
static char const* const KPX_PROPOSAL_PLUGIN_NAME{"PX_PROPOSAL_PLUGIN"};
PluginFieldCollection PxProposalROIPluginCreator::mFC{};
std::vector<PluginField> PxProposalROIPluginCreator::mPluginAttributes;
PxProposalROIPlugin::PxProposalROIPlugin(
    int pre_nms_num, int roi_num, int img_h, int img_w, float nms_score, float nms_iou)
    :mpre_nms_num(pre_nms_num),mroi_num(roi_num),mimg_h(img_h),mimg_w(img_w),mnms_score(nms_score),mnms_iou(nms_iou)
{
    PLUGIN_VALIDATE(mpre_nms_num > 0);
    PLUGIN_VALIDATE(mroi_num > 0);
    PLUGIN_VALIDATE(mpre_nms_num >= mroi_num);
    PLUGIN_VALIDATE(mimg_h > 0);
    PLUGIN_VALIDATE(mimg_w > 0);
    PLUGIN_VALIDATE(mnms_score >= 0.0F);
    PLUGIN_VALIDATE(mnms_iou >= 0.0F);
}
PxProposalROIPlugin::PxProposalROIPlugin(const void* buffer, size_t length) {
    char const *d = reinterpret_cast<char const*>(buffer), *a = d;
    mpre_nms_num = read<int>(d);
    mroi_num = read<int>(d);
    mimg_h = read<int>(d);
    mimg_w = read<int>(d);
    mnms_score = read<float>(d);
    mnms_iou = read<float>(d);
    PLUGIN_VALIDATE(d == a + length);
}
int32_t PxProposalROIPlugin::getNbOutputs() const noexcept
{
    return 1;
}
DimsExprs PxProposalROIPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(outputIndex == 0);
    return nvinfer1::DimsExprs(inputs[0]);
}
int PxProposalROIPlugin::initialize() noexcept
{
    return 0;
}
void PxProposalROIPlugin::terminate() noexcept {}
size_t PxProposalROIPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return size_t();
}
} // namespace plugin
} // namespace nvinfer1