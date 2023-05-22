#include "cassert"
#include "common.hpp"
#include <unordered_map>

namespace {

cl_sampler_info ur2clSamplerInfo(ur_sampler_info_t ur_info) {
  switch (ur_info) {
#define CASE(UR_INFO, CL_INFO)                                                 \
  case UR_INFO:                                                                \
    return CL_INFO;

    CASE(UR_SAMPLER_INFO_REFERENCE_COUNT, CL_SAMPLER_REFERENCE_COUNT)
    CASE(UR_SAMPLER_INFO_CONTEXT, CL_SAMPLER_CONTEXT)
    CASE(UR_SAMPLER_INFO_NORMALIZED_COORDS, CL_SAMPLER_NORMALIZED_COORDS)
    CASE(UR_SAMPLER_INFO_ADDRESSING_MODE, CL_SAMPLER_ADDRESSING_MODE)
    CASE(UR_SAMPLER_INFO_FILTER_MODE, CL_SAMPLER_FILTER_MODE)

#undef CASE

  default:
    assert(0 && "Unhandled: ur_sampler_info_t");
  }
}

cl_addressing_mode ur2clAddressingMode(ur_sampler_addressing_mode_t mode) {
  switch (mode) {

#define CASE(UR_MODE, CL_MODE)                                                 \
  case UR_MODE:                                                                \
    return CL_MODE;

    CASE(UR_SAMPLER_ADDRESSING_MODE_NONE, CL_ADDRESS_NONE);
    CASE(UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE, CL_ADDRESS_CLAMP_TO_EDGE);
    CASE(UR_SAMPLER_ADDRESSING_MODE_CLAMP, CL_ADDRESS_CLAMP);
    CASE(UR_SAMPLER_ADDRESSING_MODE_REPEAT, CL_ADDRESS_REPEAT);
    CASE(UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT,
         CL_ADDRESS_MIRRORED_REPEAT);

#undef CASE

  default:
    assert(0 && "Unhandled: ur_sampler_addressing_mode_t");
    break;
  }
}

cl_filter_mode ur2clFilterMode(ur_sampler_filter_mode_t mode) {
  switch (mode) {

#define CASE(UR_MODE, CL_MODE)                                                 \
  case UR_MODE:                                                                \
    return CL_MODE;

    CASE(UR_SAMPLER_FILTER_MODE_NEAREST, CL_FILTER_NEAREST)
    CASE(UR_SAMPLER_FILTER_MODE_LINEAR, CL_FILTER_LINEAR)

#undef CASE

  default:
    assert(0 && "Unhandled: ur_sampler_filter_mode_t");
    break;
  }
}

ur_sampler_addressing_mode_t cl2urAddressingMode(cl_addressing_mode mode) {
  switch (mode) {

#define CASE(CL_MODE, UR_MODE)                                                 \
  case CL_MODE:                                                                \
    return UR_MODE;

    CASE(CL_ADDRESS_NONE, UR_SAMPLER_ADDRESSING_MODE_NONE);
    CASE(CL_ADDRESS_CLAMP_TO_EDGE, UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE);
    CASE(CL_ADDRESS_CLAMP, UR_SAMPLER_ADDRESSING_MODE_CLAMP);
    CASE(CL_ADDRESS_REPEAT, UR_SAMPLER_ADDRESSING_MODE_REPEAT);
    CASE(CL_ADDRESS_MIRRORED_REPEAT,
         UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT);

#undef CASE

  default:
    std::cout << mode << std::endl;
    assert(0 && "Unhandled: cl_addressing_mode");
    break;
  }
}

ur_sampler_filter_mode_t cl2urFilterMode(cl_filter_mode mode) {
  switch (mode) {
#define CASE(CL_MODE, UR_MODE)                                                 \
  case CL_MODE:                                                                \
    return UR_MODE;

    CASE(CL_FILTER_NEAREST, UR_SAMPLER_FILTER_MODE_NEAREST)
    CASE(CL_FILTER_LINEAR, UR_SAMPLER_FILTER_MODE_LINEAR);

#undef CASE

  default:
    assert(0 && "Unhandled: cl_filter_mode");
    break;
  }
}

void cl2urSamplerInfoValue(cl_sampler_info info, size_t infoSize,
                           void *infoValue) {
  if (!infoValue) {
    return;
  }
  switch (info) {
  case CL_SAMPLER_ADDRESSING_MODE: {
    cl_addressing_mode clValue =
        *reinterpret_cast<cl_addressing_mode *>(infoValue);
    *reinterpret_cast<ur_sampler_addressing_mode_t *>(infoValue) =
        cl2urAddressingMode(clValue);
    break;
  }
  case CL_SAMPLER_FILTER_MODE: {
    cl_filter_mode clMode = *reinterpret_cast<cl_filter_mode *>(infoValue);
    *reinterpret_cast<ur_sampler_filter_mode_t *>(infoValue) =
        cl2urFilterMode(clMode);
    break;
  }

  default:
    break;
  }
}

} // namespace

ur_result_t urSamplerCreate(ur_context_handle_t hContext,
                            const ur_sampler_desc_t *pDesc,
                            ur_sampler_handle_t *phSampler) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pDesc, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phSampler, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  // Initialize properties according to OpenCL 2.1 spec.
  ur_result_t error_code;
  cl_addressing_mode addressingMode =
      ur2clAddressingMode(pDesc->addressingMode);
  cl_filter_mode filterMode = ur2clFilterMode(pDesc->filterMode);

  // Always call OpenCL 1.0 API
  *phSampler = cl_adapter::cast<ur_sampler_handle_t>(clCreateSampler(
      cl_adapter::cast<cl_context>(hContext),
      static_cast<cl_bool>(pDesc->normalizedCoords), addressingMode, filterMode,
      cl_adapter::cast<cl_int *>(&error_code)));

  return map_cl_error_to_ur(error_code);
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetInfo(ur_sampler_handle_t hSampler, ur_sampler_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pPropValue || pPropSizeRet, UR_RESULT_ERROR_INVALID_VALUE);

  cl_sampler_info sampler_info = ur2clSamplerInfo(propName);
  static_assert(sizeof(cl_addressing_mode) ==
                sizeof(ur_sampler_addressing_mode_t));

  if (ur_result_t err = map_cl_error_to_ur(
          clGetSamplerInfo(cl_adapter::cast<cl_sampler>(hSampler), sampler_info,
                           propSize, pPropValue, pPropSizeRet))) {
    return err;
  }
  // Convert OpenCL returns to UR
  cl2urSamplerInfoValue(sampler_info, propSize, pPropValue);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRetain(ur_sampler_handle_t hSampler) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  return map_cl_error_to_ur(
      clRetainSampler(cl_adapter::cast<cl_sampler>(hSampler)));
}

UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRelease(ur_sampler_handle_t hSampler) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  return map_cl_error_to_ur(
      clReleaseSampler(cl_adapter::cast<cl_sampler>(hSampler)));
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler, ur_native_handle_t *phNativeSampler) {
  UR_ASSERT(hSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeSampler, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *phNativeSampler = reinterpret_cast<ur_native_handle_t>(
      cl_adapter::cast<cl_sampler>(hSampler));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urSamplerCreateWithNativeHandle(
    ur_native_handle_t hNativeSampler, ur_context_handle_t hContext,
    const ur_sampler_native_properties_t *pProperties,
    ur_sampler_handle_t *phSampler) {
  UR_ASSERT(hNativeSampler, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phSampler, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  std::ignore = hContext;
  std::ignore = pProperties;
  *phSampler = reinterpret_cast<ur_sampler_handle_t>(
      cl_adapter::cast<cl_sampler>(hNativeSampler));
  return UR_RESULT_SUCCESS;
}
