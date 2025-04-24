#include <OffloadAPI.h>
#include <unordered_set>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "adapter.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(ur_adapter_handle_t, uint32_t NumEntries,
              ur_platform_handle_t *phPlatforms, uint32_t *pNumPlatforms) {

  if (pNumPlatforms) {
    *pNumPlatforms = Adapter.Platforms.size();
  }

  if (phPlatforms) {
    size_t PlatformIndex = 0;
    for (auto &Platform : Adapter.Platforms) {
      phPlatforms[PlatformIndex++] =
          reinterpret_cast<ur_platform_handle_t>(Platform);
      if (PlatformIndex == NumEntries) {
        break;
      }
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetInfo(ur_platform_handle_t hPlatform, ur_platform_info_t propName,
                  size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  ol_platform_info_t olInfo;
  switch (propName) {
  case UR_PLATFORM_INFO_NAME:
    olInfo = OL_PLATFORM_INFO_NAME;
    break;
  case UR_PLATFORM_INFO_VENDOR_NAME:
    olInfo = OL_PLATFORM_INFO_VENDOR_NAME;
    break;
  case UR_PLATFORM_INFO_VERSION:
    olInfo = OL_PLATFORM_INFO_VERSION;
    break;
  case UR_PLATFORM_INFO_EXTENSIONS:
    return ReturnValue("");
  case UR_PLATFORM_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");
  case UR_PLATFORM_INFO_BACKEND:
    return ReturnValue(UR_BACKEND_OFFLOAD);
    break;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  if (pPropSizeRet) {
    if (auto Res = olGetPlatformInfoSize(
            reinterpret_cast<ol_platform_handle_t>(hPlatform), olInfo,
            pPropSizeRet)) {
      return offloadResultToUR(Res);
    }
  }

  if (pPropValue) {
    if (auto Res =
            olGetPlatformInfo(reinterpret_cast<ol_platform_handle_t>(hPlatform),
                              olInfo, propSize, pPropValue)) {
      return offloadResultToUR(Res);
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetBackendOption(ur_platform_handle_t, const char *pFrontendOption,
                           const char **ppPlatformOption) {
  using namespace std::literals;
  if (pFrontendOption == nullptr)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  if (pFrontendOption == "-O0"sv || pFrontendOption == "-O1"sv ||
      pFrontendOption == "-O2"sv || pFrontendOption == "-O3"sv ||
      pFrontendOption == ""sv) {
    *ppPlatformOption = "";
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}
