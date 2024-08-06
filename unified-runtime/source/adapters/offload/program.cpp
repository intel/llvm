#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>
#include <cuda.h>

#include "context.hpp"
#include "program.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, uint32_t numDevices,
    ur_device_handle_t *phDevices, size_t *pLengths, const uint8_t **ppBinaries,
    const ur_program_properties_t *, ur_program_handle_t *phProgram) {
  if (numDevices > 1) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  // Workaround for Offload not supporting PTX binaries. Force CUDA programs
  // to be linked so they end up as CUBIN.
  uint8_t *RealBinary;
  size_t RealLength;
  ur_platform_handle_t DevicePlatform;
  bool DidLink = false;
  CUlinkState State;
  urDeviceGetInfo(phDevices[0], UR_DEVICE_INFO_PLATFORM,
                  sizeof(ur_platform_handle_t), &DevicePlatform, nullptr);
  ur_backend_t PlatformBackend;
  urPlatformGetInfo(DevicePlatform, UR_PLATFORM_INFO_BACKEND,
                    sizeof(ur_backend_t), &PlatformBackend, nullptr);
  if (PlatformBackend == UR_BACKEND_CUDA) {
    cuLinkCreate(0, nullptr, nullptr, &State);

    cuLinkAddData(State, CU_JIT_INPUT_PTX, (char *)(ppBinaries[0]), pLengths[0],
                  nullptr, 0, nullptr, nullptr);

    void *CuBin = nullptr;
    size_t CuBinSize = 0;
    cuLinkComplete(State, &CuBin, &CuBinSize);
    RealBinary = (uint8_t *)CuBin;
    RealLength = CuBinSize;
    DidLink = true;
    fprintf(stderr, "Performed CUDA bin workaround (size = %lu)\n", RealLength);
  } else {
    RealBinary = const_cast<uint8_t *>(ppBinaries[0]);
    RealLength = pLengths[0];
  }

  ur_program_handle_t Program = new ur_program_handle_t_();
  auto Res =
      olCreateProgram(reinterpret_cast<ol_device_handle_t>(hContext->Device),
                      RealBinary, RealLength, &Program->OffloadProgram);

  // Program owns the linked module now
  if (DidLink) {
    cuLinkDestroy(State);
  }

  if (Res != OL_SUCCESS) {
    delete Program;
    return offloadResultToUR(Res);
  }

  *phProgram = Program;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(ur_context_handle_t,
                                                   ur_program_handle_t,
                                                   const char *) {
  // Do nothing, program is built upon creation
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuildExp(ur_program_handle_t,
                                                      uint32_t,
                                                      ur_device_handle_t *,
                                                      const char *) {
  // Do nothing, program is built upon creation
  return UR_RESULT_SUCCESS;
}


UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(ur_program_handle_t hProgram) {
  hProgram->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(ur_program_handle_t hProgram) {
  if (--hProgram->RefCount == 0) {
    auto Res = olDestroyProgram(hProgram->OffloadProgram);
    if (Res) {
      return offloadResultToUR(Res);
    }
    delete hProgram;
  }

  return UR_RESULT_SUCCESS;
}
