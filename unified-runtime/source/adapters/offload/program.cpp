#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "context.hpp"
#include "device.hpp"
#include "program.hpp"
#include "ur2offload.hpp"

#ifdef UR_CUDA_ENABLED
#include <cuda.h>
#endif

namespace {
// Workaround for Offload not supporting PTX binaries. Force CUDA programs
// to be linked so they end up as CUBIN.
#ifdef UR_CUDA_ENABLED
ur_result_t ProgramCreateCudaWorkaround(ur_context_handle_t hContext,
                                        const uint8_t *Binary, size_t Length,
                                        ur_program_handle_t *phProgram) {
  uint8_t *RealBinary;
  size_t RealLength;
  CUlinkState State;
  cuLinkCreate(0, nullptr, nullptr, &State);

  cuLinkAddData(State, CU_JIT_INPUT_PTX, (char *)(Binary), Length, nullptr, 0,
                nullptr, nullptr);

  void *CuBin = nullptr;
  size_t CuBinSize = 0;
  cuLinkComplete(State, &CuBin, &CuBinSize);
  RealBinary = (uint8_t *)CuBin;
  RealLength = CuBinSize;
  fprintf(stderr, "Performed CUDA bin workaround (size = %lu)\n", RealLength);

  ur_program_handle_t Program = new ur_program_handle_t_();
  auto Res = olCreateProgram(hContext->Device->OffloadDevice, RealBinary,
                             RealLength, &Program->OffloadProgram);

  // Program owns the linked module now
  cuLinkDestroy(State);
  (void)State;

  if (Res != OL_SUCCESS) {
    delete Program;
    return offloadResultToUR(Res);
  }

  *phProgram = Program;

  return UR_RESULT_SUCCESS;
}
#else
ur_result_t ProgramCreateCudaWorkaround(ur_context_handle_t, const uint8_t *,
                                        size_t, ur_program_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
#endif

// https://clang.llvm.org/docs/ClangOffloadBundler.html#bundled-binary-file-layout
class HipOffloadBundleParser {
  static constexpr std::string_view Magic = "__CLANG_OFFLOAD_BUNDLE__";
  const uint8_t *Buff;
  size_t Length;

  struct __attribute__((packed)) BundleEntry {
    uint64_t ObjectOffset;
    uint64_t ObjectSize;
    uint64_t EntryIdSize;
    char EntryIdStart;
  };

  struct __attribute__((packed)) BundleHeader {
    const char HeaderMagic[Magic.size()];
    uint64_t EntryCount;
    BundleEntry FirstEntry;
  };

  HipOffloadBundleParser() = delete;
  HipOffloadBundleParser(const uint8_t *Buff, size_t Length)
      : Buff(Buff), Length(Length) {}

public:
  static std::optional<HipOffloadBundleParser> load(const uint8_t *Buff,
                                                    size_t Length) {
    if (std::string_view{reinterpret_cast<const char *>(Buff), Length}.find(
            Magic) != 0) {
      return std::nullopt;
    }
    return HipOffloadBundleParser(Buff, Length);
  }

  ur_result_t extract(std::string_view SearchTargetId,
                      const uint8_t *&OutBinary, size_t &OutLength) {
    const char *Limit = reinterpret_cast<const char *>(&Buff[Length]);

    // The different check here means that a binary consisting of only the magic
    // bytes (but nothing else) will result in INVALID_PROGRAM rather than being
    // treated as a non-bundle
    auto *Header = reinterpret_cast<const BundleHeader *>(Buff);
    if (reinterpret_cast<const char *>(&Header->FirstEntry) > Limit) {
      return UR_RESULT_ERROR_INVALID_PROGRAM;
    }

    const auto *CurrentEntry = &Header->FirstEntry;
    for (uint64_t I = 0; I < Header->EntryCount; I++) {
      if (&CurrentEntry->EntryIdStart > Limit) {
        return UR_RESULT_ERROR_INVALID_PROGRAM;
      }
      auto EntryId = std::string_view(&CurrentEntry->EntryIdStart,
                                      CurrentEntry->EntryIdSize);
      if (EntryId.end() > Limit) {
        return UR_RESULT_ERROR_INVALID_PROGRAM;
      }

      // Will match either "hip" or "hipv4"
      bool isHip = EntryId.find("hip") == 0;
      bool VersionMatches =
          EntryId.find_last_of(SearchTargetId) == EntryId.size() - 1;

      if (isHip && VersionMatches) {
        OutBinary = reinterpret_cast<const uint8_t *>(
            &Buff[CurrentEntry->ObjectOffset]);
        OutLength = CurrentEntry->ObjectSize;

        if (reinterpret_cast<const char *>(&OutBinary[OutLength]) > Limit) {
          return UR_RESULT_ERROR_INVALID_PROGRAM;
        }
        return UR_RESULT_SUCCESS;
      }

      CurrentEntry = reinterpret_cast<const BundleEntry *>(EntryId.end());
    }

    return UR_RESULT_ERROR_INVALID_PROGRAM;
  }
};

} // namespace

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, uint32_t numDevices,
    ur_device_handle_t *phDevices, size_t *pLengths, const uint8_t **ppBinaries,
    const ur_program_properties_t *, ur_program_handle_t *phProgram) {
  if (numDevices > 1) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ur_platform_handle_t DevicePlatform;
  urDeviceGetInfo(phDevices[0], UR_DEVICE_INFO_PLATFORM,
                  sizeof(ur_platform_handle_t), &DevicePlatform, nullptr);
  ur_backend_t PlatformBackend;
  urPlatformGetInfo(DevicePlatform, UR_PLATFORM_INFO_BACKEND,
                    sizeof(ur_backend_t), &PlatformBackend, nullptr);

  auto *RealBinary = ppBinaries[0];
  size_t RealLength = pLengths[0];

  if (auto Parser = HipOffloadBundleParser::load(RealBinary, RealLength)) {
    std::string DevName{};
    size_t DevNameLength;
    olGetDeviceInfoSize(phDevices[0]->OffloadDevice, OL_DEVICE_INFO_NAME,
                        &DevNameLength);
    DevName.resize(DevNameLength);
    olGetDeviceInfo(phDevices[0]->OffloadDevice, OL_DEVICE_INFO_NAME,
                    DevNameLength, DevName.data());

    auto Res = Parser->extract(DevName, RealBinary, RealLength);
    if (Res != UR_RESULT_SUCCESS) {
      return Res;
    }
  }

  if (PlatformBackend == UR_BACKEND_CUDA) {
    return ProgramCreateCudaWorkaround(hContext, RealBinary, RealLength,
                                       phProgram);
  }

  ur_program_handle_t Program = new ur_program_handle_t_();
  auto Res = olCreateProgram(hContext->Device->OffloadDevice, RealBinary,
                             RealLength, &Program->OffloadProgram);

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
