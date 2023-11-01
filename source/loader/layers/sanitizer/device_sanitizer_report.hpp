#pragma once

#include <cinttypes>

enum class DeviceSanitizerErrorType : int32_t {
  OUT_OF_BOUND,
  MISALIGNED,
  USE_AFTER_FREE,
  OUT_OF_SHADOW_BOUND,
  UNKNOWN
};

enum class DeviceSanitizerMemoryType : int32_t {
  USM_DEVICE,
  USM_HOST,
  USM_SHARED,
  LOCAL,
  PRIVATE,
  UNKNOWN
};

// NOTE Layout of this structure should be aligned with the one in
// sycl/include/sycl/detail/device_sanitizer_report.hpp
struct DeviceSanitizerReport {
  int Flag = 0;

  char File[256 + 1] = "";
  char Func[128 + 1] = "";

  int32_t Line = 0;

  uint64_t GID0 = 0;
  uint64_t GID1 = 0;
  uint64_t GID2 = 0;

  uint64_t LID0 = 0;
  uint64_t LID1 = 0;
  uint64_t LID2 = 0;

  bool IsWrite = false;
  uint32_t AccessSize = 0;
  DeviceSanitizerMemoryType MemoryType;
  DeviceSanitizerErrorType ErrorType;

  bool IsRecover = false;
};

const char *DeviceSanitizerFormat(DeviceSanitizerMemoryType MemoryType) {
  switch (MemoryType) {
  case DeviceSanitizerMemoryType::USM_DEVICE:
    return "USM Device Memory";
  case DeviceSanitizerMemoryType::USM_HOST:
    return "USM Host Memory";
  case DeviceSanitizerMemoryType::USM_SHARED:
    return "USM Shared Memory";
  case DeviceSanitizerMemoryType::LOCAL:
    return "Local Memory";
  case DeviceSanitizerMemoryType::PRIVATE:
    return "Private Memory";
  default:
    return "Unknown Memory";
  }
}

const char *DeviceSanitizerFormat(DeviceSanitizerErrorType ErrorType) {
  switch (ErrorType) {
  case DeviceSanitizerErrorType::OUT_OF_BOUND:
    return "out-of-bound-access";
  case DeviceSanitizerErrorType::MISALIGNED:
    return "misaligned-access";
  case DeviceSanitizerErrorType::USE_AFTER_FREE:
    return "use-after-free";
  case DeviceSanitizerErrorType::OUT_OF_SHADOW_BOUND:
    return "out-of-shadow-bound-access";
  default:
    return "unknown-error";
  }
}
