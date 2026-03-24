//===-- FP8 builtin helpers, mocks and stubs for float_8bit/types.hpp
//---------*- C++ -*-===//

#pragma once

#include <cstdint>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>

// Force code path that uses helpers.hpp wrappers.
#ifndef __SYCL_DEVICE_ONLY__
#define __SYCL_DEVICE_ONLY__ 1
#endif

namespace fp8_builtin_mock {

struct Counters {
  int ConvertE4M3ToFP16EXT = 0;
  int ConvertE5M2ToFP16EXT = 0;
  int ConvertE4M3ToBF16EXT = 0;
  int ConvertE5M2ToBF16EXT = 0;
  int ClampConvertFP16ToE4M3INTEL = 0;
  int ClampConvertBF16ToE4M3INTEL = 0;
  int ConvertFP16ToE4M3EXT = 0;
  int ConvertBF16ToE4M3EXT = 0;
  int ClampConvertFP16ToE5M2INTEL = 0;
  int ClampConvertBF16ToE5M2INTEL = 0;
  int ConvertFP16ToE5M2EXT = 0;
  int ConvertBF16ToE5M2EXT = 0;
  int StochasticRoundFP16ToE5M2INTEL = 0;
  int StochasticRoundBF16ToE5M2INTEL = 0;
  int ClampStochasticRoundFP16ToE5M2INTEL = 0;
  int ClampStochasticRoundBF16ToE5M2INTEL = 0;
};

inline Counters &getCounters() {
  static Counters Value;
  return Value;
}

inline void resetCounters() { getCounters() = Counters{}; }

} // namespace fp8_builtin_mock

// Builtin mocks (do not replace helpers.hpp; provide symbols here).
inline sycl::half
__builtin_spirv_ConvertE4M3ToFP16EXT(uint8_t) noexcept {
  ++fp8_builtin_mock::getCounters().ConvertE4M3ToFP16EXT;
  return static_cast<sycl::half>(2.0f);
}

inline sycl::half __builtin_spirv_ConvertE5M2ToFP16EXT(uint8_t) noexcept {
  ++fp8_builtin_mock::getCounters().ConvertE5M2ToFP16EXT;
  return static_cast<sycl::half>(3.0f);
}

inline sycl::ext::oneapi::bfloat16
__builtin_spirv_ConvertE4M3ToBF16EXT(uint8_t) noexcept {
  ++fp8_builtin_mock::getCounters().ConvertE4M3ToBF16EXT;
  return static_cast<sycl::ext::oneapi::bfloat16>(4.0f);
}

inline sycl::ext::oneapi::bfloat16
__builtin_spirv_ConvertE5M2ToBF16EXT(uint8_t) noexcept {
  ++fp8_builtin_mock::getCounters().ConvertE5M2ToBF16EXT;
  return static_cast<sycl::ext::oneapi::bfloat16>(5.0f);
}

inline uint8_t __builtin_spirv_ConvertFP16ToE4M3EXT(sycl::half) noexcept {
  ++fp8_builtin_mock::getCounters().ConvertFP16ToE4M3EXT;
  return 0x01;
}

inline uint8_t
__builtin_spirv_ConvertBF16ToE4M3EXT(sycl::ext::oneapi::bfloat16) noexcept {
  ++fp8_builtin_mock::getCounters().ConvertBF16ToE4M3EXT;
  return 0x02;
}

inline uint8_t
__builtin_spirv_ClampConvertFP16ToE4M3INTEL(sycl::half) noexcept {
  ++fp8_builtin_mock::getCounters().ClampConvertFP16ToE4M3INTEL;
  return 0x11;
}

inline uint8_t __builtin_spirv_ClampConvertBF16ToE4M3INTEL(
    sycl::ext::oneapi::bfloat16) noexcept {
  ++fp8_builtin_mock::getCounters().ClampConvertBF16ToE4M3INTEL;
  return 0x12;
}

inline uint8_t __builtin_spirv_ConvertFP16ToE5M2EXT(sycl::half) noexcept {
  ++fp8_builtin_mock::getCounters().ConvertFP16ToE5M2EXT;
  return 0x03;
}

inline uint8_t
__builtin_spirv_ClampConvertFP16ToE5M2INTEL(sycl::half) noexcept {
  ++fp8_builtin_mock::getCounters().ClampConvertFP16ToE5M2INTEL;
  return 0x21;
}

inline uint8_t
__builtin_spirv_ConvertBF16ToE5M2EXT(sycl::ext::oneapi::bfloat16) noexcept {
  ++fp8_builtin_mock::getCounters().ConvertBF16ToE5M2EXT;
  return 0x04;
}

inline uint8_t __builtin_spirv_ClampConvertBF16ToE5M2INTEL(
    sycl::ext::oneapi::bfloat16) noexcept {
  ++fp8_builtin_mock::getCounters().ClampConvertBF16ToE5M2INTEL;
  return 0x22;
}

inline uint8_t
__builtin_spirv_StochasticRoundFP16ToE5M2INTEL(sycl::half, uint32_t Seed,
                                               uint32_t *NextSeed) noexcept {
  ++fp8_builtin_mock::getCounters().StochasticRoundFP16ToE5M2INTEL;
  if (NextSeed)
    *NextSeed = Seed + 1;
  return 0x31;
}

inline uint8_t
__builtin_spirv_StochasticRoundFP16ToE4M3INTEL(sycl::half) noexcept {
  return 0x00;
}

inline uint8_t __builtin_spirv_StochasticRoundBF16ToE5M2INTEL(
    sycl::ext::oneapi::bfloat16, uint32_t Seed, uint32_t *NextSeed) noexcept {
  ++fp8_builtin_mock::getCounters().StochasticRoundBF16ToE5M2INTEL;
  if (NextSeed)
    *NextSeed = Seed + 1;
  return 0x32;
}

inline uint8_t __builtin_spirv_StochasticRoundBF16ToE4M3INTEL(
    sycl::ext::oneapi::bfloat16) noexcept {
  return 0x00;
}

inline uint8_t __builtin_spirv_ClampStochasticRoundFP16ToE5M2INTEL(
    sycl::half, uint32_t Seed, uint32_t *NextSeed) noexcept {
  ++fp8_builtin_mock::getCounters().ClampStochasticRoundFP16ToE5M2INTEL;
  if (NextSeed)
    *NextSeed = Seed + 1;
  return 0x41;
}

inline uint8_t
__builtin_spirv_ClampStochasticRoundFP16ToE4M3INTEL(sycl::half) noexcept {
  return 0x00;
}

inline uint8_t __builtin_spirv_ClampStochasticRoundBF16ToE5M2INTEL(
    sycl::ext::oneapi::bfloat16, uint32_t Seed, uint32_t *NextSeed) noexcept {
  ++fp8_builtin_mock::getCounters().ClampStochasticRoundBF16ToE5M2INTEL;
  if (NextSeed)
    *NextSeed = Seed + 1;
  return 0x42;
}

inline uint8_t __builtin_spirv_ClampStochasticRoundBF16ToE4M3INTEL(
    sycl::ext::oneapi::bfloat16) noexcept {
  return 0x00;
}
