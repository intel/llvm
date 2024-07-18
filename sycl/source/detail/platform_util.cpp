//===-- platform_util.cpp - Platform utilities implementation --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/platform_util.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/exception.hpp>

#if defined(__SYCL_RT_OS_LINUX)
#include <errno.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#elif defined(__SYCL_RT_OS_WINDOWS)
#include <intrin.h>
#elif defined(__SYCL_RT_OS_DARWIN)
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

#if defined(__x86_64__) || defined(__i386__)
// Used by methods that duplicate OpenCL behaviour in order to get CPU info
static void cpuid(uint32_t *CPUInfo, uint32_t Type, uint32_t SubType = 0) {
#if defined(__SYCL_RT_OS_LINUX) || defined(__SYCL_RT_OS_DARWIN)
  __cpuid_count(Type, SubType, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
#elif defined(__SYCL_RT_OS_WINDOWS)
  __cpuidex(reinterpret_cast<int *>(CPUInfo), Type, SubType);
#endif
}
#endif

uint32_t PlatformUtil::getMaxClockFrequency() {
  throw exception(make_error_code(errc::runtime),
                  "max_clock_frequency parameter is not supported on host");
  return 0;
}

uint32_t PlatformUtil::getMemCacheLineSize() {
#if defined(__x86_64__) || defined(__i386__)
  uint32_t CPUInfo[4];
  cpuid(CPUInfo, 0x80000006);
  return CPUInfo[2] & 0xff;
#elif defined(__SYCL_RT_OS_LINUX) && defined(_SC_LEVEL2_DCACHE_LINESIZE)
  long lineSize = sysconf(_SC_LEVEL2_DCACHE_LINESIZE);
  if (lineSize > 0) {
    return lineSize;
  }
#endif
  return 8;
}

uint64_t PlatformUtil::getMemCacheSize() {
#if defined(__x86_64__) || defined(__i386__)
  uint32_t CPUInfo[4];
  cpuid(CPUInfo, 0x80000006);
  return static_cast<uint64_t>(CPUInfo[2] >> 16) * 1024;
#elif defined(__SYCL_RT_OS_LINUX) && defined(_SC_LEVEL2_DCACHE_SIZE)
  long cacheSize = sysconf(_SC_LEVEL2_DCACHE_SIZE);
  if (cacheSize > 0) {
    return cacheSize;
  }
#endif
  return static_cast<uint64_t>(16 * 1024);
}

uint32_t PlatformUtil::getNativeVectorWidth(PlatformUtil::TypeIndex TIndex) {

#if defined(__x86_64__) || defined(__i386__)
  uint32_t Index = static_cast<uint32_t>(TIndex);

  // SSE4.2 has 16 byte (XMM) registers
  static constexpr uint32_t VECTOR_WIDTH_SSE42[] = {16, 8, 4, 2, 4, 2, 0};
  // AVX supports 32 byte (YMM) registers only for floats and doubles
  static constexpr uint32_t VECTOR_WIDTH_AVX[] = {16, 8, 4, 2, 8, 4, 0};
  // AVX2 has a full set of 32 byte (YMM) registers
  static constexpr uint32_t VECTOR_WIDTH_AVX2[] = {32, 16, 8, 4, 8, 4, 0};
  // AVX512 has 64 byte (ZMM) registers
  static constexpr uint32_t VECTOR_WIDTH_AVX512[] = {64, 32, 16, 8, 16, 8, 0};

#if defined(__SYCL_RT_OS_LINUX) || defined(__SYCL_RT_OS_DARWIN)
  if (__builtin_cpu_supports("avx512f"))
    return VECTOR_WIDTH_AVX512[Index];
  if (__builtin_cpu_supports("avx2"))
    return VECTOR_WIDTH_AVX2[Index];
  if (__builtin_cpu_supports("avx"))
    return VECTOR_WIDTH_AVX[Index];
#elif defined(__SYCL_RT_OS_WINDOWS)

  uint32_t Info[4];

  // Check that CPUID func number 7 is available.
  cpuid(Info, 0);
  if (Info[0] >= 7) {
    // avx512f = CPUID.7.EBX[16]
    cpuid(Info, 7);
    if (Info[1] & (1 << 16))
      return VECTOR_WIDTH_AVX512[Index];

    // avx2 = CPUID.7.EBX[5]
    if (Info[1] & (1 << 5))
      return VECTOR_WIDTH_AVX2[Index];
  }
  // It is assumed that CPUID func number 1 is always available.
  // avx = CPUID.1.ECX[28]
  cpuid(Info, 1);
  if (Info[2] & (1 << 28))
    return VECTOR_WIDTH_AVX[Index];
#endif

  return VECTOR_WIDTH_SSE42[Index];

#elif defined(__ARM_NEON)
  uint32_t Index = static_cast<uint32_t>(TIndex);

  // NEON has 16 byte registers
  static constexpr uint32_t VECTOR_WIDTH_NEON[] = {16, 8, 4, 2, 4, 2, 0};
  return VECTOR_WIDTH_NEON[Index];

#endif
  return 0;
}

void PlatformUtil::prefetch(const char *Ptr, size_t NumBytes) {
  if (!Ptr)
    return;

  const size_t CacheLineSize = PlatformUtil::getMemCacheLineSize();
  const size_t CacheLineMask = ~(CacheLineSize - 1);
  const char *PtrEnd = Ptr + NumBytes;

  // Set the pointer to the beginning of the current cache line.
  Ptr = reinterpret_cast<const char *>(reinterpret_cast<size_t>(Ptr) &
                                       CacheLineMask);
  for (; Ptr < PtrEnd; Ptr += CacheLineSize) {
#if defined(__SYCL_RT_OS_LINUX)
    __builtin_prefetch(Ptr);
#elif defined(__SYCL_RT_OS_WINDOWS)
    _mm_prefetch(Ptr, _MM_HINT_T0);
#endif
  }
}

} // namespace detail
} // namespace _V1
} // namespace sycl
