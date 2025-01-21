#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

namespace syclext = sycl::ext::oneapi::experimental;

// Find the least common multiple of the context and device granularities. This
// value can be used for aligning both physical memory allocations and for
// reserving virtual memory ranges.
size_t GetLCMGranularity(
    const sycl::device &Dev, const sycl::context &Ctx,
    syclext::granularity_mode Gm = syclext::granularity_mode::recommended) {
  size_t CtxGranularity = syclext::get_mem_granularity(Ctx, Gm);
  size_t DevGranularity = syclext::get_mem_granularity(Dev, Ctx, Gm);

  size_t GCD = CtxGranularity;
  size_t Rem = DevGranularity % GCD;
  while (Rem != 0) {
    std::swap(GCD, Rem);
    Rem %= GCD;
  }
  return (DevGranularity / GCD) * CtxGranularity;
}

size_t GetAlignedByteSize(const size_t UnalignedBytes,
                          const size_t AligmentGranularity) {
  return ((UnalignedBytes + AligmentGranularity - 1) / AligmentGranularity) *
         AligmentGranularity;
}
