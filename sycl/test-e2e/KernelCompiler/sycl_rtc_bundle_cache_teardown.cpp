//==-- sycl_rtc_bundle_cache_teardown.cpp - RTC cache teardown regression --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: sycl-jit

// RUN: %{build} -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out

#include <cstdio>
#include <memory>
#include <vector>

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr const char *KernelSrc = R"===(
#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::single_task_kernel))
void fft_kernel_512_fwd() {}
)===";

class FftKernelBundleCache {
public:
  using bundle_t = sycl::kernel_bundle<sycl::bundle_state::executable>;

  static FftKernelBundleCache &instance() {
    static FftKernelBundleCache Cache;
    return Cache;
  }

  void add(std::shared_ptr<bundle_t> Bundle) {
    MBundles.emplace_back(std::move(Bundle));
  }

private:
  std::vector<std::shared_ptr<bundle_t>> MBundles;
};

static std::shared_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>>
buildBundle(sycl::queue &Q, const char *BuildOpt) {
  auto SrcBundle = syclexp::create_kernel_bundle_from_source(
      Q.get_context(), syclexp::source_language::sycl, KernelSrc);
  auto ExeBundle = syclexp::build(
      SrcBundle, syclexp::properties{syclexp::build_options{BuildOpt}});

  return std::make_shared<sycl::kernel_bundle<sycl::bundle_state::executable>>(
      std::move(ExeBundle));
}

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  sycl::queue Q;

  if (!Q.get_device().ext_oneapi_can_build(syclexp::source_language::sycl)) {
    // Skipping test then.
    return 0;
  }

  auto &Cache = FftKernelBundleCache::instance();
  Cache.add(buildBundle(Q, "-DSIZE=512"));
  Cache.add(buildBundle(Q, "-DSIZE=768"));
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
