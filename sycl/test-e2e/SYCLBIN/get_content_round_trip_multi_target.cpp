// REQUIRES: aspect-usm_device_allocations, target-spir, opencl-aot, opencl-cpu-rt

// -- End-to-end round-trip test for kernel_bundle::ext_oneapi_get_content()
// -- with a multi-target SYCLBIN. Compiles with both spir64 (JIT) and
// -- spir64_x86_64 (AOT for x86_64) targets so the resulting bundle holds
// -- multiple device images. Round-tripping must preserve all of them.

// RUN: %clangxx --offload-new-driver -fsyclbin=executable -fsycl-targets=spir64,spir64_x86_64 %S/Inputs/basic_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;
static constexpr float EPS = 0.001f;

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;
  const sycl::context Ctx = Q.get_context();
  const std::vector<sycl::device> Devs{Q.get_device()};

  auto KBOriginal = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, std::string{argv[1]});

  std::vector<char> Bytes = KBOriginal.ext_oneapi_get_content();
  if (Bytes.empty()) {
    std::cout << "ext_oneapi_get_content returned an empty vector\n";
    return 1;
  }

  auto KBReloaded = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, Devs, sycl::span<char>{Bytes});

  if (!KBReloaded.ext_oneapi_has_kernel("iota")) {
    std::cout << "Re-loaded kernel_bundle does not contain expected kernel "
                 "\"iota\"\n";
    return 1;
  }

  sycl::kernel IotaKern = KBReloaded.ext_oneapi_get_kernel("iota");

  float *Ptr = sycl::malloc_shared<float>(NUM, Q);
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(3.14f, Ptr);
     CGH.parallel_for(sycl::nd_range{{NUM}, {WGSIZE}}, IotaKern);
   }).wait_and_throw();

  int Failed = 0;
  for (size_t I = 0; I < NUM; ++I) {
    const float Truth = 3.14f + static_cast<float>(I);
    if (std::abs(Ptr[I] - Truth) > EPS) {
      std::cout << "Result[" << I << "] = " << Ptr[I] << ", expected " << Truth
                << "\n";
      ++Failed;
    }
  }
  sycl::free(Ptr, Q);
  return Failed;
}
