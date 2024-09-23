// REQUIRES: gpu

// This test is written only for Intel architectures. It is expected that this
// test will fail on NVIDIA and AMD as the checks for ext_oneapi_architecture_is
// host API expect that device architecture is Intel GPU
// UNSUPPORTED: cuda, hip

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();

  assert(dev.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu));
  assert(!dev.ext_oneapi_architecture_is(syclex::arch_category::nvidia_gpu));
  assert(!dev.ext_oneapi_architecture_is(syclex::arch_category::amd_gpu));

  syclex::architecture intel_gpu_arch = syclex::architecture::intel_gpu_ehl;
  assert(intel_gpu_arch < syclex::architecture::intel_gpu_pvc);
  assert(intel_gpu_arch <= syclex::architecture::intel_gpu_pvc);
  assert(intel_gpu_arch > syclex::architecture::intel_gpu_skl);
  assert(intel_gpu_arch >= syclex::architecture::intel_gpu_ehl);

  syclex::architecture nvidia_gpu_arch = syclex::architecture::nvidia_gpu_sm_70;
  assert(nvidia_gpu_arch < syclex::architecture::nvidia_gpu_sm_80);
  assert(nvidia_gpu_arch <= syclex::architecture::nvidia_gpu_sm_80);
  assert(nvidia_gpu_arch > syclex::architecture::nvidia_gpu_sm_53);
  assert(nvidia_gpu_arch >= syclex::architecture::nvidia_gpu_sm_70);

  syclex::architecture amd_gpu_arch = syclex::architecture::amd_gpu_gfx908;
  assert(amd_gpu_arch < syclex::architecture::amd_gpu_gfx1031);
  assert(amd_gpu_arch <= syclex::architecture::amd_gpu_gfx1031);
  assert(amd_gpu_arch > syclex::architecture::amd_gpu_gfx810);
  assert(amd_gpu_arch >= syclex::architecture::amd_gpu_gfx908);

  return 0;
}
