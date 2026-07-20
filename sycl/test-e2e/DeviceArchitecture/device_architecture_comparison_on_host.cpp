// REQUIRES: gpu

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();

  bool isArchKnown = dev.get_info<syclex::info::device::architecture>() !=
                     syclex::architecture::unknown;

  // If architecture is known, it must be part of one, and only one, of the
  // known device architecture categories.
  if (isArchKnown) {
    int32_t isArchIntel =
        dev.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
    int32_t isArchNvidia =
        dev.ext_oneapi_architecture_is(syclex::arch_category::nvidia_gpu);
    int32_t isArchAmd =
        dev.ext_oneapi_architecture_is(syclex::arch_category::amd_gpu);
    int32_t isArchQcom =
        dev.ext_oneapi_architecture_is(syclex::arch_category::qcom_gpu);
    assert((isArchIntel + isArchNvidia + isArchAmd + isArchQcom) == 1);
  }

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

  syclex::architecture qcom_gpu_arch = syclex::architecture::qcom_gpu_x2_85;
  assert(qcom_gpu_arch < syclex::architecture::qcom_gpu_x2_90);
  assert(qcom_gpu_arch <= syclex::architecture::qcom_gpu_x2_85);
  assert(qcom_gpu_arch > syclex::architecture::qcom_gpu_x1_85);
  assert(qcom_gpu_arch >= syclex::architecture::qcom_gpu_x1_85);

  return 0;
}
