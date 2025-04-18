// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test has_2d_block_io_supported device descriptor for some known
// architectures.

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue Q;
  auto Arch = Q.get_device().get_info<syclex::info::device::architecture>();
  bool Has2DBlockIOSupport =
      Q.get_device()
          .get_info<
              sycl::ext::intel::esimd::info::device::has_2d_block_io_support>();
  if (Arch == syclex::architecture::intel_gpu_pvc) {
    if (!Has2DBlockIOSupport) {
      std::cerr << "Error: has_2d_block_io_support is expected to be true for "
                   "PVC architecture"
                << std::endl;
      return 1;
    }
  }
  if (Arch == syclex::architecture::intel_gpu_tgllp ||
      Arch == syclex::architecture::intel_gpu_dg2_g10 ||
      Arch == syclex::architecture::intel_gpu_dg2_g11 ||
      Arch == syclex::architecture::intel_gpu_dg2_g12) {
    if (Has2DBlockIOSupport) {
      std::cerr << "Error: has_2d_block_io_support is expected to be false for "
                   "Tiger Lake and DG2"
                << std::endl;
      return 1;
    }
  }
  return 0;
}
