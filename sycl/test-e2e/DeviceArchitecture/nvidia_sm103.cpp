// REQUIRES: cuda
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <string>
#include <sycl/detail/core.hpp>

int main() {
  namespace syclex = sycl::ext::oneapi::experimental;
  sycl::device dev{sycl::gpu_selector_v};
  std::string version = dev.get_info<sycl::info::device::version>();
  if (version.substr(0, version.find(':')) != "10.3")
    return 0;
  auto arch = dev.get_info<syclex::info::device::architecture>();
  assert(arch == syclex::architecture::nvidia_gpu_sm_103);
  assert(dev.ext_oneapi_architecture_is(syclex::arch_category::nvidia_gpu));
}
