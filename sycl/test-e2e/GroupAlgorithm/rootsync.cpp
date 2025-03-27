// Fails with opencl non-cpu, enable when fixed.
// XFAIL: (opencl && !cpu && !accelerator)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/14641

// TODO: Currently using the -Wno-deprecated-declarations flag due to issue
// https://github.com/intel/llvm/issues/16451. Rewrite testRootGroup() amd
// remove the flag once the issue is resolved.
// RUN: %{build} -I . -o %t.out -Wno-deprecated-declarations %if target-nvidia %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 %}
// RUN: %{run} %t.out

// Disabled temporarily while investigation into the failure is ongoing.

#include <cstdlib>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/kernel_bundle.hpp>
#include <type_traits>
struct RootGroupKernel {
  RootGroupKernel() {}
  void operator()(sycl::nd_item<1> it) const {
    auto root = it.ext_oneapi_get_root_group();
    sycl::group_barrier(root);
  }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
  }
};
int main() {
  sycl::queue q;
  sycl::range<1> R1{1};
  sycl::nd_range<1> NDR1{R1, R1};
  q.submit([&](sycl::handler &h) { h.parallel_for(NDR1, RootGroupKernel()); });
  return EXIT_SUCCESS;
}