// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Currently grf_size property can take value 256 (large) on PVC and DG2:
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_intel_grf_size.asciidoc
// REQUIRES: gpu && (gpu-intel-pvc || gpu-intel-dg2)
// UNSUPPORTED: cuda || hip

// clang-format off
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
// clang-format on

using namespace sycl;

// Test that kernel can be submitted with work group size returned by
// info::kernel_device_specific::work_group_size when large register file is
// used.

class MyKernel;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

__attribute__((noinline)) void f(int *result, nd_item<1> &index) {
  result[index.get_global_id()] = index.get_global_id();
}

int main() {
  queue myQueue;
  auto myContext = myQueue.get_context();
  auto myDev = myQueue.get_device();

  kernel_id kernelId = get_kernel_id<MyKernel>();
  auto myBundle =
      get_kernel_bundle<bundle_state::executable>(myContext, {kernelId});

  kernel myKernel = myBundle.get_kernel(kernelId);
  size_t maxWgSize =
      myKernel.get_info<info::kernel_device_specific::work_group_size>(myDev);

  // Submit kernel with maximum work group size.
  nd_range myRange{range{maxWgSize}, range{maxWgSize}};

  int *result = sycl::malloc_shared<int>(maxWgSize, myQueue);
  syclex::properties kernelProperties{intelex::grf_size<256>};
  myQueue.submit([&](handler &cgh) {
    cgh.use_kernel_bundle(myBundle);
    cgh.parallel_for<MyKernel>(myRange, kernelProperties,
                               ([=](nd_item<1> index) { f(result, index); }));
  });

  myQueue.wait();
  free(result, myQueue);
  return 0;
}
