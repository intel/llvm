// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks that get_kernel_ids returns all the kernels defined
// in the source regardless of whether they are expressed as lambdas, named
// function objects or free function kernels.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/kernel_bundle.hpp>

class FunctionObjectKernel {
public:
  void operator()(sycl::item<1> item) const {}
};

class LambdaKernel;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void FreeFunctionKernel() {}

int main() {
  sycl::queue Queue;
  sycl::device Dev = Queue.get_device();
  sycl::context Context = Queue.get_context();
  Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<LambdaKernel>([=]() {}); });
  Queue.submit([&](sycl::handler &CGH) {
    FunctionObjectKernel kernel;
    CGH.parallel_for(sycl::range<1>{}, kernel);
  });

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Context, {Dev});
  std::vector<sycl::kernel_id> KernelIds = KernelBundle.get_kernel_ids();
  assert(KernelIds.size() == 3);
  sycl::kernel_id FunctionObjectKernelId =
      sycl::get_kernel_id<FunctionObjectKernel>();
  sycl::kernel_id LambdaKernelId = sycl::get_kernel_id<LambdaKernel>();
  sycl::kernel_id FreeFunctionKernelId =
      sycl::ext::oneapi::experimental::get_kernel_id<FreeFunctionKernel>();
  for (const sycl::kernel_id &KernelId :
       {FunctionObjectKernelId, LambdaKernelId, FreeFunctionKernelId}) {
    auto FoundId = std::find(KernelIds.begin(), KernelIds.end(), KernelId);
    assert(FoundId != KernelIds.end());
  }
}
