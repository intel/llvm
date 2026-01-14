// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks that get_kernel_ids returns all the kernels defined
// in the source regardless of whether they are expressed as lambdas,
// function objects or free functions.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/kernel_bundle.hpp>

class FunctionObjectKernel {
public:
  void operator()() const {}
};

class LambdaKernel;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void FreeFunctionKernel() {}

int main() {
  sycl::queue Queue;
  sycl::device Dev = Queue.get_device();
  sycl::context Context = Queue.get_context();

  sycl::kernel_id FunctionObjectKernelId =
      sycl::get_kernel_id<FunctionObjectKernel>();
  sycl::kernel_id LambdaKernelId = sycl::get_kernel_id<LambdaKernel>();
  sycl::kernel_id FreeFunctionKernelId =
      sycl::ext::oneapi::experimental::get_kernel_id<FreeFunctionKernel>();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Context, {Dev});
  sycl::kernel FreeFunction = KernelBundle.get_kernel(FreeFunctionKernelId);
  Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<LambdaKernel>([=]() {}); });
  Queue.submit([&](sycl::handler &CGH) {
    FunctionObjectKernel kernel;
    CGH.single_task(kernel);
  });
  Queue.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::nd_range{{1}, {1}}, FreeFunction);
  });

  std::vector<sycl::kernel_id> KernelIds = KernelBundle.get_kernel_ids();
  assert(KernelIds.size() == 3);
  for (const sycl::kernel_id &KernelId :
       {FunctionObjectKernelId, LambdaKernelId, FreeFunctionKernelId}) {
    auto FoundId = std::find(KernelIds.begin(), KernelIds.end(), KernelId);
    assert(FoundId != KernelIds.end());
  }
  return 0;
}
