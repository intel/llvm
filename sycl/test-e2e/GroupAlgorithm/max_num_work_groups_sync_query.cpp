// XFAIL: (opencl && !cpu)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/14641

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// REQUIRES: aspect-usm_device_allocations
#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/experimental/kernel_execution_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/usm.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
namespace syclexintel = sycl::ext::intel::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))
void FreeFuncKernel1(int *data) {
  // Get a handle to the root-group.
  auto root = syclex::this_work_item::get_root_group<1>();

  // Write to some global memory location.
  data[root.get_local_linear_id()] = root.get_local_linear_id();

  // Synchronize all work-items executing the kernel, making all writes visible.
  sycl::group_barrier(root);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<2>))
void FreeFuncKernel2(int *data) {
  // Get a handle to the root-group.
  auto root = syclex::this_work_item::get_root_group<2>();

  // Write to some global memory location.
  data[root.get_local_linear_id()] = root.get_local_linear_id();

  // Synchronize all work-items executing the kernel, making all writes visible.
  sycl::group_barrier(root);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<3>))
void FreeFuncKernel3(int *data) {
  // Get a handle to the root-group.
  auto root = syclex::this_work_item::get_root_group<3>();

  // Write to some global memory location.
  data[root.get_local_linear_id()] = root.get_local_linear_id();

  // Synchronize all work-items executing the kernel, making all writes visible.
  sycl::group_barrier(root);

  int *dynamicLocalMem =
      reinterpret_cast<int *>(syclex::get_work_group_scratch_memory());
}

const size_t WGDimSize = 8;
const size_t DynamicLocalMem = 1024;

template <auto *Func, int Dimensions, typename Properties>
void testFreeFunctionKernel(sycl::queue &q, sycl::context &ctx,
                            sycl::device &dev,
                            sycl::range<Dimensions> WorkGroupSize,
                            Properties props, size_t bytes) {

  size_t maxWGs = syclex::get_kernel_info<
      Func,
      sycl::ext::oneapi::experimental::info::kernel::max_num_work_groups_sync>(
      q, WorkGroupSize, props, bytes);
  size_t maxWGs1 = syclex::get_kernel_info<
      Func,
      sycl::ext::oneapi::experimental::info::kernel::max_num_work_groups_sync>(
      ctx, dev, WorkGroupSize, props, bytes);
  assert(maxWGs == maxWGs1);
  // Construct an nd-range which launches the maximum number of work-groups.
  sycl::range<Dimensions> GlobalRange = WorkGroupSize;
  GlobalRange[0] *= maxWGs;
  int numItems = GlobalRange[0];
  if constexpr (Dimensions == 2)
    numItems *= GlobalRange[1];
  else if constexpr (Dimensions == 3)
    numItems *= GlobalRange[1] * GlobalRange[2];
  sycl::nd_range<Dimensions> ndr{GlobalRange, WorkGroupSize};

  int *data = sycl::malloc_device<int>(numItems, q);
  syclex::launch_config cfg{ndr, props};
  try {
    syclex::nd_launch(q, cfg, syclex::kernel_function<Func>, data);
  } catch (sycl::exception &e) {
    assert(maxWGs == 0);
  }
  sycl::free(data, ctx);
}

class LambdaKernel1;
class LambdaKernel2;
class LambdaKernel3;

template <typename KernelName, int Dimensions, typename Properties>
void testLambdaKernel(sycl::queue &q, sycl::context &ctx, sycl::device &dev,
                      sycl::range<Dimensions> WorkGroupSize, Properties props,
                      size_t bytes) {
  auto bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(ctx);
  const sycl::kernel k = bundle.template get_kernel<KernelName>();
<<<<<<< HEAD
  size_t maxWGs = syclex::get_kernel_info<
      KernelName,
      sycl::ext::oneapi::experimental::info::kernel::max_num_work_groups_sync>(
      q, WorkGroupSize, props, bytes);
  size_t maxWGs1 = syclex::get_kernel_info<
      KernelName,
      sycl::ext::oneapi::experimental::info::kernel::max_num_work_groups_sync>(
      ctx, dev, WorkGroupSize, props, bytes);
  size_t maxWGs2 = syclex::get_kernel_info<
      KernelName,
      sycl::ext::oneapi::experimental::info::kernel::max_num_work_groups_sync>(
      ctx, dev, WorkGroupSize, props, bytes);
  assert(maxWGs == maxWGs1);
  assert(maxWGs == maxWGs2);

  // Construct an nd-range which launches the maximum number of work-groups.
  sycl::range<Dimensions> GlobalRange = WorkGroupSize;
  GlobalRange[0] *= maxWGs;
  int numItems = GlobalRange[0];
  if constexpr (Dimensions == 2)
    numItems *= GlobalRange[1];
  else if constexpr (Dimensions == 3)
    numItems *= GlobalRange[1] * GlobalRange[2];
  sycl::nd_range<Dimensions> ndr{GlobalRange, WorkGroupSize};
  int *data = sycl::malloc_device<int>(numItems, q);

  syclex::launch_config cfg{ndr, props};
  try {
    syclex::nd_launch<KernelName>(q, cfg, [data](sycl::nd_item<Dimensions> it) {
      auto root = it.ext_oneapi_get_root_group();
      data[root.get_local_linear_id()] = root.get_local_linear_id();
      sycl::group_barrier(root);
    });
  } catch (sycl::exception &e) {
    assert(maxWGs == 0);
  }
  q.wait();
}

int main() {
  sycl::queue q;
  sycl::context ctx = q.get_context();
  sycl::device dev = q.get_device();

  // When a kernel uses root-group synchronization, the total number of
  // work-groups is limited.  This limit is specific to the kernel, the
  // device, the work-group size (which is 32 in this example), the launch
  // parameters, and the amount of dynamically allocated work-group local
  // memory.
  auto props1 = syclex::properties{syclex::use_root_sync};
  auto props2 = syclex::properties{
      syclex::use_root_sync, syclexintel::cache_config{syclexintel::large_slm}};
  auto props3 =
      syclex::properties{syclex::use_root_sync,
                         syclexintel::cache_config{syclexintel::large_data}};
  auto props4 = syclex::properties{
      syclex::use_root_sync,
      syclex::work_group_scratch_size{DynamicLocalMem * sizeof(int)}};

  testFreeFunctionKernel<FreeFuncKernel1>(q, ctx, dev, sycl::range{WGDimSize},
                                          props1, 0);
  testFreeFunctionKernel<FreeFuncKernel2>(
      q, ctx, dev, sycl::range{WGDimSize, WGDimSize}, props2, 0);
  testFreeFunctionKernel<FreeFuncKernel2>(
      q, ctx, dev, sycl::range{WGDimSize, WGDimSize}, props3, 0);
  testFreeFunctionKernel<FreeFuncKernel3>(
      q, ctx, dev, sycl::range{WGDimSize, WGDimSize, WGDimSize}, props4, 0);
  testLambdaKernel<LambdaKernel1>(q, ctx, dev, sycl::range{WGDimSize}, props1,
                                  0);
  testLambdaKernel<LambdaKernel2>(q, ctx, dev,
                                  sycl::range{WGDimSize, WGDimSize}, props2, 0);
  testLambdaKernel<LambdaKernel3>(
      q, ctx, dev, sycl::range{WGDimSize, WGDimSize, WGDimSize}, props3, 0);
  return 0;
}
