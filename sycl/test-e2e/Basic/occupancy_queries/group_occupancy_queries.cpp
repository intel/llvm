// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "sycl/backend_types.hpp"
#include "sycl/info/info_desc.hpp"
#include "sycl/range.hpp"
#include <cassert>

#include <sycl/detail/core.hpp>

#if !defined(SYCL_EXT_ONEAPI_GROUP_OCCUPANCY_QUERIES)
#error SYCL_EXT_ONEAPI_GROUP_OCCUPANCY_QUERIES is not defined!
#endif

#ifndef NDEBUG
#define NDEBUG_CHECK(cond)                                                     \
  if (!(cond))                                                                 \
    return 1;
#else
#define NDEBUG_CHECK // nop
#endif

namespace syclex = sycl::ext::oneapi::experimental;
using namespace sycl::info::device;
using namespace sycl::info::kernel_device_specific;

class QueryKernel;

int main() {
  sycl::queue q{};
  auto dev = q.get_device();
  auto ctx = q.get_context();

  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  auto kernel = bundle.get_kernel<QueryKernel>();

  const size_t MaxWorkGroupSize = dev.get_info<max_work_group_size>();
  const size_t MaxLocalMemorySizeInBytes = dev.get_info<local_mem_size>();

  size_t workGroupSize =
      std::min(MaxWorkGroupSize / 2, kernel.get_info<work_group_size>(dev));
  size_t localMemorySizeInBytes = (workGroupSize / 2) * sizeof(float);

  sycl::range<3> workGroupRange{workGroupSize, 1, 1};
  auto maxWGsPerCU = kernel.ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::recommended_num_work_groups>(
      q, workGroupRange, localMemorySizeInBytes);

  q.single_task<QueryKernel>([]() {}).wait();

  // Test the return type is as specified in the extension document.
  static_assert(std::is_same_v<std::remove_cv_t<decltype(maxWGsPerCU)>, size_t>,
                "recommended_num_work_groups query must return size_t");

  // We must have at least one active group if we are below resource limits.
  if (workGroupSize < MaxWorkGroupSize &&
      localMemorySizeInBytes < MaxLocalMemorySizeInBytes) {
    assert(maxWGsPerCU > 0 && "recommended_num_work_groups query failed");
    NDEBUG_CHECK(maxWGsPerCU > 0)
  }

  // In Cuda there cannot be any active groups for this kernel launch when all
  // the device resources maxed out, so ensure it, at least for Cuda.
  if (dev.get_backend() == sycl::backend::ext_oneapi_cuda) {
    workGroupSize = MaxWorkGroupSize;
    localMemorySizeInBytes = MaxLocalMemorySizeInBytes;
    workGroupRange[0] = workGroupSize;
    maxWGsPerCU = kernel.ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::recommended_num_work_groups>(
        q, workGroupRange, localMemorySizeInBytes);

    if (workGroupSize >= MaxWorkGroupSize &&
        localMemorySizeInBytes >= MaxLocalMemorySizeInBytes) {
      assert(maxWGsPerCU == 0 && "recommended_num_work_groups query failed");
      NDEBUG_CHECK(maxWGsPerCU == 0)
    }
  }

  return 0;
}
