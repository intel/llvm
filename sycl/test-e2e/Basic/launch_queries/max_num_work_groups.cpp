// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <cassert>
#include <cstdint>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace sycl::info::device;
using namespace sycl::info::kernel_device_specific;

using value_type = int64_t;

namespace kernels {

template <class T, size_t Dim>
using sycl_global_accessor =
    sycl::accessor<T, Dim, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>;

class TestKernel {
public:
  static constexpr bool HasLocalMemory{false};

  TestKernel(sycl_global_accessor<value_type, 1> acc) : acc_{acc} {}

  void operator()(sycl::nd_item<1> item) const {
    const auto gtid = item.get_global_linear_id();
    acc_[gtid] = gtid + 42;
  }

private:
  sycl_global_accessor<value_type, 1> acc_;
};

class TestLocalMemoryKernel {
public:
  static constexpr bool HasLocalMemory{true};

  TestLocalMemoryKernel(sycl_global_accessor<value_type, 1> acc,
                        sycl::local_accessor<value_type, 1> loc_acc)
      : acc_{acc}, loc_acc_{loc_acc} {}

  void operator()(sycl::nd_item<1> item) const {
    const auto ltid = item.get_local_linear_id();
    const auto gtid = item.get_global_linear_id();
    if (ltid < loc_acc_.size()) {
      loc_acc_[ltid] = ltid + 42;
      item.barrier(sycl::access::fence_space::local_space);
      acc_[gtid] = loc_acc_[ltid];
    } else {
      acc_[gtid] = 0;
    }
  }

private:
  sycl_global_accessor<value_type, 1> acc_;
  sycl::local_accessor<value_type, 1> loc_acc_;
};

} // namespace kernels

namespace {

template <class KernelName>
int test_max_num_work_groups(sycl::queue &q, const sycl::device &dev) {
  const auto ctx = q.get_context();
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  auto kernel = bundle.template get_kernel<KernelName>();

  const size_t maxWorkGroupSize =
      kernel.template get_info<work_group_size>(dev);
  // Will try to launch 2 work groups.
  const size_t NumWorkItems = maxWorkGroupSize * 2;

  size_t workGroupSize = 32;
  size_t localMemorySizeInBytes{0};
  if constexpr (KernelName::HasLocalMemory) {
    localMemorySizeInBytes = workGroupSize * sizeof(value_type);
  }

  sycl::buffer<value_type, 1> buf{sycl::range<1>{NumWorkItems}};

  // Tests

  // ==================== //
  // Test 1 - return type //
  // ==================== //
  sycl::range<3> workGroupRange{workGroupSize, 1, 1};
  auto maxWGs = kernel.template ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::max_num_work_groups>(
      q, workGroupRange, localMemorySizeInBytes);

  // Test the return type is as specified in the extension document.
  static_assert(std::is_same_v<std::remove_cv_t<decltype(maxWGs)>, size_t>,
                "max_num_work_groups query must return size_t");

  // ===================== //
  // Test 2 - return value //
  // ===================== //
  // We must have at least one active group if we are below resource limits.
  assert(maxWGs > 0 && "max_num_work_groups query failed");

  // Run the kernel
  auto launch_range = sycl::nd_range<1>{sycl::range<1>{NumWorkItems},
                                        sycl::range<1>{workGroupSize}};
  q.submit([&](sycl::handler &cgh) {
     auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
     if constexpr (KernelName::HasLocalMemory) {
       sycl::local_accessor<value_type, 1> loc_acc{
           sycl::range<1>{workGroupSize}, cgh};
       cgh.parallel_for(launch_range, KernelName{acc, loc_acc});
     } else {
       cgh.parallel_for(launch_range, KernelName{acc});
     }
   }).wait();
  assert(sycl::host_accessor{buf}[0] == 42);

  // ========================== //
  // Test 3 - use max resources //
  // ========================== //
  workGroupSize = maxWorkGroupSize;
  workGroupRange[0] = workGroupSize;
  size_t localSize =
      (dev.get_info<sycl::info::device::local_mem_size>() / sizeof(value_type));
  if constexpr (KernelName::HasLocalMemory) {
    localMemorySizeInBytes = localSize * sizeof(value_type);
  }
  maxWGs = kernel.template ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::max_num_work_groups>(
      q, workGroupRange, localMemorySizeInBytes);

  // We must have at least one active group if we are at resource limits.
  assert(maxWGs > 0 && "max_num_work_groups query failed");

  launch_range = sycl::nd_range<1>{sycl::range<1>{NumWorkItems},
                                   sycl::range<1>{workGroupSize}};

  q.submit([&](sycl::handler &cgh) {
     auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
     if constexpr (KernelName::HasLocalMemory) {
       sycl::local_accessor<value_type, 1> loc_acc{sycl::range<1>{localSize},
                                                   cgh};
       cgh.parallel_for(launch_range, KernelName{acc, loc_acc});
     } else {
       cgh.parallel_for(launch_range, KernelName{acc});
     }
   }).wait();
  assert(sycl::host_accessor{buf}[0] == 42);

  // =============================== //
  // Test 4 - exceed resource limits //
  // =============================== //
  // A little over the maximum work-group size for the purpose of exceeding.
  workGroupSize = maxWorkGroupSize + 32;
  workGroupRange[0] = workGroupSize;
  maxWGs = kernel.template ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::max_num_work_groups>(
      q, workGroupRange, localMemorySizeInBytes);
  // It cannot be possible to launch a kernel successfully with a configuration
  // that exceeds the available resources as in the above defined workGroupSize.
  // workGroupSize is larger than maxWorkGroupSize, hence maxWGs must equal 0.
  assert(maxWGs == 0 &&
         "max_num_work_groups query failed.\n"
         "It should return 0 possible groups when the requested resources "
         "by the lanuch config exceed those available in the hardware.");

  // As we ensured that the 'max_num_work_groups' query correctly
  // returns 0 possible work-groups, test that the kernel launch will fail.
  // A configuration that defines a work-group size larger than the maximum
  // possible should result in failure.
  try {
    launch_range = sycl::nd_range<1>{sycl::range<1>{NumWorkItems},
                                     sycl::range<1>{workGroupSize}};

    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
       if constexpr (KernelName::HasLocalMemory) {
         sycl::local_accessor<value_type, 1> loc_acc{sycl::range<1>{localSize},
                                                     cgh};
         cgh.parallel_for(launch_range, KernelName{acc, loc_acc});
       } else {
         cgh.parallel_for(launch_range, KernelName{acc});
       }
     }).wait();
  } catch (const sycl::exception &e) {
    // 'nd_range' error is the expected outcome from the above launch config.
    if (e.code() == sycl::make_error_code(sycl::errc::nd_range)) {
      return 0;
    }
    std::cerr << e.code() << ":\t";
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}

} // namespace

int main() {
  sycl::queue q{};
  sycl::device dev = q.get_device();

  using namespace kernels;

  int ret{0};
  ret &= test_max_num_work_groups<TestKernel>(q, dev);
  ret &= test_max_num_work_groups<TestLocalMemoryKernel>(q, dev);
  return ret;
}
