// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <cassert>
#include <cstdint>

namespace syclex = sycl::ext::oneapi::experimental;
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
}

int main() {
  sycl::queue q{};
  const auto ctx = q.get_context();
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  auto kernel = bundle.template get_kernel<kernels::TestKernel>();

  const size_t maxWorkGroupSizeActual =
    kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
  const auto maxWorkGroupSize = kernel.template ext_oneapi_get_info<
      syclex::info::kernel_queue_specific::max_work_group_size>(q);
  sycl::buffer<value_type, 1> buf{sycl::range<1>{maxWorkGroupSizeActual}};
  static_assert(std::is_same_v<std::remove_cv_t<decltype(maxWorkGroupSize)>, size_t>,
                "max_work_group_size query must return size_t");
  assert(maxWorkGroupSizeActual == maxWorkGroupSize);
   // Run the kernel
  auto launch_range = sycl::nd_range<1>{sycl::range<1>{maxWorkGroupSizeActual},
                                        sycl::range<1>{maxWorkGroupSize}};
  q.submit([&](sycl::handler &cgh) {
     auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
       cgh.parallel_for(launch_range, kernels::TestKernel{acc});
   }).wait();
}