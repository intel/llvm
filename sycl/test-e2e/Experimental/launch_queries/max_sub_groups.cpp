// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/kernel.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/sub_group.hpp>
#include <sycl/usm.hpp>

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
} // namespace kernels

int main() {
  sycl::queue q{};
  const auto ctx = q.get_context();
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  auto kernel = bundle.template get_kernel<kernels::TestKernel>();
  const auto dev = q.get_device();

  sycl::buffer<value_type, 1> buf{sycl::range<1>{1}};
  auto launchRange = sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}};
  q.submit([&](sycl::handler &cgh) {
     auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
     cgh.parallel_for<class kernels::TestKernel>(launchRange,
                                                 kernels::TestKernel{acc});
   }).wait();

  const size_t maxDeviceValue =
      dev.get_info<sycl::info::device::max_num_sub_groups>();
  {
    const sycl::range<3> r3D{1, 1, 1};
    const auto subSGSize = kernel.template ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::num_sub_groups>(q, r3D);

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(subSGSize)>, uint32_t>,
        "num_sub_groups query must return uint32_t");
    assert(subSGSize <= maxDeviceValue);
  }
  {
    const sycl::range<2> r2D{1, 1};
    const auto subSGSize = kernel.template ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::num_sub_groups>(q, r2D);

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(subSGSize)>, uint32_t>,
        "num_sub_groups query must return uint32_t");
    assert(subSGSize <= maxDeviceValue);
  }
  {
    const sycl::range<1> r1D{1};
    const auto subSGSize = kernel.template ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::num_sub_groups>(q, r1D);

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(subSGSize)>, uint32_t>,
        "num_sub_groups query must return uint32_t");
    assert(subSGSize <= maxDeviceValue);
  }
}
