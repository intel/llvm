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

  // get value to compare with
  auto *MaxLocalRange = sycl::malloc_shared<size_t>(1, q);
  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) {
       const auto sg = item.get_sub_group();
       *MaxLocalRange = sg.get_max_local_range()[0];
     });
   }).wait();

  {
    const auto MaxSubSGSize3D = kernel.template ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_sub_group_size>(
        q, sycl::range<3>{1, 1, 1});

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(MaxSubSGSize3D)>, uint32_t>,
        "max_sub_group_size query must return uint32_t");
    assert(MaxSubSGSize3D == *MaxLocalRange);
  }
  {
    const auto MaxSubSGSize2D = kernel.template ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_sub_group_size>(
        q, sycl::range<2>{1, 1});

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(MaxSubSGSize2D)>, uint32_t>,
        "max_sub_group_size query must return uint32_t");
    assert(MaxSubSGSize2D == *MaxLocalRange);
  }
  {
    const auto MaxSubSGSize1D = kernel.template ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_sub_group_size>(
        q, sycl::range<1>{1});

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(MaxSubSGSize1D)>, uint32_t>,
        "max_sub_group_size query must return uint32_t");
    assert(MaxSubSGSize1D == *MaxLocalRange);
  }
  sycl::free(MaxLocalRange, q);
}