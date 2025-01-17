// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/sycl.hpp>
#include <sycl/kernel.hpp>

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

template <int Dimensions>
void check_max_work_item_sizes(const sycl::queue& Q)
{
  const auto Dev = Q.get_device();
  const auto Ctx = Q.get_context();
  const auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx);
  const auto Kernel = Bundle.template get_kernel<kernels::TestKernel>();
  // get value to test
  const auto DevValues = Dev.get_info<sycl::info::device::max_work_item_sizes<Dimensions>>();
  const auto KernelValues = Kernel.ext_oneapi_get_info<syclex::info::kernel_queue_specific::max_work_item_sizes<Dimensions>>(Q);

  static_assert(std::is_same_v<std::remove_cv_t<decltype(KernelValues)>, sycl::id<Dimensions>>,
                "max_work_item_sizes query must return sycl::id<Dimensions>, Dimensions in range[1,3]");
  for(int i = 0; i < Dimensions; i++)
  {
    assert(KernelValues[i] == DevValues[i]);
  }
}

int main() {
  sycl::queue Q{};

  check_max_work_item_sizes<1>(Q);
  check_max_work_item_sizes<2>(Q);
  check_max_work_item_sizes<3>(Q);

  const auto Dev = Q.get_device();
  const auto Ctx = Q.get_context();
  const auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx);
  const auto Kernel = Bundle.template get_kernel<kernels::TestKernel>();
  const size_t MaxWorkGroupSizeActual =
    Kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(Dev);
  sycl::buffer<value_type, 1> Buf{sycl::range<1>{MaxWorkGroupSizeActual}};
  auto LaunchRange = sycl::nd_range<1>{sycl::range<1>{MaxWorkGroupSizeActual},
                                        sycl::range<1>{MaxWorkGroupSizeActual}};
  Q.submit([&](sycl::handler &cgh) {
       auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
       cgh.parallel_for(LaunchRange, kernels::TestKernel{Acc});
   }).wait();
}