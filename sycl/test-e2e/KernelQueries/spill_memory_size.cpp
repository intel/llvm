//==--- spill_memory_size.cpp --- kernel_queries extension tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/kernel.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/nd_range.hpp>

#include <cassert>
#include <cstddef>
#include <type_traits>

using value_type = uint64_t;

namespace kernels {
template <class T, std::size_t Dim>
using sycl_global_accessor =
    sycl::accessor<T, Dim, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>;
class TestKernel {
public:
  TestKernel(sycl_global_accessor<value_type, 1> acc) : acc_(acc) {}

  void operator()(sycl::nd_item<1> item) const {
    const auto gtid = item.get_global_linear_id();
    acc_[gtid] = 42;
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

  assert(!bundle.empty());

  if (dev.has(sycl::aspect::ext_intel_spill_memory_size)) {
    const auto spillMemSz = kernel.template get_info<
        sycl::ext::intel::info::kernel_device_specific::spill_memory_size>(dev);

    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(spillMemSz)>, std::size_t>,
        "spill_memory_size query must return size_t");
  } else {
    try {
      const auto spillMemSz = kernel.template get_info<
          sycl::ext::intel::info::kernel_device_specific::spill_memory_size>(
          dev);
    } catch (const sycl::exception &e) {
      // 'feature_not_supported' is the expected outcome from the query above.
      if (e.code() ==
          sycl::make_error_code(sycl::errc::feature_not_supported)) {
        return 0;
      }
      std::cerr << e.code() << ":\t";
      std::cerr << e.what() << std::endl;
      return 1;
    }
  }
}