// Test to check type traits of SYCL types when crossing ABI boundaries.
// Type traits, like POD, trivially copyable, standard layout, etc if changed
// outside the ABI breaking window can break the ABI and cause UB.

// RUN: %clangxx %fsycl-host-only %s

#include <iostream>
#include <sycl/detail/common.hpp>
#include <sycl/detail/kernel_launch_helper.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

using namespace sycl;

// For a type to be POD, it has to be trivial and standard layout.
// For a type to be trivial, it has to be trivially copyable and
// trivially default constructible.
// Don't use std::is_pod as it is deprecated in C++20 or std::is_trivial
// as it will be deprecated in C++26.
template <typename T, bool TriviallyCopyable,
          bool TriviallyDefaultConstructible, bool StandardLayout>
void check_type_traits() {
  static_assert(std::is_trivially_copyable_v<T> == TriviallyCopyable,
                "Trivially copyable mismatch");
  static_assert(std::is_trivially_default_constructible_v<T> ==
                    TriviallyDefaultConstructible,
                "Trivially default constructible mismatch");
  static_assert(std::is_standard_layout_v<T> == StandardLayout,
                "Standard layout mismatch");
}

int main() {

  check_type_traits<device, false, false, true>();
  check_type_traits<buffer<int, 1>, false, false, false>();
  check_type_traits<
      accessor<int, 1, access::mode::read_write, access::target::device>, false,
      false, false>();
  check_type_traits<local_accessor<int, 1>, false, false, false>();
  check_type_traits<id<3>, true, false, true>();
  check_type_traits<range<3>, true, false, true>();
  check_type_traits<item<3>, true, false, true>();
  check_type_traits<nd_item<3>, true, false, true>();
  check_type_traits<nd_range<3>, true, false, true>();
  check_type_traits<
      detail::compile_time_kernel_info_v1::CompileTimeKernelInfoTy, true, false,
      true>();
  check_type_traits<exception, false, false, false>();
  check_type_traits<handler, false, false, false>();
  check_type_traits<detail::HostKernelRefBase, false, false, false>();
  check_type_traits<image<1>, false, false, true>();
  check_type_traits<detail::nd_range_view_v1::nd_range_view, true, false,
                    true>();
  check_type_traits<detail::KernelPropertyHolderStructTy, true, false, false>();
  check_type_traits<span<int, 4>, true, false, true>();
  check_type_traits<detail::tls_code_loc_t, false, false, false>();
  check_type_traits<vec<int, 4>, true, true, true>();
  check_type_traits<detail::PropertyWithDataBase, false, false, false>();
  check_type_traits<detail::SYCLMemObjAllocator, false, false, false>();
  check_type_traits<device_selector, false, false, false>();

  return 0;
}
