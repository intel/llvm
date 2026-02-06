// Test to check type traits of SYCL types when crossing ABI boundaries.
// Type traits, like POD, trivially copyable, standard layout, etc if changed
// outside the ABI breaking window can break the ABI and cause UB.

// RUN: %clangxx -fsycl -o %t.out %s
// RUN: %t.out 2>&1 | FileCheck %s

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
template <typename T> void check_type_traits() {
  std::cout << "Trivially copyable: "
            << std::is_trivially_copyable_v<T> << "\n";
  std::cout << "Trivially default constructible: "
            << std::is_trivially_default_constructible_v<T> << "\n";
  std::cout << "Standard layout: " << std::is_standard_layout_v<T> << "\n";
}

int main() {

  // CHECK: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<device>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<buffer<int, 1>>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<
      accessor<int, 1, access::mode::read_write, access::target::device>>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<local_accessor<int, 1>>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<id<3>>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<range<3>>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<item<3>>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<nd_item<3>>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<nd_range<3>>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<
      detail::compile_time_kernel_info_v1::CompileTimeKernelInfoTy>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<exception>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<handler>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::HostKernelRefBase>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<image<1>>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<detail::nd_range_view_v1::nd_range_view>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::KernelPropertyHolderStructTy>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<span<int, 4>>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::tls_code_loc_t>();

  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 1
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<vec<int, 4>>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::PropertyWithDataBase>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::SYCLMemObjAllocator>();

  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<device_selector>();

  return 0;
}
