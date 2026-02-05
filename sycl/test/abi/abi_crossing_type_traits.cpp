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
template <typename T>
void check_type_traits() {
  std::cout << "Type: " << typeid(T).name() << "\n";
  std::cout << "Trivially copyable: "
            << std::is_trivially_copyable_v<T> << "\n";
  std::cout << "Trivially default constructible: "
            << std::is_trivially_default_constructible_v<T> << "\n";
  std::cout << "Standard layout: " << std::is_standard_layout_v<T> << "\n";
}

int main() {

  // CHECK: Type: N4sycl3_V16deviceE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<device>();

  // CHECK-NEXT: Type: N4sycl3_V16bufferIiLi1ENS0_6detail17aligned_allocatorIiEEvEE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<buffer<int, 1>>();

  // CHECK-NEXT: Type: N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<
  accessor<int, 1, access::mode::read_write, access::target::device>>();

  // CHECK-NEXT: Type: N4sycl3_V114local_accessorIiLi1EEE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<local_accessor<int, 1>>();

  // CHECK-NEXT: Type: N4sycl3_V12idILi3EEE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<id<3>>();

  // CHECK-NEXT: Type: N4sycl3_V15rangeILi3EEE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<range<3>>();

  // CHECK-NEXT: Type: N4sycl3_V14itemILi3ELb1EEE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<item<3>>();

  // CHECK-NEXT: Type: N4sycl3_V17nd_itemILi3EEE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<nd_item<3>>();

  // CHECK-NEXT: Type: N4sycl3_V18nd_rangeILi3EEE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<nd_range<3>>();

  // CHECK-NEXT: Type: N4sycl3_V16detail27compile_time_kernel_info_v123CompileTimeKernelInfoTyE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<
  detail::compile_time_kernel_info_v1::CompileTimeKernelInfoTy>();

  // CHECK-NEXT: Type: N4sycl3_V19exceptionE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<exception>();

  // CHECK-NEXT: Type: N4sycl3_V17handlerE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<handler>();

  // CHECK-NEXT: Type: N4sycl3_V16detail17HostKernelRefBaseE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::HostKernelRefBase>();

  // CHECK-NEXT: Type: N4sycl3_V15imageILi1ENS0_6detail17aligned_allocatorIhEEEE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<image<1>>();

  // CHECK-NEXT: Type: N4sycl3_V16detail16nd_range_view_v113nd_range_viewE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<detail::nd_range_view_v1::nd_range_view>();

  // CHECK-NEXT: Type: N4sycl3_V16detail27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENS4_5intel12experimental12cache_configENS6_17use_root_sync_keyENS6_23work_group_progress_keyENS6_22sub_group_progress_keyENS6_22work_item_progress_keyENS6_4cuda12cluster_sizeILi1EEENSG_ILi2EEENSG_ILi3EEEEEE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::KernelPropertyHolderStructTy>();

  // CHECK-NEXT: Type: N4sycl3_V14spanIiLm4EEE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<span<int, 4>>();

  // CHECK-NEXT: Type: N4sycl3_V16detail14tls_code_loc_tE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::tls_code_loc_t>();

  // CHECK-NEXT: Type: N4sycl3_V13vecIiLi4EEE
  // CHECK-NEXT: Trivially copyable: 1
  // CHECK-NEXT: Trivially default constructible: 1
  // CHECK-NEXT: Standard layout: 1
  check_type_traits<vec<int, 4>>();

  // CHECK-NEXT: Type: N4sycl3_V16detail20PropertyWithDataBaseE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::PropertyWithDataBase>();

  // CHECK-NEXT: Type: N4sycl3_V16detail19SYCLMemObjAllocatorE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<detail::SYCLMemObjAllocator>();

  // CHECK-NEXT: Type: N4sycl3_V115device_selectorE
  // CHECK-NEXT: Trivially copyable: 0
  // CHECK-NEXT: Trivially default constructible: 0
  // CHECK-NEXT: Standard layout: 0
  check_type_traits<device_selector>();

  return 0;
}