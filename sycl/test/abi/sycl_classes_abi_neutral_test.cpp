// RUN: %clangxx -std=c++17 -I %sycl_include -I %sycl_include/sycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts-complete %s -o %t.out | grep -Pzo "0 \| (class|struct) sycl::.*\n([^\n].*\n)*" | sort -z | FileCheck --implicit-check-not "{{std::basic_string|std::list}}" %s
// RUN: %clangxx -std=c++17 -I %sycl_include -I %sycl_include/sycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts-complete %s -o %t.out | grep -Pzo "0 \| (class|struct) sycl::.*\n([^\n].*\n)*" | sort -z | FileCheck --implicit-check-not "{{std::basic_string|std::list}}" %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// The purpose of this test is to check that classes in sycl namespace that are
// defined in SYCL headers don't have std::string and std::list data members to
// avoid having the dual ABI issue (see
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html). I.e. if
// application is built with the old ABI and such data member is crossing ABI
// boundary then it will result in issues as SYCL RT is using new ABI by
// default. All such data members can potentially cross ABI boundaries and
// that's why we need to be sure that we use only ABI-neutral data members.

// New exclusions are NOT ALLOWED to this file unless it is guaranteed that data
// member is not crossing ABI boundary. All current exclusions are listed below.

// CHECK: 0 | struct sycl::ext::oneapi::experimental::build_options
// CHECK-NEXT:         0 |   struct sycl::ext::oneapi::experimental::detail::run_time_property_key<sycl::ext::oneapi::experimental::detail::BuildOptions> (base) (empty)
// CHECK-NEXT:         0 |     struct sycl::ext::oneapi::experimental::detail::property_key_base_tag (base) (empty)
// CHECK-NEXT:         0 |   class std::vector<class std::basic_string<char> > opts
// CHECK-NEXT:         0 |     struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > > (base)
// CHECK-NEXT:         0 |       struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >::_Vector_impl _M_impl
// CHECK-NEXT:         0 |         class std::allocator<class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |           class {{(std::__new_allocator|__gnu_cxx::new_allocator)}}<class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |         struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >::_Vector_impl_data (base)


// CHECK: 0 | struct sycl::ext::oneapi::experimental::include_files
// CHECK-NEXT:         0 |   struct sycl::ext::oneapi::experimental::detail::run_time_property_key<sycl::ext::oneapi::experimental::detail::IncludeFiles> (base) (empty)
// CHECK-NEXT:         0 |     struct sycl::ext::oneapi::experimental::detail::property_key_base_tag (base) (empty)
// CHECK-NEXT:         0 |   class std::vector<struct std::pair<class std::basic_string<char>, class std::basic_string<char> > > record
// CHECK-NEXT:         0 |     struct std::_Vector_base<struct std::pair<class std::basic_string<char>, class std::basic_string<char> >, class std::allocator<struct std::pair<class std::basic_string<char>, class std::basic_string<char> > > > (base)
// CHECK-NEXT:         0 |       struct std::_Vector_base<struct std::pair<class std::basic_string<char>, class std::basic_string<char> >, class std::allocator<struct std::pair<class std::basic_string<char>, class std::basic_string<char> > > >::_Vector_impl _M_impl
// CHECK-NEXT:         0 |         class std::allocator<struct std::pair<class std::basic_string<char>, class std::basic_string<char> > > (base) (empty)
// CHECK-NEXT:         0 |           class {{(std::__new_allocator|__gnu_cxx::new_allocator)}}<struct std::pair<class std::basic_string<char>, class std::basic_string<char> > > (base) (empty)
// CHECK-NEXT:         0 |         struct std::_Vector_base<struct std::pair<class std::basic_string<char>, class std::basic_string<char> >, class std::allocator<struct std::pair<class std::basic_string<char>, class std::basic_string<char> > > >::_Vector_impl_data (base)


// CHECK: 0 | struct sycl::ext::oneapi::experimental::registered_kernel_names
// CHECK-NEXT:         0 |   struct sycl::ext::oneapi::experimental::detail::run_time_property_key<sycl::ext::oneapi::experimental::detail::RegisteredKernelNames> (base) (empty)
// CHECK-NEXT:         0 |     struct sycl::ext::oneapi::experimental::detail::property_key_base_tag (base) (empty)
// CHECK-NEXT:         0 |   class std::vector<class std::basic_string<char> > kernel_names
// CHECK-NEXT:         0 |     struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > > (base)
// CHECK-NEXT:         0 |       struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >::_Vector_impl _M_impl
// CHECK-NEXT:         0 |         class std::allocator<class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |           class {{(std::__new_allocator|__gnu_cxx::new_allocator)}}<class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |         struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >::_Vector_impl_data (base)

#include <sycl/sycl.hpp>
