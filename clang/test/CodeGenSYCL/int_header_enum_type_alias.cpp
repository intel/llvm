// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

#include "sycl.hpp"
// Test to ensure that the integration header doesn't get confused by type
// aliases. The sycl::access::target and sycl::access::mode scoped enums
// were brought into the 'sycl' namespace (as target and acccess_mode) via
// type aliases, which confused the int-header printer.  Make sure we correctly
// print the actual name, not the alias name.

// CHECK: Forward declarations of templated kernel function types:
// CHECK-NEXT: inline namespace cl { namespace sycl { namespace access {
// CHECK-NEXT: enum class mode : int;
// CHECK-NEXT: }}}
// CHECK-NEXT: inline namespace cl { namespace sycl { namespace access {
// CHECK-NEXT: enum class target : int;
// CHECK-NEXT: }}}
// This is the important line, we make sure we look through the type alias.
// CHECK-NEXT: template <sycl::access::mode mode, sycl::access::target target> struct get_kernel;

// The key here is that we are accessing these types via the type alias.
template<sycl::access_mode mode, sycl::target target>
struct get_kernel{};

void foo() {
  using kernel_name = get_kernel<sycl::access_mode::read,
                                 sycl::target::host_buffer>;
  sycl::kernel_single_task<kernel_name>([](){});
}
