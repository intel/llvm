// RUN: %clang_cc1 -fsycl-is-device -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// This test checks if the SYCL device compiler is able to generate correct
// integration header when the kernel name class is wrapped in an inline
// namespace.

#include "Inputs/sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
  kernelFunc();
}

// Top-level inline namespace
// CHECK: inline namespace ns1 {
// CHECK-NEXT: class Foo11;
// CHECK-NEXT: }
// CHECK-NEXT: inline namespace ns1 {
// CHECK-NEXT: class Foo12;
// CHECK-NEXT: }
// CHECK-NEXT: inline namespace ns1 {
// CHECK-NEXT: class Foo13;
// CHECK-NEXT: }
inline namespace ns1 {
class Foo11 {};
class Foo12;
class Foo13 {};
} // namespace ns1

// Nested inline namespace
// CHECK-NEXT: namespace ns2 { inline namespace ns3 {
// CHECK-NEXT: class Foo21;
// CHECK-NEXT: }}
// CHECK-NEXT: namespace ns2 { inline namespace ns3 {
// CHECK-NEXT: class Foo22;
// CHECK-NEXT: }}
namespace ns2 {
inline namespace ns3 {
class Foo21 {};
class Foo22;
} // namespace ns3
} // namespace ns2

// Namespace nested inside nested inline namespace
// CHECK-NEXT: namespace ns4 { inline namespace ns5 { namespace ns6 {
// CHECK-NEXT: class Foo31;
// CHECK-NEXT: }}}
// CHECK-NEXT: namespace ns4 { inline namespace ns5 { namespace ns6 {
// CHECK-NEXT: class Foo32;
// CHECK-NEXT: }}}
namespace ns4 {
inline namespace ns5 {
namespace ns6 {
class Foo31 {};
class Foo32;
} // namespace ns6
} // namespace ns5
} // namespace ns4

int main() {
  kernel_single_task<Foo11>([]() {});
  kernel_single_task<::Foo12>([]() {});
  kernel_single_task<ns1::Foo13>([]() {});

  kernel_single_task<ns2::Foo21>([]() {});
  kernel_single_task<ns2::ns3::Foo22>([]() {});

  kernel_single_task<ns4::ns6::Foo31>([]() {});
  kernel_single_task<ns4::ns5::ns6::Foo32>([]() {});

  return 0;
}
