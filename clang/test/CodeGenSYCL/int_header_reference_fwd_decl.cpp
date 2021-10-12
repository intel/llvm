// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// This test validates that we handle reference types properly in forward
// declarations, the bug was that the PassedAsRefType didn't get forward
// declared, so it resulted in a compile error during host compile. SO, we make
// sure that we properly forward declare the type (and any template children!).

#include <sycl.hpp>

// CHECK: // Forward declarations of templated kernel function types:
// CHECK-NEXT: namespace WrapsType {
// CHECK-NEXT: struct InsidePassedAsRef;
// CHECK-NEXT: }
// CHECK: namespace WrapsType {
// CHECK-NEXT: template <typename T> struct PassedAsRef;
// CHECK-NEXT: }
// CHECK: template <typename T> class Wrapper;

namespace WrapsType {
  struct InsidePassedAsRef{};
  template<typename T>
  struct PassedAsRef{};
}

template<typename T>
class Wrapper{};

void foo() {
  using namespace WrapsType;
  using KernelName = Wrapper<PassedAsRef<InsidePassedAsRef>&>;
  sycl::kernel_single_task<KernelName>([]{});
}
