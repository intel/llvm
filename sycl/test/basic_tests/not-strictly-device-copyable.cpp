// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm -I %sycl_include %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;

// Not device copyable because of the user defined Dtor.
struct NonDeviceCopiable {
  int a;
  std::string s; // not good for device, but never used in kernel.
};
template <> struct is_device_copyable<NonDeviceCopiable> : std::true_type {};

// Check clang is not emitting anything due to std::string
// CHECK-NOT: _ZSt19__throw_logic_errorPKc
// CHECK-NOT: _ZSt20__throw_length_errorPKc
// CHECK-NOT: _ZSt17__throw_bad_allocv
// CHECK-NOT: _Znwm
// CHECK-NOT: _ZdlPv

void test() {
  NonDeviceCopiable k{42};

  queue Q;
  Q.single_task<class TestA>([=] { (void)k.a; });
}
