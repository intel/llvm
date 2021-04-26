// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -internal-isystem %S/Inputs -fcxx-exceptions -sycl-std=2020 -fsyntax-only -fsycl-int-footer=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
//
// This test checks if compiler reports compilation error on an attempt to pass
// an array of non-trivially copyable structs as SYCL kernel parameter or
// a non-constant size array.

#include "sycl.hpp"

sycl::queue q;

struct NonTrivialCopyStruct {
  int i;
  NonTrivialCopyStruct(int _i) : i(_i) {}
  NonTrivialCopyStruct(const NonTrivialCopyStruct &x) : i(x.i) {}
};

struct NonTrivialDestructorStruct {
  int i;
  ~NonTrivialDestructorStruct();
};

void test() {
  NonTrivialCopyStruct NTCSObject[4] = {1, 2, 3, 4};
  NonTrivialDestructorStruct NTDSObject[5];

  q.submit([&](sycl::handler &h) {
    h.single_task<class kernel_capture_refs>([=] {
      int b = NTCSObject[2].i;
      int d = NTDSObject[4].i;
    });
  });
}

// CHECK-LABEL: #include <CL/sycl/detail/sycl_fe_intrins.hpp>
// CHECK: static_assert(::sycl::is_device_copyable<NonTrivialCopyStruct>_v, "error: kernel parameter type ('NonTrivialCopyStruct') is not device copyable");
// CHECK-NEXT: static_assert(::sycl::is_device_copyable<NonTrivialDestructorStruct>_v, "error: kernel parameter type ('NonTrivialDestructorStruct') is not device copyable");
