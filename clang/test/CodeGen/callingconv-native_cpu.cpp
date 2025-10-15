// RUN: %clang_cc1 -triple native_cpu -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -emit-llvm -o - %s | FileCheck %s

struct S {
  char c;
  short s;
};
using short16 = short __attribute__((ext_vector_type(16)));

__attribute__((sycl_device))
S foo(short16, S);

__attribute__((sycl_device, libclc_call))
S bar(short16, S);

// CHECK: define noundef <16 x i16> @_Z3bazRDv16_sR1S(ptr noundef nonnull align 32 dereferenceable(32) %x, ptr noundef nonnull align 2 dereferenceable(4) %y)
__attribute__((sycl_device))
short16 baz(short16 &x, S &y) {
  // Host ABI:
  //   short16 argument is passed by reference.
  //   S is passed by value.
  //   S is returned by value.
  // CHECK: call i32 @_Z3fooDv16_s1S(ptr noundef byval(<16 x i16>) align 32 {{%.*}}, i32 {{%.*}})
  y = foo(x, y);
  // Libclc ABI:
  //   short16 is passed by value.
  //   S is passed by reference.
  //   S is returned by reference.
  // CHECK: call void @_Z3barDv16_s1S(ptr dead_on_unwind writable sret(%struct.S) align 2 {{%.*}}, <16 x i16> noundef {{%.*}}, ptr noundef byval(%struct.S) align 2 {{%.*}})
  y = bar(x, y);
  return x;
}
