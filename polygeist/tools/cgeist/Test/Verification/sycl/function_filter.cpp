// RUN: clang++ -fsycl -fsycl-device-only -emit-mlir -Xcgeist -function=match %s -o - | FileCheck %s
// RUN: clang++ -fsycl -fsycl-device-only -emit-mlir -Xcgeist -function=match$ %s -o - | FileCheck %s --check-prefix=CHECK-END

// COM: No GPU functions.
// CHECK-NOT: gpu.func

// COM: No definitions with name contain func.
// CHECK-NOT: func.func {{.*}}func{{.*}} {

// COM: func1 is a declaration.
// CHECK: func.func private @func1()

// COM: match and match1 are kept as definitions.
// CHECK: func.func @match() {{.*}} {
// CHECK: func.func @match1() {{.*}} {

// COM: Only match is kept as definitions.
// CHECK-END: func.func @match() {{.*}} {
// CHECK-END-NOT: func.func @match1() {{.*}} {

#include <sycl/sycl.hpp>
using namespace sycl;

extern "C" SYCL_EXTERNAL void func1() {}
extern "C" SYCL_EXTERNAL void match() {
  func1();
}
extern "C" SYCL_EXTERNAL void match1() {}
extern "C" SYCL_EXTERNAL void func2() {}

void single_task(std::array<int, 1> &A) {
  auto q = queue{};
  {
    auto buf = buffer<int, 1>{A.data(), 1};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class kernel_single_task>([=]() {
        A[0] = 1;
      });
    });
  }
}
