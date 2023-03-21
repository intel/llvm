// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

// We should be generating the following error, but not the `sycl.id.get`
// operation.

// CHECK-LITERAL:  error: 'sycl.id.get' op result #0 must be 64-bit signless integer or memref of 64-bit signless integer values, but got 'f32'
// CHECK-NOT:      sycl.id.get

namespace sycl {
class tricky_class {
 public:
  static float get(const id<1> &, int) {
    return 7;
  }
};
}  // namespace sycl

SYCL_EXTERNAL float foo(const sycl::id<1> &i, int dimension) {
  return sycl::tricky_class::get(i, dimension);
}
