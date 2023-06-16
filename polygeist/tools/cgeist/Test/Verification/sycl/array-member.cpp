// RUN: clang++ -O0 -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-device-only -w -emit-mlir %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;

class Kernel;

int main(){
  queue q;
  range range{8};

  buffer<float, 1> buf{nullptr, range};

  q.submit([&](handler& cgh){
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
    const float array[] = {1, 2, 3, 4, 5, 6, 7, 8};

// COM: Test the array member of the structure is built in place
// CHECK-LABEL:     gpu.func @_ZTS6Kernel(
// CHECK-SAME:                            %[[VAL_219:.*]]: !llvm.ptr
// CHECK:             %[[VAL_220:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_227:.*]] = llvm.alloca %[[VAL_220]] x !llvm.struct<(array<8 x f32>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:        %[[VAL_228:.*]] = llvm.getelementptr inbounds %[[VAL_219]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<8 x f32>)>
// CHECK-NEXT:        %[[VAL_229:.*]] = llvm.getelementptr inbounds %[[VAL_228]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x f32>
// CHECK-NEXT:        affine.for %[[VAL_230:.*]] = 0 to 8 {
// CHECK-NEXT:          %[[VAL_231:.*]] = arith.index_cast %[[VAL_230]] : index to i64
// CHECK-NEXT:          %[[VAL_232:.*]] = llvm.getelementptr %[[VAL_229]]{{\[}}%[[VAL_231]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:          %[[VAL_233:.*]] = llvm.load %[[VAL_232]] : !llvm.ptr -> f32
// CHECK-NEXT:          %[[VAL_234:.*]] = arith.index_cast %[[VAL_230]] : index to i32
// CHECK-NEXT:          %[[VAL_235:.*]] = llvm.getelementptr %[[VAL_227]]{{\[}}%[[VAL_234]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK-NEXT:          llvm.store %[[VAL_233]], %[[VAL_235]] : f32, !llvm.ptr
// CHECK-NEXT:        }

// COM: (void)array to ensure the array is captured in the lambda.
    cgh.parallel_for<Kernel>(range, [=](id<1>){ (void)array; });
  });

  return 0;
}
