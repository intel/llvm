// RUN: clang++  -O0 -S -fsycl -fsycl-device-only -w -emit-mlir %s -o - | FileCheck %s

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

// COM: Check the array member is constructed as expected:
// CHECK-LABEL:     gpu.func @_ZTS6Kernel(
// CHECK-SAME:                            %{{.*}}: memref<?xf32, 1>
// CHECK-SAME:                            %{{.*}}: memref<?x!sycl_range_1_> {{.*}}, %{{.*}}: memref<?x!sycl_range_1_>
// CHECK-SAME:                            %{{.*}}: memref<?x!sycl_id_1_>
// CHECK-SAME:                            %[[VAL_303:.*]]: !llvm.ptr
// CHECK:             %[[VAL_305:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_326:.*]] = memref.alloca() : memref<1x!llvm.struct<(!sycl_accessor_1_f32_rw_dev, array<8 x f32>)>>
// CHECK-NEXT:        %[[VAL_327:.*]] = memref.cast %[[VAL_326]] : memref<1x!llvm.struct<(!sycl_accessor_1_f32_rw_dev, array<8 x f32>)>> to memref<?x!llvm.struct<(!sycl_accessor_1_f32_rw_dev, array<8 x f32>)>>
// CHECK:             %[[VAL_331:.*]] = "polygeist.subindex"(%[[VAL_327]], %[[VAL_305]]) : (memref<?x!llvm.struct<(!sycl_accessor_1_f32_rw_dev, array<8 x f32>)>>, index) -> memref<?x!llvm.array<8 x f32>>
// CHECK-NEXT:        %[[VAL_332:.*]] = "polygeist.memref2pointer"(%[[VAL_331]]) : (memref<?x!llvm.array<8 x f32>>) -> !llvm.ptr
// CHECK-NEXT:        %[[VAL_333:.*]] = llvm.getelementptr inbounds %[[VAL_303]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<8 x f32>)>
// CHECK-NEXT:        %[[VAL_334:.*]] = llvm.getelementptr inbounds %[[VAL_333]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x f32>
// CHECK-NEXT:        affine.for %[[VAL_335:.*]] = 0 to 8 {
// CHECK-NEXT:          %[[VAL_336:.*]] = arith.index_cast %[[VAL_335]] : index to i64
// CHECK-NEXT:          %[[VAL_337:.*]] = llvm.getelementptr %[[VAL_334]]{{\[}}%[[VAL_336]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:          %[[VAL_338:.*]] = llvm.load %[[VAL_337]] : !llvm.ptr -> f32
// CHECK-NEXT:          %[[VAL_339:.*]] = arith.index_cast %[[VAL_335]] : index to i32
// CHECK-NEXT:          %[[VAL_340:.*]] = llvm.getelementptr %[[VAL_332]]{{\[}}%[[VAL_339]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK-NEXT:          llvm.store %[[VAL_338]], %[[VAL_340]] : f32, !llvm.ptr
// CHECK-NEXT:        }

// COM: Check the *= operation is performed:
// CHECK-LABEL:     func.func private @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_2idILi1EEEE_clES5_(
// CHECK-SAME:                                                                                               %[[VAL_358:.*]]: memref<?x!llvm.struct<(!sycl_accessor_1_f32_rw_dev, array<8 x f32>)>, 4>
// CHECK-SAME:                                                                                               %[[VAL_359:.*]]: memref<?x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_360:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_361:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_362:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_363:.*]] = memref.cast %[[VAL_362]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_364:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_365:.*]] = memref.cast %[[VAL_364]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_366:.*]] = "polygeist.subindex"(%[[VAL_358]], %[[VAL_360]]) : (memref<?x!llvm.struct<(!sycl_accessor_1_f32_rw_dev, array<8 x f32>)>, 4>, index) -> memref<?x!llvm.array<8 x f32>, 4>
// CHECK-NEXT:        %[[VAL_367:.*]] = "polygeist.memref2pointer"(%[[VAL_366]]) : (memref<?x!llvm.array<8 x f32>, 4>) -> !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_368:.*]] = sycl.id.get %[[VAL_359]][] : (memref<?x!sycl_id_1_>) -> i64
// CHECK-NEXT:        %[[VAL_369:.*]] = arith.trunci %[[VAL_368]] : i64 to i32
// CHECK-NEXT:        %[[VAL_370:.*]] = llvm.getelementptr %[[VAL_367]]{{\[}}%[[VAL_369]]] : (!llvm.ptr<4>, i32) -> !llvm.ptr<4>, f32
// CHECK-NEXT:        %[[VAL_371:.*]] = llvm.load %[[VAL_370]] : !llvm.ptr<4> -> f32
// CHECK-NEXT:        %[[VAL_372:.*]] = "polygeist.subindex"(%[[VAL_358]], %[[VAL_361]]) : (memref<?x!llvm.struct<(!sycl_accessor_1_f32_rw_dev, array<8 x f32>)>, 4>, index) -> memref<?x!sycl_accessor_1_f32_rw_dev, 4>
// CHECK-NEXT:        %[[VAL_373:.*]] = memref.memory_space_cast %[[VAL_365]] : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
// CHECK-NEXT:        %[[VAL_374:.*]] = memref.memory_space_cast %[[VAL_359]] : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
// CHECK-NEXT:        sycl.constructor @id(%[[VAL_373]], %[[VAL_374]]) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ERKS2_} : (memref<?x!sycl_id_1_, 4>, memref<?x!sycl_id_1_, 4>)
// CHECK-NEXT:        %[[VAL_375:.*]] = affine.load %[[VAL_364]][0] : memref<1x!sycl_id_1_>
// CHECK-NEXT:        affine.store %[[VAL_375]], %[[VAL_362]][0] : memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_376:.*]] = sycl.accessor.subscript %[[VAL_372]]{{\[}}%[[VAL_363]]] : (memref<?x!sycl_accessor_1_f32_rw_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xf32, 4>
// CHECK-NEXT:        %[[VAL_377:.*]] = affine.load %[[VAL_376]][0] : memref<?xf32, 4>
// CHECK-NEXT:        %[[VAL_378:.*]] = arith.mulf %[[VAL_377]], %[[VAL_371]] : f32
// CHECK-NEXT:        affine.store %[[VAL_378]], %[[VAL_376]][0] : memref<?xf32, 4>
// CHECK-NEXT:        return
    cgh.parallel_for<Kernel>(range, [=](id<1> i){
				      acc[i] *= array[static_cast<unsigned>(i)];
				    });
  });

  return 0;
}
