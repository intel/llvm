// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - -fpreview-breaking-changes | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK:           func.func @_Z4vnotN4sycl3_V13vecIjLi2EEE(%[[VAL_151:.*]]: memref<?x!sycl_vec_i32_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_vec_i32_2_, llvm.noundef}) -> !sycl_vec_i32_2_ attributes {{.*}} {
// CHECK:             %[[VAL_152:.*]] = memref.memory_space_cast %[[VAL_151]] : memref<?x!sycl_vec_i32_2_> to memref<?x!sycl_vec_i32_2_, 4>
// CHECK:             %[[VAL_153:.*]] = sycl.call @"operator~"(%[[VAL_152]]) {MangledFunctionName = @_ZNK4sycl3_V13vecIjLi2EEcoIjEENSt9enable_ifIXaantsr3stdE19is_floating_point_vINS0_6detail10vec_helperIT_E7RetTypeEEaantL_ZNS2_20IsUsingArrayOnDeviceEEntL_ZNS2_18IsUsingArrayOnHostEEES2_E4typeEv, TypeName = @vec} : (memref<?x!sycl_vec_i32_2_, 4>) -> !sycl_vec_i32_2_
// CHECK:             return %[[VAL_153]] : !sycl_vec_i32_2_
// CHECK:           }

// CHECK:           func.func @_ZNK4sycl3_V13vecIjLi2EEcoIjEENSt9enable_ifIXaantsr3stdE19is_floating_point_vINS0_6detail10vec_helperIT_E7RetTypeEEaantL_ZNS2_20IsUsingArrayOnDeviceEEntL_ZNS2_18IsUsingArrayOnHostEEES2_E4typeEv(%[[VAL_154:.*]]: memref<?x!sycl_vec_i32_2_, 4> {llvm.align = 8 : i64, llvm.dereferenceable_or_null = 8 : i64, llvm.noundef}) -> !sycl_vec_i32_2_ attributes {{.*}} {
// CHECK:             %[[VAL_155:.*]] = arith.constant dense<-1> : vector<2xi32>
// CHECK:             %[[VAL_156:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_157:.*]] = memref.alloca() : memref<1x!sycl_vec_i32_2_>
// CHECK:             %[[VAL_158:.*]] = memref.cast %[[VAL_157]] : memref<1x!sycl_vec_i32_2_> to memref<?x!sycl_vec_i32_2_>
// CHECK:             %[[VAL_159:.*]] = memref.alloca() : memref<1x!sycl_vec_i32_2_>
// CHECK:             %[[VAL_160:.*]] = memref.cast %[[VAL_159]] : memref<1x!sycl_vec_i32_2_> to memref<?x!sycl_vec_i32_2_>
// CHECK:             %[[VAL_161:.*]] = "polygeist.subindex"(%[[VAL_154]], %[[VAL_156]]) : (memref<?x!sycl_vec_i32_2_, 4>, index) -> memref<?xvector<2xi32>, 4>
// CHECK:             %[[VAL_162:.*]] = affine.load %[[VAL_161]][0] : memref<?xvector<2xi32>, 4>
// CHECK:             %[[VAL_163:.*]] = arith.xori %[[VAL_162]], %[[VAL_155]] : vector<2xi32>
// CHECK:             %[[VAL_164:.*]] = memref.memory_space_cast %[[VAL_160]] : memref<?x!sycl_vec_i32_2_> to memref<?x!sycl_vec_i32_2_, 4>
// CHECK:             sycl.constructor @vec(%[[VAL_164]], %[[VAL_163]]) {MangledFunctionName = @_ZN4sycl3_V13vecIjLi2EEC1IDv2_jvEES4_} : (memref<?x!sycl_vec_i32_2_, 4>, vector<2xi32>)
// CHECK:             %[[VAL_165:.*]] = memref.memory_space_cast %[[VAL_158]] : memref<?x!sycl_vec_i32_2_> to memref<?x!sycl_vec_i32_2_, 4>
// CHECK:             %[[VAL_166:.*]] = memref.memory_space_cast %[[VAL_160]] : memref<?x!sycl_vec_i32_2_> to memref<?x!sycl_vec_i32_2_, 4>
// CHECK:             sycl.constructor @vec(%[[VAL_165]], %[[VAL_166]]) {MangledFunctionName = @_ZN4sycl3_V13vecIjLi2EEC1EOS2_} : (memref<?x!sycl_vec_i32_2_, 4>, memref<?x!sycl_vec_i32_2_, 4>)
// CHECK:             %[[VAL_167:.*]] = affine.load %[[VAL_157]][0] : memref<1x!sycl_vec_i32_2_>
// CHECK:             return %[[VAL_167]] : !sycl_vec_i32_2_
// CHECK:           }

SYCL_EXTERNAL sycl::uint2 vnot(sycl::uint2 v) {
  return ~v;
}
