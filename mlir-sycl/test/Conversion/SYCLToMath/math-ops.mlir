// RUN: sycl-mlir-opt -convert-sycl-to-math %s | FileCheck %s

// CHECK-LABEL: test_math_ops_float
func.func @test_math_ops_float(%arg0 : f32, %arg1 : f32, %arg2 : f32) {
  // CHECK: %{{.*}} = math.ceil %arg0 : f32
  %c0 = sycl.math.ceil %arg0 : f32
  // CHECK: %{{.*}} = math.copysign %arg0, %arg1 : f32
  %c1 = sycl.math.copysign %arg0, %arg1 : f32
  // CHECK: %{{.*}} = math.cos %arg0 : f32
  %c2 = sycl.math.cos %arg0 : f32
  // CHECK: %{{.*}} = math.exp2 %arg0 : f32
  %e0 = sycl.math.exp2 %arg0 : f32
  // CHECK: %{{.*}} = math.expm1 %arg0 : f32
  %e1 = sycl.math.expm1 %arg0 : f32
  // CHECK: %{{.*}} = math.exp %arg0 : f32
  %e2 = sycl.math.exp %arg0 : f32
  // CHECK: %{{.*}} = math.absf %arg0 : f32
  %f0 = sycl.math.fabs %arg0 : f32
  // CHECK: %{{.*}} = math.floor %arg0 : f32
  %f1 = sycl.math.floor %arg0 : f32
  // CHECK: %{{.*}} = math.fma %arg0, %arg1, %arg2 : f32
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : f32
  // CHECK: %{{.*}} = math.log10 %arg0 : f32
  %l0 = sycl.math.log10 %arg0 : f32
  // CHECK: %{{.*}} = math.log2 %arg0 : f32
  %l1 = sycl.math.log2 %arg0 : f32
  // CHECK: %{{.*}} = math.log %arg0 : f32
  %l2 = sycl.math.log %arg0 : f32
  // CHECK: %{{.*}} = math.powf %arg0, %arg1 : f32
  %p0 = sycl.math.pow %arg0, %arg1 : f32
  // CHECK: %{{.*}} = math.round %arg0 : f32
  %r0 = sycl.math.round %arg0 : f32
  // CHECK: %{{.*}} = math.rsqrt %arg0 : f32
  %r1 = sycl.math.rsqrt %arg0 : f32
  // CHECK: %{{.*}} = math.sin %arg0 : f32
  %s0 = sycl.math.sin %arg0 : f32
  // CHECK: %{{.*}} = math.sqrt %arg0 : f32
  %s1 = sycl.math.sqrt %arg0 : f32
  // CHECK: %{{.*}} = math.trunc %arg0 : f32
  %t0 = sycl.math.trunc %arg0 : f32

  return
}

// CHECK-LABEL: test_math_ops_double
func.func @test_math_ops_double(%arg0 : f64, %arg1 : f64, %arg2 : f64) {
  // CHECK: %{{.*}} = math.ceil %arg0 : f64
  %c0 = sycl.math.ceil %arg0 : f64
  // CHECK: %{{.*}} = math.copysign %arg0, %arg1 : f64
  %c1 = sycl.math.copysign %arg0, %arg1 : f64
  // CHECK: %{{.*}} = math.cos %arg0 : f64
  %c2 = sycl.math.cos %arg0 : f64
  // CHECK: %{{.*}} = math.exp2 %arg0 : f64
  %e0 = sycl.math.exp2 %arg0 : f64
  // CHECK: %{{.*}} = math.expm1 %arg0 : f64
  %e1 = sycl.math.expm1 %arg0 : f64
  // CHECK: %{{.*}} = math.exp %arg0 : f64
  %e2 = sycl.math.exp %arg0 : f64
  // CHECK: %{{.*}} = math.absf %arg0 : f64
  %f0 = sycl.math.fabs %arg0 : f64
  // CHECK: %{{.*}} = math.floor %arg0 : f64
  %f1 = sycl.math.floor %arg0 : f64
  // CHECK: %{{.*}} = math.fma %arg0, %arg1, %arg2 : f64
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : f64
  // CHECK: %{{.*}} = math.log10 %arg0 : f64
  %l0 = sycl.math.log10 %arg0 : f64
  // CHECK: %{{.*}} = math.log2 %arg0 : f64
  %l1 = sycl.math.log2 %arg0 : f64
  // CHECK: %{{.*}} = math.log %arg0 : f64
  %l2 = sycl.math.log %arg0 : f64
  // CHECK: %{{.*}} = math.powf %arg0, %arg1 : f64
  %p0 = sycl.math.pow %arg0, %arg1 : f64
  // CHECK: %{{.*}} = math.round %arg0 : f64
  %r0 = sycl.math.round %arg0 : f64
  // CHECK: %{{.*}} = math.rsqrt %arg0 : f64
  %r1 = sycl.math.rsqrt %arg0 : f64
  // CHECK: %{{.*}} = math.sin %arg0 : f64
  %s0 = sycl.math.sin %arg0 : f64
  // CHECK: %{{.*}} = math.sqrt %arg0 : f64
  %s1 = sycl.math.sqrt %arg0 : f64
  // CHECK: %{{.*}} = math.trunc %arg0 : f64
  %t0 = sycl.math.trunc %arg0 : f64

  return
}

!sycl_half = !sycl.half<(f16)>

// CHECK-LABEL: func.func @test_math_ops_half(
// CHECK-SAME:    %[[VAL_0:.*]]: !sycl_half, %[[VAL_1:.*]]: !sycl_half, %[[VAL_2:.*]]: !sycl_half) {
func.func @test_math_ops_half(%arg0 : !sycl_half, %arg1 : !sycl_half, %arg2 : !sycl_half) {
  // CHECK:  %[[VAL_3:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_4:.*]] = math.ceil %[[VAL_3]] : f16
  // CHECK:  %[[VAL_5:.*]] = sycl.mlir.wrap %[[VAL_4]] : f16 to !sycl_half
  %c0 = sycl.math.ceil %arg0 : !sycl_half

  // CHECK:  %[[VAL_6:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_7:.*]] = sycl.mlir.unwrap %[[VAL_1]] : !sycl_half to f16
  // CHECK:  %[[VAL_8:.*]] = math.copysign %[[VAL_6]], %[[VAL_7]] : f16
  // CHECK:  %[[VAL_9:.*]] = sycl.mlir.wrap %[[VAL_8]] : f16 to !sycl_half
  %c1 = sycl.math.copysign %arg0, %arg1 : !sycl_half

  // CHECK:  %[[VAL_10:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_11:.*]] = math.cos %[[VAL_10]] : f16
  // CHECK:  %[[VAL_12:.*]] = sycl.mlir.wrap %[[VAL_11]] : f16 to !sycl_half
  %c2 = sycl.math.cos %arg0 : !sycl_half

  // CHECK:  %[[VAL_13:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_14:.*]] = math.exp2 %[[VAL_13]] : f16
  // CHECK:  %[[VAL_15:.*]] = sycl.mlir.wrap %[[VAL_14]] : f16 to !sycl_half
  %e0 = sycl.math.exp2 %arg0 : !sycl_half

  // CHECK:  %[[VAL_16:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_17:.*]] = math.expm1 %[[VAL_16]] : f16
  // CHECK:  %[[VAL_18:.*]] = sycl.mlir.wrap %[[VAL_17]] : f16 to !sycl_half
  %e1 = sycl.math.expm1 %arg0 : !sycl_half

  // CHECK:  %[[VAL_19:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_20:.*]] = math.exp %[[VAL_19]] : f16
  // CHECK:  %[[VAL_21:.*]] = sycl.mlir.wrap %[[VAL_20]] : f16 to !sycl_half
  %e2 = sycl.math.exp %arg0 : !sycl_half

  // CHECK:  %[[VAL_22:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_23:.*]] = math.absf %[[VAL_22]] : f16
  // CHECK:  %[[VAL_24:.*]] = sycl.mlir.wrap %[[VAL_23]] : f16 to !sycl_half
  %f0 = sycl.math.fabs %arg0 : !sycl_half

  // CHECK:  %[[VAL_25:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_26:.*]] = math.floor %[[VAL_25]] : f16
  // CHECK:  %[[VAL_27:.*]] = sycl.mlir.wrap %[[VAL_26]] : f16 to !sycl_half
  %f1 = sycl.math.floor %arg0 : !sycl_half

  // CHECK:  %[[VAL_28:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_29:.*]] = sycl.mlir.unwrap %[[VAL_1]] : !sycl_half to f16
  // CHECK:  %[[VAL_30:.*]] = sycl.mlir.unwrap %[[VAL_2]] : !sycl_half to f16
  // CHECK:  %[[VAL_31:.*]] = math.fma %[[VAL_28]], %[[VAL_29]], %[[VAL_30]] : f16
  // CHECK:  %[[VAL_32:.*]] = sycl.mlir.wrap %[[VAL_31]] : f16 to !sycl_half
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : !sycl_half

  // CHECK:  %[[VAL_33:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_34:.*]] = math.log10 %[[VAL_33]] : f16
  // CHECK:  %[[VAL_35:.*]] = sycl.mlir.wrap %[[VAL_34]] : f16 to !sycl_half
  %l0 = sycl.math.log10 %arg0 : !sycl_half

  // CHECK:  %[[VAL_36:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_37:.*]] = math.log2 %[[VAL_36]] : f16
  // CHECK:  %[[VAL_38:.*]] = sycl.mlir.wrap %[[VAL_37]] : f16 to !sycl_half
  %l1 = sycl.math.log2 %arg0 : !sycl_half

  // CHECK:  %[[VAL_39:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_40:.*]] = math.log %[[VAL_39]] : f16
  // CHECK:  %[[VAL_41:.*]] = sycl.mlir.wrap %[[VAL_40]] : f16 to !sycl_half
  %l2 = sycl.math.log %arg0 : !sycl_half

  // CHECK:  %[[VAL_42:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_43:.*]] = sycl.mlir.unwrap %[[VAL_1]] : !sycl_half to f16
  // CHECK:  %[[VAL_44:.*]] = math.powf %[[VAL_42]], %[[VAL_43]] : f16
  // CHECK:  %[[VAL_45:.*]] = sycl.mlir.wrap %[[VAL_44]] : f16 to !sycl_half
  %p0 = sycl.math.pow %arg0, %arg1 : !sycl_half

  // CHECK:  %[[VAL_46:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_47:.*]] = math.round %[[VAL_46]] : f16
  // CHECK:  %[[VAL_48:.*]] = sycl.mlir.wrap %[[VAL_47]] : f16 to !sycl_half
  %r0 = sycl.math.round %arg0 : !sycl_half

  // CHECK:  %[[VAL_49:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_50:.*]] = math.rsqrt %[[VAL_49]] : f16
  // CHECK:  %[[VAL_51:.*]] = sycl.mlir.wrap %[[VAL_50]] : f16 to !sycl_half
  %r1 = sycl.math.rsqrt %arg0 : !sycl_half

  // CHECK:  %[[VAL_52:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_53:.*]] = math.sin %[[VAL_52]] : f16
  // CHECK:  %[[VAL_54:.*]] = sycl.mlir.wrap %[[VAL_53]] : f16 to !sycl_half
  %s0 = sycl.math.sin %arg0 : !sycl_half

  // CHECK:  %[[VAL_55:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_56:.*]] = math.sqrt %[[VAL_55]] : f16
  // CHECK:  %[[VAL_57:.*]] = sycl.mlir.wrap %[[VAL_56]] : f16 to !sycl_half
  %s1 = sycl.math.sqrt %arg0 : !sycl_half

  // CHECK:  %[[VAL_58:.*]] = sycl.mlir.unwrap %[[VAL_0]] : !sycl_half to f16
  // CHECK:  %[[VAL_59:.*]] = math.trunc %[[VAL_58]] : f16
  // CHECK:  %[[VAL_60:.*]] = sycl.mlir.wrap %[[VAL_59]] : f16 to !sycl_half
  %t0 = sycl.math.trunc %arg0 : !sycl_half
  
  return
}
