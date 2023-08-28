// RUN: cgeist  %s -O0 -w --function=* -S | FileCheck %s

typedef int int3 __attribute__((ext_vector_type(3)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef unsigned long ulong4 __attribute__((ext_vector_type(4)));
typedef long long4 __attribute__((ext_vector_type(4)));

template <typename T>
struct Compare {
  T lt;
  T gt;
  T le;
  T ge;
  T eq;
  T ne;
};

template <typename T, typename RT>
Compare<RT> compare(T lhs, T rhs) {
  return {lhs < rhs, lhs > rhs, lhs <= rhs, lhs >= rhs, lhs == rhs, lhs != rhs};
}

// CHECK-LABEL:   func.func @_Z7compareIfbE7CompareIT0_ET_S3_(
// CHECK-SAME:                                                %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32)
// CHECK:           %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_8:.*]] = arith.cmpf ogt, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_11:.*]] = arith.cmpf ole, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_14:.*]] = arith.cmpf oge, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_17:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_20:.*]] = arith.cmpf une, %[[VAL_0]], %[[VAL_1]] : f32
template Compare<bool> compare(float lhs, float rhs);

// CHECK-LABEL:   func.func @_Z7compareIdbE7CompareIT0_ET_S3_(
// CHECK-SAME:                                                %[[VAL_0:.*]]: f64, %[[VAL_1:.*]]: f64)
// CHECK:           %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_8:.*]] = arith.cmpf ogt, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_11:.*]] = arith.cmpf ole, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_14:.*]] = arith.cmpf oge, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_17:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_20:.*]] = arith.cmpf une, %[[VAL_0]], %[[VAL_1]] : f64
template Compare<bool> compare(double lhs, double rhs);

// CHECK-LABEL:   func.func @_Z7compareIibE7CompareIT0_ET_S3_(
// CHECK-SAME:                                                %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32)
// CHECK:           %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sle, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_14:.*]] = arith.cmpi sge, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_20:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_1]] : i32
template Compare<bool> compare(int lhs, int rhs);

// CHECK-LABEL:   func.func @_Z7compareIjbE7CompareIT0_ET_S3_(
// CHECK-SAME:                                                %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32)
// CHECK:           %[[VAL_5:.*]] = arith.cmpi ult, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_8:.*]] = arith.cmpi ugt, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_11:.*]] = arith.cmpi ule, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_14:.*]] = arith.cmpi uge, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_20:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_1]] : i32
template Compare<bool> compare(unsigned lhs, unsigned rhs);

// CHECK-LABEL:   func.func @_Z7compareIPvbE7CompareIT0_ET_S4_(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr)
// CHECK:           %[[VAL_5:.*]] = llvm.icmp "ult" %[[VAL_0]], %[[VAL_1]] : !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.icmp "ugt" %[[VAL_0]], %[[VAL_1]] : !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = llvm.icmp "ule" %[[VAL_0]], %[[VAL_1]] : !llvm.ptr
// CHECK:           %[[VAL_14:.*]] = llvm.icmp "uge" %[[VAL_0]], %[[VAL_1]] : !llvm.ptr
// CHECK:           %[[VAL_17:.*]] = llvm.icmp "eq" %[[VAL_0]], %[[VAL_1]] : !llvm.ptr
// CHECK:           %[[VAL_20:.*]] = llvm.icmp "ne" %[[VAL_0]], %[[VAL_1]] : !llvm.ptr
template Compare<bool> compare(void *lhs, void *rhs);

// CHECK-LABEL:   func.func @_Z7compareIPibE7CompareIT0_ET_S4_(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: memref<?xi32>, %[[VAL_1:.*]]: memref<?xi32>)
// CHECK:           %[[VAL_5:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?xi32>) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?xi32>) -> !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.icmp "ult" %[[VAL_5]], %[[VAL_6]] : !llvm.ptr
// CHECK:           %[[VAL_10:.*]] = llvm.icmp "ugt" %[[VAL_5]], %[[VAL_6]] : !llvm.ptr
// CHECK:           %[[VAL_13:.*]] = llvm.icmp "ule" %[[VAL_5]], %[[VAL_6]] : !llvm.ptr
// CHECK:           %[[VAL_16:.*]] = llvm.icmp "uge" %[[VAL_5]], %[[VAL_6]] : !llvm.ptr
// CHECK:           %[[VAL_19:.*]] = llvm.icmp "eq" %[[VAL_5]], %[[VAL_6]] : !llvm.ptr
// CHECK:           %[[VAL_22:.*]] = llvm.icmp "ne" %[[VAL_5]], %[[VAL_6]] : !llvm.ptr
template Compare<bool> compare(int *lhs, int *rhs);

// CHECK-LABEL:   func.func @_Z7compareIDv4_iS0_E7CompareIT0_ET_S4_(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: vector<4xi32>, %[[VAL_1:.*]]: vector<4xi32>)
// CHECK:           %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : vector<4xi32>
// CHECK:           %[[VAL_6:.*]] = arith.extsi %[[VAL_5]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_0]], %[[VAL_1]] : vector<4xi32>
// CHECK:           %[[VAL_9:.*]] = arith.extsi %[[VAL_8]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sle, %[[VAL_0]], %[[VAL_1]] : vector<4xi32>
// CHECK:           %[[VAL_12:.*]] = arith.extsi %[[VAL_11]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_14:.*]] = arith.cmpi sge, %[[VAL_0]], %[[VAL_1]] : vector<4xi32>
// CHECK:           %[[VAL_15:.*]] = arith.extsi %[[VAL_14]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_1]] : vector<4xi32>
// CHECK:           %[[VAL_18:.*]] = arith.extsi %[[VAL_17]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_20:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_1]] : vector<4xi32>
// CHECK:           %[[VAL_21:.*]] = arith.extsi %[[VAL_20]] : vector<4xi1> to vector<4xi32>
template Compare<int4> compare(int4 lhs, int4 rhs);

// CHECK-LABEL:   func.func @_Z7compareIDv4_fDv4_iE7CompareIT0_ET_S5_(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: vector<4xf32>, %[[VAL_1:.*]]: vector<4xf32>)
// CHECK:           %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : vector<4xf32>
// CHECK:           %[[VAL_6:.*]] = arith.extsi %[[VAL_5]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_8:.*]] = arith.cmpf ogt, %[[VAL_0]], %[[VAL_1]] : vector<4xf32>
// CHECK:           %[[VAL_9:.*]] = arith.extsi %[[VAL_8]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_11:.*]] = arith.cmpf ole, %[[VAL_0]], %[[VAL_1]] : vector<4xf32>
// CHECK:           %[[VAL_12:.*]] = arith.extsi %[[VAL_11]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_14:.*]] = arith.cmpf oge, %[[VAL_0]], %[[VAL_1]] : vector<4xf32>
// CHECK:           %[[VAL_15:.*]] = arith.extsi %[[VAL_14]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_17:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : vector<4xf32>
// CHECK:           %[[VAL_18:.*]] = arith.extsi %[[VAL_17]] : vector<4xi1> to vector<4xi32>
// CHECK:           %[[VAL_20:.*]] = arith.cmpf une, %[[VAL_0]], %[[VAL_1]] : vector<4xf32>
// CHECK:           %[[VAL_21:.*]] = arith.extsi %[[VAL_20]] : vector<4xi1> to vector<4xi32>
template Compare<int4> compare(float4 lhs, float4 rhs);

// CHECK-LABEL:   func.func @_Z7compareIDv4_mDv4_lE7CompareIT0_ET_S5_(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: memref<?xvector<4xi64>>, %[[VAL_1:.*]]: memref<?xvector<4xi64>>)
// CHECK:           %[[VAL_7:.*]] = arith.cmpi ult, %{{.*}}, %{{.*}} : vector<4xi64>
// CHECK:           %[[VAL_8:.*]] = arith.extsi %[[VAL_7]] : vector<4xi1> to vector<4xi64>
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ugt, %{{.*}}, %{{.*}} : vector<4xi64>
// CHECK:           %[[VAL_13:.*]] = arith.extsi %[[VAL_12]] : vector<4xi1> to vector<4xi64>
// CHECK:           %[[VAL_17:.*]] = arith.cmpi ule, %{{.*}}, %{{.*}} : vector<4xi64>
// CHECK:           %[[VAL_18:.*]] = arith.extsi %[[VAL_17]] : vector<4xi1> to vector<4xi64>
// CHECK:           %[[VAL_22:.*]] = arith.cmpi uge, %{{.*}}, %{{.*}} : vector<4xi64>
// CHECK:           %[[VAL_23:.*]] = arith.extsi %[[VAL_22]] : vector<4xi1> to vector<4xi64>
// CHECK:           %[[VAL_27:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : vector<4xi64>
// CHECK:           %[[VAL_28:.*]] = arith.extsi %[[VAL_27]] : vector<4xi1> to vector<4xi64>
// CHECK:           %[[VAL_32:.*]] = arith.cmpi ne, %{{.*}}, %{{.*}} : vector<4xi64>
// CHECK:           %[[VAL_33:.*]] = arith.extsi %[[VAL_32]] : vector<4xi1> to vector<4xi64>
template Compare<long4> compare(ulong4 lhs, ulong4 rhs);
