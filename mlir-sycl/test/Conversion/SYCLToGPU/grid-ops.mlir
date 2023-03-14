// RUN: sycl-mlir-opt -convert-sycl-to-gpu %s -o - | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   func.func @test_num_work_items() -> !sycl_range_1_ {
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.grid_dim  x
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : index to i64
// CHECK-NEXT:      %[[VAL_5:.*]] = sycl.range.get %[[VAL_0]]{{\[}}%[[VAL_2]]] {ArgumentTypes = [memref<1x!sycl_range_1_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @range} : (memref<1x!sycl_range_1_>, i32) -> memref<1xi64>
// CHECK-NEXT:      memref.store %[[VAL_4]], %[[VAL_5]]{{\[}}%[[VAL_1]]] : memref<1xi64>
// CHECK-NEXT:      %[[VAL_6:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_1]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:      return %[[VAL_6]] : !sycl_range_1_
// CHECK-NEXT:    }
func.func @test_num_work_items() -> !sycl_range_1_ {
  %0 = sycl.num_work_items : !sycl_range_1_
  return %0 : !sycl_range_1_
}

// CHECK-LABEL:   func.func @test_num_work_items_dim(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i32) -> index {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant dense<0> : vector<3xindex>
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.grid_dim  x
// CHECK-NEXT:      %[[VAL_4:.*]] = vector.insert %[[VAL_3]], %[[VAL_1]] [0] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_5:.*]] = gpu.grid_dim  y
// CHECK-NEXT:      %[[VAL_6:.*]] = vector.insert %[[VAL_5]], %[[VAL_4]] [1] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_7:.*]] = gpu.grid_dim  z
// CHECK-NEXT:      %[[VAL_8:.*]] = vector.insert %[[VAL_7]], %[[VAL_6]] [2] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_9:.*]] = vector.extractelement %[[VAL_8]]{{\[}}%[[VAL_0]] : i32] : vector<3xindex>
// CHECK-NEXT:      return %[[VAL_9]] : index
// CHECK-NEXT:    }
func.func @test_num_work_items_dim(%i: i32) -> index {
  %0 = sycl.num_work_items %i : index
  return %0 : index
}

// CHECK-LABEL:   func.func @test_global_id() -> !sycl_id_2_ {
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.global_id  x
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : index to i64
// CHECK-NEXT:      %[[VAL_5:.*]] = sycl.id.get %[[VAL_0]]{{\[}}%[[VAL_2]]] {ArgumentTypes = [memref<1x!sycl_id_2_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @id} : (memref<1x!sycl_id_2_>, i32) -> memref<2xi64>
// CHECK-NEXT:      memref.store %[[VAL_4]], %[[VAL_5]]{{\[}}%[[VAL_1]]] : memref<2xi64>
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[VAL_7:.*]] = gpu.global_id  y
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : index to i64
// CHECK-NEXT:      %[[VAL_9:.*]] = sycl.id.get %[[VAL_0]]{{\[}}%[[VAL_6]]] {ArgumentTypes = [memref<1x!sycl_id_2_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @id} : (memref<1x!sycl_id_2_>, i32) -> memref<2xi64>
// CHECK-NEXT:      memref.store %[[VAL_8]], %[[VAL_9]]{{\[}}%[[VAL_1]]] : memref<2xi64>
// CHECK-NEXT:      %[[VAL_10:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_1]]] : memref<1x!sycl_id_2_>
// CHECK-NEXT:      return %[[VAL_10]] : !sycl_id_2_
// CHECK-NEXT:    }
func.func @test_global_id() -> !sycl_id_2_ {
  %0 = sycl.global_id : !sycl_id_2_
  return %0 : !sycl_id_2_
}

// CHECK-LABEL:   func.func @test_global_id_dim(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32) -> index {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant dense<0> : vector<3xindex>
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.global_id  x
// CHECK-NEXT:      %[[VAL_4:.*]] = vector.insert %[[VAL_3]], %[[VAL_1]] [0] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_5:.*]] = gpu.global_id  y
// CHECK-NEXT:      %[[VAL_6:.*]] = vector.insert %[[VAL_5]], %[[VAL_4]] [1] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_7:.*]] = gpu.global_id  z
// CHECK-NEXT:      %[[VAL_8:.*]] = vector.insert %[[VAL_7]], %[[VAL_6]] [2] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_9:.*]] = vector.extractelement %[[VAL_8]]{{\[}}%[[VAL_0]] : i32] : vector<3xindex>
// CHECK-NEXT:      return %[[VAL_9]] : index
// CHECK-NEXT:    }
func.func @test_global_id_dim(%i: i32) -> index {
  %0 = sycl.global_id %i : index
  return %0 : index
}

// CHECK-LABEL:   func.func @test_local_id() -> !sycl_id_3_ {
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.thread_id  x
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : index to i64
// CHECK-NEXT:      %[[VAL_5:.*]] = sycl.id.get %[[VAL_0]]{{\[}}%[[VAL_2]]] {ArgumentTypes = [memref<1x!sycl_id_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @id} : (memref<1x!sycl_id_3_>, i32) -> memref<3xi64>
// CHECK-NEXT:      memref.store %[[VAL_4]], %[[VAL_5]]{{\[}}%[[VAL_1]]] : memref<3xi64>
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[VAL_7:.*]] = gpu.thread_id  y
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : index to i64
// CHECK-NEXT:      %[[VAL_9:.*]] = sycl.id.get %[[VAL_0]]{{\[}}%[[VAL_6]]] {ArgumentTypes = [memref<1x!sycl_id_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @id} : (memref<1x!sycl_id_3_>, i32) -> memref<3xi64>
// CHECK-NEXT:      memref.store %[[VAL_8]], %[[VAL_9]]{{\[}}%[[VAL_1]]] : memref<3xi64>
// CHECK-NEXT:      %[[VAL_10:.*]] = arith.constant 2 : i32
// CHECK-NEXT:      %[[VAL_11:.*]] = gpu.thread_id  z
// CHECK-NEXT:      %[[VAL_12:.*]] = arith.index_cast %[[VAL_11]] : index to i64
// CHECK-NEXT:      %[[VAL_13:.*]] = sycl.id.get %[[VAL_0]]{{\[}}%[[VAL_10]]] {ArgumentTypes = [memref<1x!sycl_id_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @id} : (memref<1x!sycl_id_3_>, i32) -> memref<3xi64>
// CHECK-NEXT:      memref.store %[[VAL_12]], %[[VAL_13]]{{\[}}%[[VAL_1]]] : memref<3xi64>
// CHECK-NEXT:      %[[VAL_14:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_1]]] : memref<1x!sycl_id_3_>
// CHECK-NEXT:      return %[[VAL_14]] : !sycl_id_3_
// CHECK-NEXT:    }
func.func @test_local_id() -> !sycl_id_3_ {
  %0 = sycl.local_id : !sycl_id_3_
  return %0 : !sycl_id_3_
}

// CHECK-LABEL:   func.func @test_local_id_dim(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32) -> index {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant dense<0> : vector<3xindex>
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.thread_id  x
// CHECK-NEXT:      %[[VAL_4:.*]] = vector.insert %[[VAL_3]], %[[VAL_1]] [0] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_5:.*]] = gpu.thread_id  y
// CHECK-NEXT:      %[[VAL_6:.*]] = vector.insert %[[VAL_5]], %[[VAL_4]] [1] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_7:.*]] = gpu.thread_id  z
// CHECK-NEXT:      %[[VAL_8:.*]] = vector.insert %[[VAL_7]], %[[VAL_6]] [2] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_9:.*]] = vector.extractelement %[[VAL_8]]{{\[}}%[[VAL_0]] : i32] : vector<3xindex>
// CHECK-NEXT:      return %[[VAL_9]] : index
// CHECK-NEXT:    }
func.func @test_local_id_dim(%i: i32) -> index {
  %0 = sycl.local_id %i : index
  return %0 : index
}

// CHECK-LABEL:   func.func @test_work_group_size() -> !sycl_range_3_ {
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.block_dim  x
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : index to i64
// CHECK-NEXT:      %[[VAL_5:.*]] = sycl.range.get %[[VAL_0]]{{\[}}%[[VAL_2]]] {ArgumentTypes = [memref<1x!sycl_range_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @range} : (memref<1x!sycl_range_3_>, i32) -> memref<3xi64>
// CHECK-NEXT:      memref.store %[[VAL_4]], %[[VAL_5]]{{\[}}%[[VAL_1]]] : memref<3xi64>
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[VAL_7:.*]] = gpu.block_dim  y
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : index to i64
// CHECK-NEXT:      %[[VAL_9:.*]] = sycl.range.get %[[VAL_0]]{{\[}}%[[VAL_6]]] {ArgumentTypes = [memref<1x!sycl_range_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @range} : (memref<1x!sycl_range_3_>, i32) -> memref<3xi64>
// CHECK-NEXT:      memref.store %[[VAL_8]], %[[VAL_9]]{{\[}}%[[VAL_1]]] : memref<3xi64>
// CHECK-NEXT:      %[[VAL_10:.*]] = arith.constant 2 : i32
// CHECK-NEXT:      %[[VAL_11:.*]] = gpu.block_dim  z
// CHECK-NEXT:      %[[VAL_12:.*]] = arith.index_cast %[[VAL_11]] : index to i64
// CHECK-NEXT:      %[[VAL_13:.*]] = sycl.range.get %[[VAL_0]]{{\[}}%[[VAL_10]]] {ArgumentTypes = [memref<1x!sycl_range_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @range} : (memref<1x!sycl_range_3_>, i32) -> memref<3xi64>
// CHECK-NEXT:      memref.store %[[VAL_12]], %[[VAL_13]]{{\[}}%[[VAL_1]]] : memref<3xi64>
// CHECK-NEXT:      %[[VAL_14:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_1]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:      return %[[VAL_14]] : !sycl_range_3_
// CHECK-NEXT:    }
func.func @test_work_group_size() -> !sycl_range_3_ {
  %0 = sycl.work_group_size : !sycl_range_3_
  return %0 : !sycl_range_3_
}

// CHECK-LABEL:   func.func @test_work_group_size_dim(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i32) -> index {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant dense<0> : vector<3xindex>
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.block_dim  x
// CHECK-NEXT:      %[[VAL_4:.*]] = vector.insert %[[VAL_3]], %[[VAL_1]] [0] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_5:.*]] = gpu.block_dim  y
// CHECK-NEXT:      %[[VAL_6:.*]] = vector.insert %[[VAL_5]], %[[VAL_4]] [1] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_7:.*]] = gpu.block_dim  z
// CHECK-NEXT:      %[[VAL_8:.*]] = vector.insert %[[VAL_7]], %[[VAL_6]] [2] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_9:.*]] = vector.extractelement %[[VAL_8]]{{\[}}%[[VAL_0]] : i32] : vector<3xindex>
// CHECK-NEXT:      return %[[VAL_9]] : index
// CHECK-NEXT:    }
func.func @test_work_group_size_dim(%i: i32) -> index {
  %0 = sycl.work_group_size %i : index
  return %0 : index
}

// CHECK-LABEL:   func.func @test_work_group_id() -> !sycl_id_1_ {
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.block_id  x
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : index to i64
// CHECK-NEXT:      %[[VAL_5:.*]] = sycl.id.get %[[VAL_0]]{{\[}}%[[VAL_2]]] {ArgumentTypes = [memref<1x!sycl_id_1_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @id} : (memref<1x!sycl_id_1_>, i32) -> memref<1xi64>
// CHECK-NEXT:      memref.store %[[VAL_4]], %[[VAL_5]]{{\[}}%[[VAL_1]]] : memref<1xi64>
// CHECK-NEXT:      %[[VAL_6:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_1]]] : memref<1x!sycl_id_1_>
// CHECK-NEXT:      return %[[VAL_6]] : !sycl_id_1_
// CHECK-NEXT:    }
func.func @test_work_group_id() -> !sycl_id_1_ {
  %0 = sycl.work_group_id : !sycl_id_1_
  return %0 : !sycl_id_1_
}

// CHECK-LABEL:   func.func @test_work_group_id_dim(
// CHECK-SAME:                                      %[[VAL_0:.*]]: i32) -> index {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant dense<0> : vector<3xindex>
// CHECK-NEXT:      %[[VAL_3:.*]] = gpu.block_id  x
// CHECK-NEXT:      %[[VAL_4:.*]] = vector.insert %[[VAL_3]], %[[VAL_1]] [0] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_5:.*]] = gpu.block_id  y
// CHECK-NEXT:      %[[VAL_6:.*]] = vector.insert %[[VAL_5]], %[[VAL_4]] [1] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_7:.*]] = gpu.block_id  z
// CHECK-NEXT:      %[[VAL_8:.*]] = vector.insert %[[VAL_7]], %[[VAL_6]] [2] : index into vector<3xindex>
// CHECK-NEXT:      %[[VAL_9:.*]] = vector.extractelement %[[VAL_8]]{{\[}}%[[VAL_0]] : i32] : vector<3xindex>
// CHECK-NEXT:      return %[[VAL_9]] : index
// CHECK-NEXT:    }
func.func @test_work_group_id_dim(%i: i32) -> index {
  %0 = sycl.work_group_id %i : index
  return %0 : index
}

// CHECK-LABEL:   func.func @test_num_sub_groups() -> i32 {
// CHECK-NEXT:      %[[VAL_0:.*]] = gpu.num_subgroups : index
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.index_cast %[[VAL_0]] : index to i32
// CHECK-NEXT:      return %[[VAL_1]] : i32
// CHECK-NEXT:    }
func.func @test_num_sub_groups() -> i32 {
  %0 = sycl.num_sub_groups : i32
  return %0 : i32
}

// CHECK-LABEL:   func.func @test_sub_group_size() -> i32 {
// CHECK-NEXT:      %[[VAL_0:.*]] = gpu.subgroup_size : index
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.index_cast %[[VAL_0]] : index to i32
// CHECK-NEXT:      return %[[VAL_1]] : i32
// CHECK-NEXT:    }
func.func @test_sub_group_size() -> i32 {
  %0 = sycl.sub_group_size : i32
  return %0 : i32
}

// CHECK-LABEL:   func.func @test_sub_group_id() -> i32 {
// CHECK-NEXT:      %[[VAL_0:.*]] = gpu.subgroup_id : index
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.index_cast %[[VAL_0]] : index to i32
// CHECK-NEXT:      return %[[VAL_1]] : i32
// CHECK-NEXT:    }
func.func @test_sub_group_id() -> i32 {
  %0 = sycl.sub_group_id : i32
  return %0 : i32
}
