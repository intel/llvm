// RUN: polygeist-opt %s --convert-polygeist-to-llvm --split-input-file | FileCheck %s

// CHECK-LABEL:   llvm.func @ptr_ret_static(i64) -> !llvm.ptr

func.func private @ptr_ret_static(%arg0: i64) -> memref<4xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_ret_dynamic(i64) -> !llvm.ptr

func.func private @ptr_ret_dynamic(%arg0: i64) -> memref<?xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_ret_nd_static(i64) -> !llvm.ptr

func.func private @ptr_ret_nd_static(%arg0: i64) -> memref<4x4xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_ret_nd_dynamic(i64) -> !llvm.ptr

func.func private @ptr_ret_nd_dynamic(%arg0: i64) -> memref<?x4x4xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_args_and_ret(!llvm.ptr, !llvm.ptr) -> !llvm.ptr

func.func private @ptr_args_and_ret(%arg0: memref<1xi64>, %arg1: memref<?xi64>) -> memref<?x4x4xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_args_and_ret_with_attrs(!llvm.ptr {llvm.byval = i64}, !llvm.ptr {llvm.byval = i64}) -> !llvm.ptr

func.func private @ptr_args_and_ret_with_attrs(%arg0: memref<1xi64> {llvm.byval = i64},
                                               %arg1: memref<?xi64> {llvm.byval = i64}) -> memref<?x4x4xi64>

// -----

gpu.module @kernels {

// CHECK-LABEL:   llvm.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr {llvm.byval = i64},
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr {llvm.byval = i64}) attributes {gpu.kernel, workgroup_attributions = 0 : i64} {
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:         }

  gpu.func @kernel(%arg0: memref<1xi64> {llvm.byval = i64},
                   %arg1: memref<?xi64> {llvm.byval = i64}) kernel {
    gpu.return
  }
}

// -----

// CHECK-LABEL:   llvm.mlir.global external @global() {addr_space = 0 : i32} : !llvm.array<3 x i64>

memref.global @global : memref<3xi64>

// CHECK-LABEL:   llvm.func @get_global() -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.addressof @global : !llvm.ptr
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
// CHECK-NEXT:      llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @get_global() -> memref<3xi64> {
  %0 = memref.get_global @global : memref<3xi64>
  return %0 : memref<3xi64>
}

// -----

// CHECK-LABEL:   llvm.mlir.global external @global_addrspace() {addr_space = 4 : i32} : !llvm.array<3 x i64>

memref.global @global_addrspace : memref<3xi64, 4>

// CHECK-LABEL:   llvm.func @get_global_addrspace() -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.addressof @global_addrspace : !llvm.ptr<4>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr<4>) -> !llvm.ptr<4>, !llvm.array<3 x i64>
// CHECK-NEXT:      llvm.return %[[VAL_1]] : !llvm.ptr<4>
// CHECK-NEXT:    }

func.func private @get_global_addrspace() -> memref<3xi64, 4> {
  %0 = memref.get_global @global_addrspace : memref<3xi64, 4>
  return %0 : memref<3xi64, 4>
}

// -----

// CHECK-LABEL:   llvm.mlir.global external @global_sycl_addrspace() {addr_space = 1 : i32} : !llvm.array<3 x i64>

memref.global @global_sycl_addrspace : memref<3xi64, #sycl.access.address_space<global>>

// CHECK-LABEL:   llvm.func @get_global_sycl_addrspace() -> !llvm.ptr<1>
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.addressof @global_sycl_addrspace : !llvm.ptr<1>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, !llvm.array<3 x i64>
// CHECK-NEXT:      llvm.return %[[VAL_1]] : !llvm.ptr<1>
// CHECK-NEXT:    }

func.func private @get_global_sycl_addrspace() -> memref<3xi64, #sycl.access.address_space<global>> {
  %0 = memref.get_global @global_sycl_addrspace : memref<3xi64, #sycl.access.address_space<global>>
  return %0 : memref<3xi64, #sycl.access.address_space<global>>
}

// -----

memref.global "private" constant @shape : memref<2xi64> = dense<[2, 2]>

// CHECK-LABEL:   llvm.func @reshape(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @reshape(%arg0: memref<4xi32>) -> memref<2x2xi32> {
  %shape = memref.get_global @shape : memref<2xi64>
  %0 = memref.reshape %arg0(%shape) : (memref<4xi32>, memref<2xi64>) -> memref<2x2xi32>
  return %0 : memref<2x2xi32>
}

// -----

memref.global "private" constant @shape : memref<1xindex>

// CHECK-LABEL:   llvm.func @reshape_dyn(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @reshape_dyn(%arg0: memref<4xi32>) -> memref<?xi32> {
  %shape = memref.get_global @shape : memref<1xindex>
  %0 = memref.reshape %arg0(%shape) : (memref<4xi32>, memref<1xindex>) -> memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// CHECK-LABEL:   llvm.func @alloca()
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x i32 : (i64) -> !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

func.func private @alloca() {
  %0 = memref.alloca() : memref<2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @alloca_nd()
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(60 : index) : i64
// CHECK:           %[[VAL_6:.*]] = llvm.alloca %[[VAL_5]] x i32 : (i64) -> !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

func.func private @alloca_nd() {
  %0 = memref.alloca() : memref<3x10x2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @alloca_aligned()
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

func.func private @alloca_aligned() {
  %0 = memref.alloca() {alignment = 8} : memref<2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @alloca_nd_aligned()
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(60 : index) : i64
// CHECK:           %[[VAL_6:.*]] = llvm.alloca %[[VAL_5]] x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

func.func private @alloca_nd_aligned() {
  %0 = memref.alloca() {alignment = 8} : memref<3x10x2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @mixed_alloca(
// CHECK-SAME:                            %[[VAL_0:.*]]: i64)
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(42 : index) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mul %[[VAL_1]], %[[VAL_0]]  : i64
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x f32 : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(294 : index) : i64
// CHECK:           %[[VAL_9:.*]] = llvm.mul %[[VAL_8]], %[[VAL_0]]  : i64
// CHECK:           %[[VAL_10:.*]] = llvm.alloca %[[VAL_9]] x f32 {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

func.func private @mixed_alloca(%arg0 : index) {
  %0 = memref.alloca(%arg0) : memref<?x42xf32>
  %1 = memref.alloca(%arg0) {alignment = 8} : memref<?x42x7xf32>
  return
}

// -----

// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL:   llvm.func @alloc() attributes {sym_visibility = "private"} {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_0]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           %[[VAL_4:.*]] = llvm.ptrtoint %[[VAL_3]] : !llvm.ptr to i64
// CHECK:           %[[VAL_5:.*]] = llvm.call @malloc(%[[VAL_4]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

func.func private @alloc() {
  %0 = memref.alloc() : memref<2xi32>
  return
}

// -----

// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL:   llvm.func @alloc_nd() attributes {sym_visibility = "private"} {
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(60 : index) : i64
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_5]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           %[[VAL_8:.*]] = llvm.ptrtoint %[[VAL_7]] : !llvm.ptr to i64
// CHECK:           %[[VAL_9:.*]] = llvm.call @malloc(%[[VAL_8]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

func.func private @alloc_nd() {
  %0 = memref.alloc() : memref<3x10x2xi32>
  return
}

// -----

// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL:   llvm.func @mixed_alloc(
// CHECK-SAME:                           %[[VAL_0:.*]]: i64)
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(294 : index) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.mul %[[VAL_4]], %[[VAL_0]]  : i64
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_5]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           %[[VAL_8:.*]] = llvm.ptrtoint %[[VAL_7]] : !llvm.ptr to i64
// CHECK:           %[[VAL_9:.*]] = llvm.call @malloc(%[[VAL_8]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

func.func private @mixed_alloc(%arg0 : index) {
  %0 = memref.alloc(%arg0) : memref<?x42x7xf32>
  return
}


// -----

// CHECK-LABEL:   llvm.func @dealloc(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr)
// CHECK-NEXT:      llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @dealloc(%arg0: memref<?xi32>) {
  memref.dealloc %arg0 : memref<?xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @cast(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @cast(%arg0: memref<2xi32>) -> memref<?xi32> {
  %0 = memref.cast %arg0 : memref<2xi32> to memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// CHECK-LABEL:   llvm.func @load(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> f32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return %[[VAL_3]] : f32
// CHECK-NEXT:    }

func.func private @load(%arg0: memref<100xf32>, %index: index) -> f32 {
  %0 = memref.load %arg0[%index] : memref<100xf32>
  return %0 : f32
}

// -----

// CHECK-LABEL:   llvm.func @load_nd(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                       %[[VAL_1:.*]]: i64,
// CHECK-SAME:                       %[[VAL_2:.*]]: i64) -> f32
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.add %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_5]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr
// CHECK-NEXT:      llvm.return %[[VAL_7]] : f32
// CHECK-NEXT:    }

func.func private @load_nd(%arg0: memref<100x100xf32>, %index0: index, %index1: index) -> f32 {
  %0 = memref.load %arg0[%index0, %index1] : memref<100x100xf32>
  return %0 : f32
}

// -----

// CHECK-LABEL:   llvm.func @load_nd_dyn(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                           %[[VAL_1:.*]]: i64,
// CHECK-SAME:                           %[[VAL_2:.*]]: i64) -> f32
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.add %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_5]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr
// CHECK-NEXT:      llvm.return %[[VAL_7]] : f32
// CHECK-NEXT:    }

func.func private @load_nd_dyn(%arg0: memref<?x100xf32>, %index0: index, %index1: index) -> f32 {
  %0 = memref.load %arg0[%index0, %index1] : memref<?x100xf32>
  return %0 : f32
}

// -----

// CHECK-LABEL:   llvm.func @store(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                     %[[VAL_1:.*]]: i64,
// CHECK-SAME:                     %[[VAL_2:.*]]: f32)
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:      llvm.store %[[VAL_2]], %[[VAL_3]] : f32, !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @store(%arg0: memref<100xf32>, %index: index, %val: f32) {
  memref.store %val, %arg0[%index] : memref<100xf32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @store_nd(
// CHECK-SAME:                        %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                        %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64,
// CHECK-SAME:                        %[[VAL_3:.*]]: f32)
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_1]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.add %[[VAL_5]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_6]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:      llvm.store %[[VAL_3]], %[[VAL_7]] : f32, !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @store_nd(%arg0: memref<100x100xf32>, %index0: index, %index1: index, %val: f32) {
  memref.store %val, %arg0[%index0, %index1] : memref<100x100xf32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @store_nd_dyn(
// CHECK-SAME:                            %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                            %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64,
// CHECK-SAME:                            %[[VAL_3:.*]]: f32)
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_1]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.add %[[VAL_5]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_6]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:      llvm.store %[[VAL_3]], %[[VAL_7]] : f32, !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @store_nd_dyn(%arg0: memref<?x100xf32>, %index0: index, %index1: index, %val: f32) {
  memref.store %val, %arg0[%index0, %index1] : memref<?x100xf32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @impl(!llvm.ptr, i64) -> !llvm.ptr

func.func private @impl(%arg0: memref<?xf32>, %arg1: index) -> memref<?xf32>

// CHECK-LABEL:   llvm.func @call(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.call @impl(%[[VAL_0]], %[[VAL_1]]) : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @call(%arg0: memref<?xf32>, %arg1: index) -> memref<?xf32> {
  %res = func.call @impl(%arg0, %arg1) : (memref<?xf32>, index) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                 %[[VAL_1:.*]]: i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mul %[[VAL_1]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_3]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:      llvm.return %[[VAL_4]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @subindexop_memref(%arg0: memref<4x4xf32>, %arg1: index) -> memref<4xf32> {
  %res = "polygeist.subindex"(%arg0 , %arg1) : (memref<4x4xf32>, index) -> memref<4xf32>
  return %res : memref<4xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref_same_dim(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                          %[[VAL_1:.*]]: i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @subindexop_memref_same_dim(%arg0: memref<4x4xf32>, %arg1: index) -> memref<4x4xf32> {
  %res = "polygeist.subindex"(%arg0 , %arg1) : (memref<4x4xf32>, index) -> memref<4x4xf32>
  return %res : memref<4x4xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref_struct(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32)>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @subindexop_memref_struct(%arg0: memref<4x!llvm.struct<(f32)>>) -> memref<?xf32> {
  %c_0 = arith.constant 0 : index
  %res = "polygeist.subindex"(%arg0, %c_0) : (memref<4x!llvm.struct<(f32)>>, index) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref_nested_struct(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(struct<(f32)>)>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @subindexop_memref_nested_struct(%arg0: memref<4x!llvm.struct<(struct<(f32)>)>>) -> memref<?xf32> {
  %c_0 = arith.constant 0 : index
  %res = "polygeist.subindex"(%arg0, %c_0) : (memref<4x!llvm.struct<(struct<(f32)>)>>, index) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

// CHECK-LABEL: llvm.func @subindexop_memref_nested_ptr(
// CHECK-SAME:     %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:     %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:     %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:     %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr)>
// CHECK-NEXT:     llvm.return %[[VAL_3]] : !llvm.ptr
// CHECK-NEXT: }

func.func private @subindexop_memref_nested_ptr(%arg0: memref<4x!llvm.struct<(ptr)>>) -> memref<?x!llvm.ptr> {
  %c_0 = arith.constant 0 : index
  %res = "polygeist.subindex"(%arg0, %c_0) : (memref<4x!llvm.struct<(ptr)>>, index) -> memref<?x!llvm.ptr>
  return %res : memref<?x!llvm.ptr>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref_nested_struct_array(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 0, %[[VAL_2]], 0] : (!llvm.ptr, i64, i64) -> !llvm.ptr,  !llvm.struct<(array<4 x struct<(f32)>>)>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @subindexop_memref_nested_struct_array(%arg0: memref<4x!llvm.struct<(array<4x!llvm.struct<(f32)>>)>>) -> memref<?xf32> {
  %c_0 = arith.constant 0 : index
  %res = "polygeist.subindex"(%arg0, %c_0) : (memref<4x!llvm.struct<(array<4x!llvm.struct<(f32)>>)>>, index) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

// CHECK-LABEL:   llvm.func @memref2ptr(
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @memref2ptr(%arg0: memref<4xf32>) -> !llvm.ptr {
  %res = "polygeist.memref2pointer"(%arg0) : (memref<4xf32>) -> !llvm.ptr
  return %res : !llvm.ptr
}

// -----

// CHECK-LABEL:   llvm.func @ptr2memref(
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT:    }

func.func private @ptr2memref(%arg0: !llvm.ptr) -> memref<?xf32> {
  %res = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

#layout = affine_map<(s0) -> (s0 - 1)>

// CHECK-LABEL:   llvm.func private @non_bare_due_to_layout(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: i64) -> i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(-1 : index) : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]][%[[VAL_3]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_4]][%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr
// CHECK-NEXT:      llvm.return %[[VAL_6]] : i64
// CHECK-NEXT:    }

func.func private @non_bare_due_to_layout(%arg0: memref<100xi64, #layout>, %arg1: index) -> i64 attributes {llvm.linkage = #llvm.linkage<private>} {
  %res = memref.load %arg0[%arg1] : memref<100xi64, #layout>
  return %res : i64
}

// -----

// CHECK-LABEL:   llvm.func @view(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<3>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> !llvm.ptr<3> attributes {sym_visibility = "private"} {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.ptr<3>
// CHECK-NEXT:    }

func.func private @view(%arg0: memref<8xi8, 3>, %arg1: index) -> memref<2xf32, 3> {
  %res = memref.view %arg0[%arg1][] : memref<8xi8, 3> to memref<2xf32, 3>
  return %res : memref<2xf32, 3>
}

// -----

// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
func.func private @malloc(i64) -> !llvm.ptr

// CHECK-LABEL:   llvm.func @f0(
// CHECK-SAME:                  %[[VAL_0:.*]]: i64) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_0]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[VAL_4:.*]] = llvm.ptrtoint %[[VAL_3]] : !llvm.ptr to i64
// CHECK:           %[[VAL_5:.*]] = llvm.call @malloc(%[[VAL_4]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_5]] : !llvm.ptr
// CHECK:         }
func.func private @f0(%size: index) -> memref<?xi8> {
  %res = memref.alloc(%size) : memref<?xi8>
  return %res : memref<?xi8>
}

// CHECK-LABEL:   llvm.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: i64) -> !llvm.ptr
// CHECK:           %[[VAL_1:.*]] = llvm.call @malloc(%[[VAL_0]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK:         }
func.func private @f1(%size: i64) -> !llvm.ptr {
  %res = func.call @malloc(%size) : (i64) -> !llvm.ptr
  return %res : !llvm.ptr
}

// -----

// CHECK:         llvm.func @free(!llvm.ptr)
func.func private @free(!llvm.ptr)

// CHECK-LABEL:   llvm.func @f0(
// CHECK-SAME:                  %[[VAL_0:.*]]: !llvm.ptr)
// CHECK:           llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func private @f0(%ptr: memref<?xi8>) {
  memref.dealloc %ptr : memref<?xi8>
  return
}

// CHECK-LABEL:   llvm.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: !llvm.ptr)
// CHECK:           llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func private @f1(%ptr: !llvm.ptr) {
  func.call @free(%ptr) : (!llvm.ptr) -> ()
  return
}

// -----

// CHECK-LABEL:   llvm.func private @view(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<3>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64) -> !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_5]][0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_6]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_7]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_9]][3, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_10]][4, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_12]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.mul %[[VAL_11]], %[[VAL_3]]  : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_13]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      llvm.return %[[VAL_15]] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    }

func.func private @view(%arg0: memref<8xi8, 3>, %arg1: index, %arg2: index, %arg3: index) -> memref<?x?xf32, 3> attributes {llvm.linkage = #llvm.linkage<private>} {
  %res = memref.view %arg0[%arg1][%arg2, %arg3] : memref<8xi8, 3> to memref<?x?xf32, 3>
  return %res : memref<?x?xf32, 3>
}
