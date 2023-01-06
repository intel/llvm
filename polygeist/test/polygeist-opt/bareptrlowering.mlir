// RUN: polygeist-opt %s --convert-polygeist-to-llvm="use-bare-ptr-memref-call-conv" --split-input-file | FileCheck %s

// CHECK-LABEL:   llvm.func @ptr_ret_static(i64) -> !llvm.ptr<i64>

func.func private @ptr_ret_static(%arg0: i64) -> memref<4xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_ret_dynamic(i64) -> !llvm.ptr<i64>

func.func private @ptr_ret_dynamic(%arg0: i64) -> memref<?xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_ret_nd_static(i64) -> !llvm.ptr<i64>

func.func private @ptr_ret_nd_static(%arg0: i64) -> memref<4x4xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_ret_nd_dynamic(i64) -> !llvm.ptr<i64>

func.func private @ptr_ret_nd_dynamic(%arg0: i64) -> memref<?x4x4xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_args_and_ret(!llvm.ptr<i64>, !llvm.ptr<i64>) -> !llvm.ptr<i64>

func.func private @ptr_args_and_ret(%arg0: memref<1xi64>, %arg1: memref<?xi64>) -> memref<?x4x4xi64>

// -----

// CHECK-LABEL:   llvm.func @ptr_args_and_ret_with_attrs(!llvm.ptr<i64> {llvm.byval = !llvm.ptr<i64>}, !llvm.ptr<i64> {llvm.byval = !llvm.ptr<i64>}) -> !llvm.ptr<i64>

func.func private @ptr_args_and_ret_with_attrs(%arg0: memref<1xi64> {llvm.byval = memref<1xi64>},
                                               %arg1: memref<?xi64> {llvm.byval = memref<?xi64>}) -> memref<?x4x4xi64>

// -----

gpu.module @kernels {

// CHECK-LABEL:   llvm.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<i64> {llvm.byval = !llvm.ptr<i64>},
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr<i64> {llvm.byval = !llvm.ptr<i64>}) attributes {gpu.kernel, workgroup_attributions = 0 : i64} {
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:         }

  gpu.func @kernel(%arg0: memref<1xi64> {llvm.byval = memref<1xi64>},
                   %arg1: memref<?xi64> {llvm.byval = memref<?xi64>}) kernel {
    gpu.return
  }
}

// -----

// CHECK-LABEL:   llvm.mlir.global external @global() {addr_space = 0 : i32} : !llvm.array<3 x i64>

memref.global @global : memref<3xi64>

// CHECK-LABEL:   llvm.func @get_global() -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.addressof @global : !llvm.ptr<array<3 x i64>>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][0, 0] : (!llvm.ptr<array<3 x i64>>) -> !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:    }

func.func private @get_global() -> memref<3xi64> {
  %0 = memref.get_global @global : memref<3xi64>
  return %0 : memref<3xi64>
}

// -----

// CHECK-LABEL:   llvm.mlir.global external @global_addrspace() {addr_space = 4 : i32} : !llvm.array<3 x i64>

memref.global @global_addrspace : memref<3xi64, 4>

// CHECK-LABEL:   llvm.func @get_global_addrspace() -> !llvm.ptr<i64, 4>
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.addressof @global_addrspace : !llvm.ptr<array<3 x i64>, 4>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][0, 0] : (!llvm.ptr<array<3 x i64>, 4>) -> !llvm.ptr<i64, 4>
// CHECK-NEXT:      llvm.return %[[VAL_1]] : !llvm.ptr<i64, 4>
// CHECK-NEXT:    }

func.func private @get_global_addrspace() -> memref<3xi64, 4> {
  %0 = memref.get_global @global_addrspace : memref<3xi64, 4>
  return %0 : memref<3xi64, 4>
}

// -----

memref.global "private" constant @shape : memref<2xi64> = dense<[2, 2]>

// CHECK-LABEL:   llvm.func @reshape(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr<i32>) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr %{{.*}}[0, 0] : (!llvm.ptr<array<2 x i64>>) -> !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_0]] : !llvm.ptr<i32>
// CHECK-NEXT:    }

func.func private @reshape(%arg0: memref<4xi32>) -> memref<2x2xi32> {
  %shape = memref.get_global @shape : memref<2xi64>
  %0 = memref.reshape %arg0(%shape) : (memref<4xi32>, memref<2xi64>) -> memref<2x2xi32>
  return %0 : memref<2x2xi32>
}

// -----

memref.global "private" constant @shape : memref<1xindex>

// CHECK-LABEL:   llvm.func @reshape_dyn(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm.ptr<i32>) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr %{{.*}}[0, 0] : (!llvm.ptr<array<1 x i64>>) -> !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_0]] : !llvm.ptr<i32>
// CHECK-NEXT:    }

func.func private @reshape_dyn(%arg0: memref<4xi32>) -> memref<?xi32> {
  %shape = memref.get_global @shape : memref<1xindex>
  %0 = memref.reshape %arg0(%shape) : (memref<4xi32>, memref<1xindex>) -> memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// CHECK-LABEL:   llvm.func @alloca()
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.null : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.ptrtoint %[[VAL_2]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x i32 : (i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @alloca() {
  %0 = memref.alloca() : memref<2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @alloca_nd()
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.null : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(60 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.ptrtoint %[[VAL_2]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x i32 : (i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @alloca_nd() {
  %0 = memref.alloca() : memref<3x10x2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @alloca_aligned()
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.null : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.ptrtoint %[[VAL_2]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @alloca_aligned() {
  %0 = memref.alloca() {alignment = 8} : memref<2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @alloca_nd_aligned()
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.null : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(60 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.ptrtoint %[[VAL_2]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @alloca_nd_aligned() {
  %0 = memref.alloca() {alignment = 8} : memref<3x10x2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @malloc(i64) -> !llvm.ptr<i8>

// CHECK-LABEL:   llvm.func @alloc()
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.null : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_0]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.ptrtoint %[[VAL_3]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.call @malloc(%[[VAL_4]]) : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.bitcast %[[VAL_5]] : !llvm.ptr<i8> to !llvm.ptr<i32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @alloc() {
  %0 = memref.alloc() : memref<2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @alloc_nd()
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(20 : index) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mlir.constant(60 : index) : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.mlir.null : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_5]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.ptrtoint %[[VAL_7]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.call @malloc(%[[VAL_8]]) : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.bitcast %[[VAL_9]] : !llvm.ptr<i8> to !llvm.ptr<i32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @alloc_nd() {
  %0 = memref.alloc() : memref<3x10x2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @alloc_aligned()
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.null : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_0]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.ptrtoint %[[VAL_3]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.add %[[VAL_4]], %[[VAL_5]]  : i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.call @malloc(%[[VAL_6]]) : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.bitcast %[[VAL_7]] : !llvm.ptr<i8> to !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.ptrtoint %[[VAL_8]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.sub %[[VAL_5]], %[[VAL_10]]  : i64
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.add %[[VAL_9]], %[[VAL_11]]  : i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.urem %[[VAL_12]], %[[VAL_5]]  : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.sub %[[VAL_12]], %[[VAL_13]]  : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.inttoptr %[[VAL_14]] : i64 to !llvm.ptr<i32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @alloc_aligned() {
  %0 = memref.alloc() {alignment = 8} : memref<2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @alloc_nd_aligned()
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(20 : index) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mlir.constant(60 : index) : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.mlir.null : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_5]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.ptrtoint %[[VAL_7]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.add %[[VAL_8]], %[[VAL_9]]  : i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.call @malloc(%[[VAL_10]]) : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.bitcast %[[VAL_11]] : !llvm.ptr<i8> to !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.ptrtoint %[[VAL_12]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.sub %[[VAL_9]], %[[VAL_14]]  : i64
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.add %[[VAL_13]], %[[VAL_15]]  : i64
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.urem %[[VAL_16]], %[[VAL_9]]  : i64
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.sub %[[VAL_16]], %[[VAL_17]]  : i64
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.inttoptr %[[VAL_18]] : i64 to !llvm.ptr<i32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @alloc_nd_aligned() {
  %0 = memref.alloc() {alignment = 8} : memref<3x10x2xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @dealloc(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr<i32>)
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.bitcast %[[VAL_0]] : !llvm.ptr<i32> to !llvm.ptr<i8>
// CHECK-NEXT:      llvm.call @free(%[[VAL_1]]) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @dealloc(%arg0: memref<?xi32>) {
  memref.dealloc %arg0 : memref<?xi32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @cast(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<i32>) -> !llvm.ptr<i32>
// CHECK-NEXT:      llvm.return %[[VAL_0]] : !llvm.ptr<i32>
// CHECK-NEXT:    }

func.func private @cast(%arg0: memref<2xi32>) -> memref<?xi32> {
  %0 = memref.cast %arg0 : memref<2xi32> to memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// CHECK-LABEL:   llvm.func @load(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<f32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> f32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : f32
// CHECK-NEXT:    }

func.func private @load(%arg0: memref<100xf32>, %index: index) -> f32 {
  %0 = memref.load %arg0[%index] : memref<100xf32>
  return %0 : f32
}

// -----

// CHECK-LABEL:   llvm.func @load_nd(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr<f32>,
// CHECK-SAME:                       %[[VAL_1:.*]]: i64,
// CHECK-SAME:                       %[[VAL_2:.*]]: i64) -> f32
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.add %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_5]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_7]] : f32
// CHECK-NEXT:    }

func.func private @load_nd(%arg0: memref<100x100xf32>, %index0: index, %index1: index) -> f32 {
  %0 = memref.load %arg0[%index0, %index1] : memref<100x100xf32>
  return %0 : f32
}

// -----

// CHECK-LABEL:   llvm.func @load_nd_dyn(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm.ptr<f32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: i64,
// CHECK-SAME:                           %[[VAL_2:.*]]: i64) -> f32
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.add %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_5]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_7]] : f32
// CHECK-NEXT:    }

func.func private @load_nd_dyn(%arg0: memref<?x100xf32>, %index0: index, %index1: index) -> f32 {
  %0 = memref.load %arg0[%index0, %index1] : memref<?x100xf32>
  return %0 : f32
}

// -----

// CHECK-LABEL:   llvm.func @store(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<f32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: i64,
// CHECK-SAME:                     %[[VAL_2:.*]]: f32)
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.store %[[VAL_2]], %[[VAL_3]] : !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @store(%arg0: memref<100xf32>, %index: index, %val: f32) {
  memref.store %val, %arg0[%index] : memref<100xf32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @store_nd(
// CHECK-SAME:                        %[[VAL_0:.*]]: !llvm.ptr<f32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64,
// CHECK-SAME:                        %[[VAL_3:.*]]: f32)
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_1]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.add %[[VAL_5]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_6]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.store %[[VAL_3]], %[[VAL_7]] : !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @store_nd(%arg0: memref<100x100xf32>, %index0: index, %index1: index, %val: f32) {
  memref.store %val, %arg0[%index0, %index1] : memref<100x100xf32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @store_nd_dyn(
// CHECK-SAME:                            %[[VAL_0:.*]]: !llvm.ptr<f32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64,
// CHECK-SAME:                            %[[VAL_3:.*]]: f32)
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_1]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.add %[[VAL_5]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_6]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.store %[[VAL_3]], %[[VAL_7]] : !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

func.func private @store_nd_dyn(%arg0: memref<?x100xf32>, %index0: index, %index1: index, %val: f32) {
  memref.store %val, %arg0[%index0, %index1] : memref<?x100xf32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @impl(!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>

func.func private @impl(%arg0: memref<?xf32>, %arg1: index) -> memref<?xf32>

// CHECK-LABEL:   llvm.func @call(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<f32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.call @impl(%[[VAL_0]], %[[VAL_1]]) : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.ptr<f32>
// CHECK-NEXT:    }

func.func private @call(%arg0: memref<?xf32>, %arg1: index) -> memref<?xf32> {
  %res = func.call @impl(%arg0, %arg1) : (memref<?xf32>, index) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !llvm.ptr<f32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mul %[[VAL_1]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_3]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_4]] : !llvm.ptr<f32>
// CHECK-NEXT:    }

func.func private @subindexop_memref(%arg0: memref<4x4xf32>, %arg1: index) -> memref<4xf32> {
  %res = "polygeist.subindex"(%arg0 , %arg1) : (memref<4x4xf32>, index) -> memref<4xf32>
  return %res : memref<4xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref_same_dim(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr<f32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.ptr<f32>
// CHECK-NEXT:    }

func.func private @subindexop_memref_same_dim(%arg0: memref<4x4xf32>, %arg1: index) -> memref<4x4xf32> {
  %res = "polygeist.subindex"(%arg0 , %arg1) : (memref<4x4xf32>, index) -> memref<4x4xf32>
  return %res : memref<4x4xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref_struct(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !llvm.ptr<struct<(f32)>>) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 0] : (!llvm.ptr<struct<(f32)>>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : !llvm.ptr<f32>
// CHECK-NEXT:    }

func.func private @subindexop_memref_struct(%arg0: memref<4x!llvm.struct<(f32)>>) -> memref<?xf32> {
  %c_0 = arith.constant 0 : index
  %res = "polygeist.subindex"(%arg0, %c_0) : (memref<4x!llvm.struct<(f32)>>, index) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref_nested_struct(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !llvm.ptr<struct<(struct<(f32)>)>>) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 0, 0] : (!llvm.ptr<struct<(struct<(f32)>)>>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : !llvm.ptr<f32>
// CHECK-NEXT:    }

func.func private @subindexop_memref_nested_struct(%arg0: memref<4x!llvm.struct<(struct<(f32)>)>>) -> memref<?xf32> {
  %c_0 = arith.constant 0 : index
  %res = "polygeist.subindex"(%arg0, %c_0) : (memref<4x!llvm.struct<(struct<(f32)>)>>, index) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref_nested_struct_ptr(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !llvm.ptr<struct<(ptr<struct<(f32)>>)>>) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 0, %[[VAL_2]], %[[VAL_1]]] : (!llvm.ptr<struct<(ptr<struct<(f32)>>)>>, i64, i64, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : !llvm.ptr<f32>
// CHECK-NEXT:    }

func.func private @subindexop_memref_nested_struct_ptr(%arg0: memref<4x!llvm.struct<(ptr<struct<(f32)>>)>>) -> memref<?xf32> {
  %c_0 = arith.constant 0 : index
  %res = "polygeist.subindex"(%arg0, %c_0) : (memref<4x!llvm.struct<(ptr<struct<(f32)>>)>>, index) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

// CHECK-LABEL:   llvm.func @subindexop_memref_nested_struct_array(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: !llvm.ptr<struct<(array<4 x struct<(f32)>>)>>) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 0, %[[VAL_2]], 0] : (!llvm.ptr<struct<(array<4 x struct<(f32)>>)>>, i64, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : !llvm.ptr<f32>
// CHECK-NEXT:    }

func.func private @subindexop_memref_nested_struct_array(%arg0: memref<4x!llvm.struct<(array<4x!llvm.struct<(f32)>>)>>) -> memref<?xf32> {
  %c_0 = arith.constant 0 : index
  %res = "polygeist.subindex"(%arg0, %c_0) : (memref<4x!llvm.struct<(array<4x!llvm.struct<(f32)>>)>>, index) -> memref<?xf32>
  return %res : memref<?xf32>
}

// -----

// CHECK-LABEL:   llvm.func @memref2ptr(
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr<f32>) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.bitcast %[[VAL_0]] : !llvm.ptr<f32> to !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_1]] : !llvm.ptr<f32>
// CHECK-NEXT:    }

func.func private @memref2ptr(%arg0: memref<4xf32>) -> !llvm.ptr<f32> {
  %res = "polygeist.memref2pointer"(%arg0) : (memref<4xf32>) -> !llvm.ptr<f32>
  return %res : !llvm.ptr<f32>
}

// -----

// CHECK-LABEL:   llvm.func @ptr2memref(
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr<f32>) -> !llvm.ptr<f32>
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.bitcast %[[VAL_0]] : !llvm.ptr<f32> to !llvm.ptr<f32>
// CHECK-NEXT:      llvm.return %[[VAL_1]] : !llvm.ptr<f32>
// CHECK-NEXT:    }

func.func private @ptr2memref(%arg0: !llvm.ptr<f32>) -> memref<?xf32> {
  %res = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr<f32>) -> memref<?xf32>
  return %res : memref<?xf32>
}
