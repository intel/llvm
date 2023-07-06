// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>)>
!sycl_item_3_ = !sycl.item<[3, false], (!sycl.item_base<[3, false], (!sycl_range_3_, !sycl_id_3_)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_range_2_ = !sycl.nd_range<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL: func.func @id_default
func.func @id_default()
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor() : () -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor() : () -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor() : () -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}

// CHECK-LABEL: func.func @id_index
func.func @id_index(%arg0: index, %arg1: index, %arg2: index)
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor(%arg0)
      : (index) -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor(%arg0, %arg1)
      : (index, index) -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor(%arg0, %arg1, %arg2)
      : (index, index, index) -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}

// CHECK-LABEL: func.func @id_range
func.func @id_range(%arg0: memref<?x!sycl_range_1_>,
                    %arg1: memref<?x!sycl_range_2_>,
                    %arg2: memref<?x!sycl_range_3_>)
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor(%arg0)
      : (memref<?x!sycl_range_1_>) -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor(%arg1)
      : (memref<?x!sycl_range_2_>) -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor(%arg2)
      : (memref<?x!sycl_range_3_>) -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}

// CHECK-LABEL: func.func @id_item
func.func @id_item(%arg0: memref<?x!sycl_item_1_>,
                   %arg1: memref<?x!sycl_item_2_>,
                   %arg2: memref<?x!sycl_item_3_>)
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor(%arg0)
      : (memref<?x!sycl_item_1_>) -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor(%arg1)
      : (memref<?x!sycl_item_2_>) -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor(%arg2)
      : (memref<?x!sycl_item_3_>) -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}

// CHECK-LABEL: func.func @id_id
func.func @id_id(%arg0: memref<?x!sycl_id_1_>,
                 %arg1: memref<?x!sycl_id_2_>,
                 %arg2: memref<?x!sycl_id_3_>)
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor(%arg0)
      : (memref<?x!sycl_id_1_>) -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor(%arg1)
      : (memref<?x!sycl_id_2_>) -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor(%arg2)
      : (memref<?x!sycl_id_3_>) -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}

// CHECK-LABEL: func.func @range_index
func.func @range_index(%arg0: index, %arg1: index, %arg2: index)
    -> (memref<1x!sycl_range_1_>,
        memref<1x!sycl_range_2_>,
        memref<1x!sycl_range_3_>) {
  %range1 = sycl.range.constructor(%arg0)
      : (index) -> memref<1x!sycl_range_1_>
  %range2 = sycl.range.constructor(%arg0, %arg1)
      : (index, index) -> memref<1x!sycl_range_2_>
  %range3 = sycl.range.constructor(%arg0, %arg1, %arg2)
      : (index, index, index) -> memref<1x!sycl_range_3_>
  func.return %range1, %range2, %range3
      : memref<1x!sycl_range_1_>,
        memref<1x!sycl_range_2_>,
        memref<1x!sycl_range_3_>
}

// CHECK-LABEL: func.func @range_range
func.func @range_range(%arg0: memref<?x!sycl_range_1_>,
                       %arg1: memref<?x!sycl_range_2_>,
                       %arg2: memref<?x!sycl_range_3_>)
    -> (memref<1x!sycl_range_1_>,
        memref<1x!sycl_range_2_>,
        memref<1x!sycl_range_3_>) {
  %range1 = sycl.range.constructor(%arg0)
      : (memref<?x!sycl_range_1_>) -> memref<1x!sycl_range_1_>
  %range2 = sycl.range.constructor(%arg1)
      : (memref<?x!sycl_range_2_>) -> memref<1x!sycl_range_2_>
  %range3 = sycl.range.constructor(%arg2)
      : (memref<?x!sycl_range_3_>) -> memref<1x!sycl_range_3_>
  func.return %range1, %range2, %range3
      : memref<1x!sycl_range_1_>,
        memref<1x!sycl_range_2_>,
        memref<1x!sycl_range_3_>
}

// CHECK-LABEL: func.func @nd_range_with_offset
func.func @nd_range_with_offset(%arg0: memref<?x!sycl_range_1_>,
                                %arg1: memref<?x!sycl_range_1_>,
                                %arg2: memref<?x!sycl_id_1_>,
                                %arg3: memref<?x!sycl_range_2_>,
                                %arg4: memref<?x!sycl_range_2_>,
                                %arg5: memref<?x!sycl_id_2_>,
                                %arg6: memref<?x!sycl_range_3_>,
                                %arg7: memref<?x!sycl_range_3_>,
                                %arg8: memref<?x!sycl_id_3_>)
    -> (memref<1x!sycl_nd_range_1_>,
        memref<1x!sycl_nd_range_2_>,
	memref<1x!sycl_nd_range_3_>) {
  %nd1 = sycl.nd_range.constructor(%arg0, %arg1, %arg2)
      : (memref<?x!sycl_range_1_>,
         memref<?x!sycl_range_1_>,
	 memref<?x!sycl_id_1_>) -> memref<1x!sycl_nd_range_1_>
  %nd2 = sycl.nd_range.constructor(%arg3, %arg4, %arg5)
      : (memref<?x!sycl_range_2_>,
         memref<?x!sycl_range_2_>,
	 memref<?x!sycl_id_2_>) -> memref<1x!sycl_nd_range_2_>
  %nd3 = sycl.nd_range.constructor(%arg6, %arg7, %arg8)
      : (memref<?x!sycl_range_3_>,
         memref<?x!sycl_range_3_>,
	 memref<?x!sycl_id_3_>) -> memref<1x!sycl_nd_range_3_>
  func.return %nd1, %nd2, %nd3
      : memref<1x!sycl_nd_range_1_>,
        memref<1x!sycl_nd_range_2_>,
	memref<1x!sycl_nd_range_3_>
}

// CHECK-LABEL: func.func @nd_range_with_no_offset
func.func @nd_range_with_no_offset(%arg0: memref<?x!sycl_range_1_>,
                                   %arg1: memref<?x!sycl_range_1_>,
                                   %arg2: memref<?x!sycl_range_2_>,
                                   %arg3: memref<?x!sycl_range_2_>,
                                   %arg4: memref<?x!sycl_range_3_>,
                                   %arg5: memref<?x!sycl_range_3_>)
    -> (memref<1x!sycl_nd_range_1_>,
        memref<1x!sycl_nd_range_2_>,
	memref<1x!sycl_nd_range_3_>) {
  %nd1 = sycl.nd_range.constructor(%arg0, %arg1)
      : (memref<?x!sycl_range_1_>,
         memref<?x!sycl_range_1_>) -> memref<1x!sycl_nd_range_1_>
  %nd2 = sycl.nd_range.constructor(%arg2, %arg3)
      : (memref<?x!sycl_range_2_>,
         memref<?x!sycl_range_2_>) -> memref<1x!sycl_nd_range_2_>
  %nd3 = sycl.nd_range.constructor(%arg4, %arg5)
      : (memref<?x!sycl_range_3_>,
         memref<?x!sycl_range_3_>) -> memref<1x!sycl_nd_range_3_>
  func.return %nd1, %nd2, %nd3
      : memref<1x!sycl_nd_range_1_>,
        memref<1x!sycl_nd_range_2_>,
	memref<1x!sycl_nd_range_3_>
}

// CHECK-LABEL: func.func @nd_range_nd_range
func.func @nd_range_nd_range(%arg0: memref<?x!sycl_nd_range_1_>,
                             %arg1: memref<?x!sycl_nd_range_2_>,
                             %arg2: memref<?x!sycl_nd_range_3_>)
    -> (memref<1x!sycl_nd_range_1_>,
        memref<1x!sycl_nd_range_2_>,
	memref<1x!sycl_nd_range_3_>) {
  %nd1 = sycl.nd_range.constructor(%arg0)
      : (memref<?x!sycl_nd_range_1_>) -> memref<1x!sycl_nd_range_1_>
  %nd2 = sycl.nd_range.constructor(%arg1)
      : (memref<?x!sycl_nd_range_2_>) -> memref<1x!sycl_nd_range_2_>
  %nd3 = sycl.nd_range.constructor(%arg2)
      : (memref<?x!sycl_nd_range_3_>) -> memref<1x!sycl_nd_range_3_>
  func.return %nd1, %nd2, %nd3
      : memref<1x!sycl_nd_range_1_>,
        memref<1x!sycl_nd_range_2_>,
	memref<1x!sycl_nd_range_3_>
}
