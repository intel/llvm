// RUN: cgeist %s -O2 --function=* -S | FileCheck %s

struct MOperandInfo {
  char device;
  char dtype;
};

template <typename T>
class MSmallVector {
 public:
  
  struct MOperandInfo *BeginX;
  
  const struct MOperandInfo& operator[](int idx) const {
    return BeginX[idx];
  }
};


struct MTensorIterator {
  char input_dtype() const { return operands_[0].dtype; }
  char device() const { return operands_[0].device; }
  MSmallVector<MOperandInfo> operands_;
};

template <typename func_t>
void igpu_kernel(MTensorIterator& iter, const func_t& f) {
  iter.device();
}

extern "C" {
void lt_kernel_cuda(MTensorIterator& iter) {
    if (iter.input_dtype()) 
	{                                                                                              
		([&]() {
      igpu_kernel(iter, []() -> bool {
        return false;
      });
	  })();
    }              
}
}

// CHECK:   func @lt_kernel_cuda(%arg0: !llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c0_i8 = arith.constant 0 : i8
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)> : (i64) -> !llvm.ptr<!llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)>>
// CHECK-NEXT:     %1 = llvm.alloca %c1_i64 x !llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)> : (i64) -> !llvm.ptr<!llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)>>
// CHECK-NEXT:     %2 = call @_ZNK15MTensorIterator11input_dtypeEv(%arg0) : (!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8
// CHECK-NEXT:     %3 = arith.cmpi ne, %2, %c0_i8 : i8
// CHECK-NEXT:     scf.if %3 {
// CHECK-NEXT:       %4 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<!llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)>>) -> !llvm.ptr<!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK-NEXT:       llvm.store %arg0, %4 : !llvm.ptr<!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK-NEXT:       %5 = llvm.load %1 : !llvm.ptr<!llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)>>
// CHECK-NEXT:       llvm.store %5, %0 : !llvm.ptr<!llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)>>
// CHECK-NEXT:       call @_ZZ14lt_kernel_cudaENK3$_0clEv(%0) : (!llvm.ptr<!llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)>>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZNK15MTensorIterator11input_dtypeEv(%arg0: !llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> memref<?x1xmemref<?x2xi8>>
// CHECK-NEXT:     %1 = call @_ZNK12MSmallVectorI12MOperandInfoEixEi(%0, %c0_i32) : (memref<?x1xmemref<?x2xi8>>, i32) -> memref<?x2xi8>
// CHECK-NEXT:     %2 = affine.load %1[0, 1] : memref<?x2xi8>
// CHECK-NEXT:     return %2 : i8
// CHECK-NEXT:   }
// CHECK:   func private @_ZZ14lt_kernel_cudaENK3$_0clEv(%arg0: !llvm.ptr<!llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<!llvm.struct<(!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>)>>) -> !llvm.ptr<!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>>
// CHECK-NEXT:     %2 = call @_ZNK15MTensorIterator6deviceEv(%1) : (!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZNK12MSmallVectorI12MOperandInfoEixEi(%arg0: memref<?x1xmemref<?x2xi8>>, %arg1: i32) -> memref<?x2xi8> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = affine.load %arg0[0, 0] : memref<?x1xmemref<?x2xi8>>
// CHECK-NEXT:     %1 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %2 = "polygeist.subindex"(%0, %1) : (memref<?x2xi8>, index) -> memref<?x2xi8>
// CHECK-NEXT:     return %2 : memref<?x2xi8>
// CHECK-NEXT:   }
// CHECK-NEXT:   func @_ZNK15MTensorIterator6deviceEv(%arg0: !llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr<!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> memref<?x1xmemref<?x2xi8>>
// CHECK-NEXT:     %1 = call @_ZNK12MSmallVectorI12MOperandInfoEixEi(%0, %c0_i32) : (memref<?x1xmemref<?x2xi8>>, i32) -> memref<?x2xi8>
// CHECK-NEXT:     %2 = affine.load %1[0, 0] : memref<?x2xi8>
// CHECK-NEXT:     return %2 : i8
// CHECK-NEXT:   }

