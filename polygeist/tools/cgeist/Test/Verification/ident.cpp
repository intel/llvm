// RUN: cgeist  %s -O0 -w --function=* -S | FileCheck %s

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

// CHECK-LABEL:   func.func @lt_kernel_cuda(
// CHECK-SAME:                              %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : i8
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(ptr)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(ptr)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = call @_ZNK15MTensorIterator11input_dtypeEv(%[[VAL_0]]) : (!llvm.ptr) -> i8
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_1]] : i8
// CHECK-NEXT:      scf.if %[[VAL_6]] {
// CHECK-NEXT:        %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr)>
// CHECK-NEXT:        llvm.store %[[VAL_0]], %[[VAL_7]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:        %[[VAL_8:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> !llvm.struct<(ptr)>
// CHECK-NEXT:        llvm.store %[[VAL_8]], %[[VAL_3]] : !llvm.struct<(ptr)>, !llvm.ptr
// CHECK-NEXT:        func.call @_ZZ14lt_kernel_cudaENK3$_0clEv(%[[VAL_3]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZNK15MTensorIterator11input_dtypeEv(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: !llvm.ptr) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(ptr)>)>
// CHECK-NEXT:      %[[VAL_3:.*]] = call @_ZNK12MSmallVectorI12MOperandInfoEixEi(%[[VAL_2]], %[[VAL_1]]) : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_3]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, i8)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i8
// CHECK-NEXT:      return %[[VAL_5]] : i8
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func private @_ZZ14lt_kernel_cudaENK3$_0clEv(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i8)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i8)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> !llvm.struct<(i8)>
// CHECK-NEXT:      llvm.store %[[VAL_6]], %[[VAL_2]] : !llvm.struct<(i8)>, !llvm.ptr
// CHECK-NEXT:      call @_Z11igpu_kernelIZZ14lt_kernel_cudaENK3$_0clEvEUlvE_EvR15MTensorIteratorRKT_(%[[VAL_5]], %[[VAL_2]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZNK12MSmallVectorI12MOperandInfoEixEi(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[VAL_1:.*]]: i32) -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK-NEXT:      %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : index to i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_5]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i8, i8)>
// CHECK-NEXT:      return %[[VAL_6]] : !llvm.ptr
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func private @_Z11igpu_kernelIZZ14lt_kernel_cudaENK3$_0clEvEUlvE_EvR15MTensorIteratorRKT_(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:         %[[VAL_1:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = call @_ZNK15MTensorIterator6deviceEv(%[[VAL_0]]) : (!llvm.ptr) -> i8
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZNK15MTensorIterator6deviceEv(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(ptr)>)>
// CHECK-NEXT:      %[[VAL_3:.*]] = call @_ZNK12MSmallVectorI12MOperandInfoEixEi(%[[VAL_2]], %[[VAL_1]]) : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_3]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, i8)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i8
// CHECK-NEXT:      return %[[VAL_5]] : i8
// CHECK-NEXT:    }
