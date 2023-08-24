// RUN: cgeist  %s --function=* -S | FileCheck %s

struct AOperandInfo {
  void* data;

  bool is_output;

  bool is_read_write;
};


/// This is all the non-templated stuff common to all SmallVectors.

/// This is the part of SmallVectorTemplateBase which does not depend on whether
/// the type T is a POD. The extra dummy template argument is used by ArrayRef
/// to avoid unnecessarily requiring T to be complete.
template <typename T>
class ASmallVectorTemplateCommon {
 public:
  void *BeginX, *EndX;

  // forward iterator creation methods.
  const T* begin() const {
    return (const T*)this->BeginX;
  }
};

unsigned long long int div_kernel_cuda(ASmallVectorTemplateCommon<AOperandInfo> &operands) {
  return (const AOperandInfo*)operands.EndX - operands.begin();
}


// CHECK-LABEL:   func.func @_Z15div_kernel_cudaR26ASmallVectorTemplateCommonI12AOperandInfoE(
// CHECK-SAME:                                                                                %[[VAL_0:.*]]: !llvm.ptr) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 16 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = call @_ZNK26ASmallVectorTemplateCommonI12AOperandInfoE5beginEv(%[[VAL_0]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.ptrtoint %[[VAL_3]] : !llvm.ptr to i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.ptrtoint %[[VAL_4]] : !llvm.ptr to i64
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.subi %[[VAL_5]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.divsi %[[VAL_7]], %[[VAL_1]] : i64
// CHECK-NEXT:      return %[[VAL_8]] : i64
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZNK26ASmallVectorTemplateCommonI12AOperandInfoE5beginEv(
// CHECK-SAME:                                                                        %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:      return %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:    }
