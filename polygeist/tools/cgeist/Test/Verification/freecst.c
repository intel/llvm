// RUN: cgeist %s --function=* -S | FileCheck %s

#include <stdlib.h>
    struct band {
        int dimX; 
    };
    struct dimensions {
        struct band LL;
    };
void writeNStage2DDWT(struct dimensions* bandDims) 
{
    free(bandDims);
}

// CHECK-LABEL:   func.func @writeNStage2DDWT(
// CHECK-SAME:                                %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
