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

// CHECK:   func @writeNStage2DDWT(%arg0: !llvm.ptr<struct<(struct<(i32)>)>>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[a1:.+]] = llvm.bitcast %arg0 : !llvm.ptr<struct<(struct<(i32)>)>> to !llvm.ptr<i8>
// CHECK-NEXT:     llvm.call @free(%[[a1]]) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
