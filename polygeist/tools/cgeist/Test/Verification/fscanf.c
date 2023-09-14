// RUN: cgeist %s -O2 %stdinclude --function=alloc -S --raise-scf-to-affine=false | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

int* alloc() {
    int no_of_nodes;

	scanf("%d",&no_of_nodes);
   
	// allocate host memory
	int* h_graph_nodes = (int*) malloc(sizeof(int)*no_of_nodes);

	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		scanf("%d\n", &h_graph_nodes[i]);
    }
	return h_graph_nodes;
}

// CHECK-LABEL:   llvm.mlir.global internal constant @str1("
// CHECK-SAME:                                              %d\0A\00") {addr_space = 0 : i32}
// CHECK-NEXT:    llvm.mlir.global internal constant @str0("%d\00") {addr_space = 0 : i32}
// CHECK-NEXT:    llvm.func @__isoc99_scanf(!llvm.ptr, ...) -> i32 attributes {sym_visibility = "private"}

// CHECK-LABEL:   func.func @alloc() -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 4 : index
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_4]] x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_6]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.call @__isoc99_scanf(%[[VAL_7]], %[[VAL_5]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i32
// CHECK-NEXT:      %[[VAL_10:.*]] = arith.extsi %[[VAL_9]] : i32 to i64
// CHECK-NEXT:      %[[VAL_11:.*]] = arith.muli %[[VAL_10]], %[[VAL_3]] : i64
// CHECK-NEXT:      %[[VAL_12:.*]] = arith.index_cast %[[VAL_11]] : i64 to index
// CHECK-NEXT:      %[[VAL_13:.*]] = arith.divui %[[VAL_12]], %[[VAL_2]] : index
// CHECK-NEXT:      %[[VAL_14:.*]] = memref.alloc(%[[VAL_13]]) : memref<?xi32>
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.mlir.addressof @str1 : !llvm.ptr
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.getelementptr inbounds %[[VAL_15]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
// CHECK-NEXT:      %[[VAL_17:.*]] = arith.index_cast %[[VAL_9]] : i32 to index
// CHECK-NEXT:      scf.for %[[VAL_18:.*]] = %[[VAL_0]] to %[[VAL_17]] step %[[VAL_1]] {
// CHECK-NEXT:        %[[VAL_19:.*]] = llvm.call @__isoc99_scanf(%[[VAL_16]], %[[VAL_5]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:        %[[VAL_20:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i32
// CHECK-NEXT:        memref.store %[[VAL_20]], %[[VAL_14]]{{\[}}%[[VAL_18]]] : memref<?xi32>
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[VAL_14]] : memref<?xi32>
// CHECK-NEXT:    }
