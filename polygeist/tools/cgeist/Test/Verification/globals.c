// RUN: cgeist %s --function=kernel_deriche -S | FileCheck %s
// RUN: cgeist %s --function=kernel_deriche -S -emit-llvm | FileCheck %s --check-prefix=LLVM

int local;
int local_init = 4;
static int internal;
static int internal_init = 5;
extern int external;

void run(int*, int*, int*, int*, int*);
void kernel_deriche() {
    run(&local, &local_init, &internal, &internal_init, &external);
}

// CHECK-DAG:   memref.global @external : memref<i32> {alignment = 4 : i64}
// CHECK-DAG:   memref.global "private" @internal_init : memref<i32> = dense<5> {alignment = 4 : i64}
// CHECK-DAG:   memref.global "private" @internal : memref<i32> = dense<0> {alignment = 4 : i64}
// CHECK-DAG:   memref.global @local_init : memref<i32> = dense<4> {alignment = 4 : i64}
// CHECK-DAG:   memref.global @local : memref<i32> = dense<0> {alignment = 4 : i64}

// CHECK-LABEL: func @kernel_deriche()
// CHECK-NEXT:     %0 = memref.get_global @local : memref<i32>
// CHECK-NEXT:     %alloca = memref.alloca() : memref<1xindex>
// CHECK-NEXT:     %reshape = memref.reshape %0(%alloca) : (memref<i32>, memref<1xindex>) -> memref<1xi32>
// CHECK-NEXT:     %cast = memref.cast %reshape : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %1 = memref.get_global @local_init : memref<i32>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<1xindex>
// CHECK-NEXT:     %reshape_1 = memref.reshape %1(%alloca_0) : (memref<i32>, memref<1xindex>) -> memref<1xi32>
// CHECK-NEXT:     %cast_2 = memref.cast %reshape_1 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %2 = memref.get_global @internal : memref<i32>
// CHECK-NEXT:     %alloca_3 = memref.alloca() : memref<1xindex>
// CHECK-NEXT:     %reshape_4 = memref.reshape %2(%alloca_3) : (memref<i32>, memref<1xindex>) -> memref<1xi32>
// CHECK-NEXT:     %cast_5 = memref.cast %reshape_4 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %3 = memref.get_global @internal_init : memref<i32>
// CHECK-NEXT:     %alloca_6 = memref.alloca() : memref<1xindex>
// CHECK-NEXT:     %reshape_7 = memref.reshape %3(%alloca_6) : (memref<i32>, memref<1xindex>) -> memref<1xi32>
// CHECK-NEXT:     %cast_8 = memref.cast %reshape_7 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %4 = memref.get_global @external : memref<i32>
// CHECK-NEXT:     %alloca_9 = memref.alloca() : memref<1xindex>
// CHECK-NEXT:     %reshape_10 = memref.reshape %4(%alloca_9) : (memref<i32>, memref<1xindex>) -> memref<1xi32>
// CHECK-NEXT:     %cast_11 = memref.cast %reshape_10 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     call @run(%cast, %cast_2, %cast_5, %cast_8, %cast_11) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// LLVM-DAG: @local = global i32 0, align 4
// LLVM-DAG: @local_init = global i32 4, align 4
// LLVM-DAG: @internal = private global i32 0, align 4
// LLVM-DAG: @internal_init = private global i32 5, align 4
// LLVM-DAG: @external = external global i32, align 4

// LLVM-LABEL: define void @kernel_deriche() {
// LLVM-NEXT:   call void @run(i32* @local, i32* @local_init, i32* @internal, i32* @internal_init, i32* @external)
// LLVM-NEXT:   ret void
// LLVM-LABEL: declare void @run(i32*, i32*, i32*, i32*, i32*)
