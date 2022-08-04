// RUN: cgeist %s --function=kernel_deriche -S | FileCheck %s

int local;
int local_init = 4;
static int internal;
static int internal_init = 5;
extern int external;

void run(int*, int*, int*, int*, int*);
void kernel_deriche() {
    run(&local, &local_init, &internal, &internal_init, &external);
}

// CHECK-DAG:   memref.global @external : memref<1xi32>
// CHECK-DAG:   memref.global "private" @internal_init : memref<1xi32> = dense<5>
// CHECK-DAG:   memref.global "private" @internal : memref<1xi32> = uninitialized
// CHECK-DAG:   memref.global @local_init : memref<1xi32> = dense<4>
// CHECK-DAG:   memref.global @local : memref<1xi32> = uninitialized
// CHECK:   func @kernel_deriche()
// CHECK-NEXT:     %0 = memref.get_global @local : memref<1xi32>
// CHECK-NEXT:     %1 = memref.cast %0 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %2 = memref.get_global @local_init : memref<1xi32>
// CHECK-NEXT:     %3 = memref.cast %2 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %4 = memref.get_global @internal : memref<1xi32>
// CHECK-NEXT:     %5 = memref.cast %4 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %6 = memref.get_global @internal_init : memref<1xi32>
// CHECK-NEXT:     %7 = memref.cast %6 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %8 = memref.get_global @external : memref<1xi32>
// CHECK-NEXT:     %9 = memref.cast %8 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     call @run(%1, %3, %5, %7, %9) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
