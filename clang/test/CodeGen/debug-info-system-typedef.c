// Ensure that debug info for typedefs in system headers is still generated
// even if -fno-system-debug is used.  This is justified because debug size 
// savings is small, but debugging is commonly done with types that are
// typedef-ed in system headers.  Thus, the increased debuggability
// is worth the small extra cost.

// RUN: %clang -fno-system-debug -emit-llvm -S -g %s -o %t.ll

// RUN: FileCheck %s < %t.ll

// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast8_t",
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast16_t",
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast32_t",
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "int_fast64_t",

// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast8_t",
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast16_t",
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast32_t",
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "uint_fast64_t",
  
#include <stdint.h>

int_fast8_t   f8;
int_fast16_t  f16;
int_fast32_t  f32;
int_fast64_t  f64;

uint_fast8_t   uf8;
uint_fast16_t  uf16;
uint_fast32_t  uf32;
uint_fast64_t  uf64;

int
main (void)
{
  return 0;
}
