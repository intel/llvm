// RUN: cgeist -O0 %s -w -o - -S --function=* | FileCheck %s

// COM: Test bswap builtin lowering

// CHECK:   func.func @_Z5bswap
void bswap(unsigned short* s, unsigned int* i, unsigned long *l) {
  // CHECK: llvm.intr.bswap({{.*}}) : (i16) -> i16
  *s = __builtin_bswap16(*s);
  // CHECK: llvm.intr.bswap({{.*}}) : (i32) -> i32
  *i = __builtin_bswap32(*i);
  // CHECK: llvm.intr.bswap({{.*}}) : (i64) -> i64
  *l = __builtin_bswap64(*l);
}
