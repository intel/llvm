// RUN: %clang_cc1 -disable-llvm-passes %s -emit-llvm -o - | sycl-post-link -split=auto -ir-output-only -o %t.out_ir_only.bc
// RUN: llvm-dis %t.out_ir_only.bc -o  %t.out_ir_only.ll
// RUN: FileCheck %s --input-file %t.out_ir_only.ll
//
// RUN: %clang_cc1 -disable-llvm-passes %s -emit-llvm -o %t.out.bc
// RUN: sycl-post-link -split=auto -symbols -split-esimd -lower-esimd -O2 -spec-const=default %t.out.bc -o %t.out.table
// RUN: llvm-dis "$(sed -n '2p' %t.out.table | sed 's/|.*//')" -o %t.out.ll
// RUN: FileCheck %s --input-file %t.out.ll
// CHECK-NOT: @llvm.used

struct a;
class g {
public:
  int c;
  ~g();
};
template <class>
class h {
public:
  static const void k();
  static g i;
};
template <class j>
const void h<j>::k() { i.c = 0; }
template <class j>
g h<j>::i;
template <class>
struct f { f() __attribute__((used)); };
template <class j>
f<j>::f() { h<j>::k(); }
template struct f<a>;
