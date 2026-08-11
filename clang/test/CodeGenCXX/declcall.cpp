// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -fexperimental-new-constant-interpreter -emit-llvm %s -o - | FileCheck %s

int f(int);
int f(char);
using FP = int (*)(int);

// A qualified call to a virtual member is devirtualized: the constant is a
// direct pointer to the member, not a vtable index.
// CHECK: @pv = {{.*}}global { i64, i64 } { i64 ptrtoint (ptr @_ZN1B1gEi to i64), i64 0 }
// A non-virtual member is a direct pointer as well.
// CHECK: @ph = {{.*}}global { i64, i64 } { i64 ptrtoint (ptr @_ZN1B1hEi to i64), i64 0 }
struct B { virtual int g(int); int h(int); };
auto pv = declcall(((B *)0)->B::g(0));
auto ph = declcall(((B *)0)->h(0));

// A free-function declcall lowers to the address of the selected overload.
// CHECK-LABEL: define {{.*}} ptr @_Z3getv()
// CHECK: ret ptr @_Z1fi
FP get() { return declcall(f(0)); }

// A declcall in a local (non-constant-initializer) context is emitted through
// the scalar emitter; it must still produce a devirtualized direct pointer,
// not a vtable index.
// CHECK-LABEL: define {{.*}} @_Z3usePM1BFiiE
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr @_ZN1B1gEi to i64), i64 0 }
void use(int (B::**out)(int)) {
  auto pv = declcall(((B *)0)->B::g(0));
  *out = pv;
}
