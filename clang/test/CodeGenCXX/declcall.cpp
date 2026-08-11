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
// An *unqualified* virtual call is NOT devirtualized: the constant keeps
// virtual dispatch, i.e. a vtable index ({ i64 1, i64 0 }).
// CHECK: @pu = {{.*}}global { i64, i64 } { i64 1, i64 0 }
// declcall on an operator call selects the operator function: a free operator
// yields a function pointer, a member operator yields a member pointer.
// CHECK: @pfree = {{.*}}global ptr @_Zpl2OpS_
// CHECK: @pmem = {{.*}}global { i64, i64 } { i64 ptrtoint (ptr @_ZN2OmmiES_ to i64), i64 0 }
struct B { virtual int g(int); int h(int); };
auto pv = declcall(((B *)0)->B::g(0));
auto ph = declcall(((B *)0)->h(0));
auto pu = declcall(((B *)0)->g(0));

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

// declcall selects and instantiates a function template specialization, and
// that specialization is emitted.
template <class T> T tfn(T v) { return v; }
// CHECK-LABEL: define {{.*}} ptr @_Z8get_tmplv()
// CHECK: ret ptr @_Z3tfnIiET_S0_
// CHECK: define {{.*}} @_Z3tfnIiET_S0_
FP get_tmpl() { return declcall(tfn(0)); }

// Operator-call declcall values (checked with the other globals above).
struct Op {};
int operator+(Op, Op);
struct Om { int operator-(Om); };
auto pfree = declcall(Op{} + Op{});
auto pmem = declcall(((Om *)0)->operator-(Om{}));
