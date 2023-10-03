// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-linux -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

struct Struct {
  virtual void foo() {}
  void bar() {}
};

int main() {
  kernel_single_task<class kernel_function>([]() {
                                            Struct S;
                                            S.bar(); });
  return 0;
}


// Struct layout big enough for vtable.
// CHECK: %struct.Struct = type { ptr }
// VTable:
// CHECK: @_ZTV6Struct = linkonce_odr unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI6Struct, ptr @_ZN6Struct3fooEv] }, comdat, align 8
// CHECK: @[[TYPEINFO:.+]] = external addrspace(1) global [0 x ptr addrspace(4)]
// TypeInfo Name:
// CHECK: @_ZTS6Struct = linkonce_odr constant [8 x i8] c"6Struct\00", comdat, align 1
// TypeInfo:
// CHECK: @_ZTI6Struct = linkonce_odr constant { ptr addrspace(4), ptr } { ptr addrspace(4) getelementptr inbounds (ptr addrspace(4), ptr addrspace(4) addrspacecast (ptr addrspace(1) @[[TYPEINFO]] to ptr addrspace(4)), i64 2), ptr @_ZTS6Struct }, comdat, align 8
