// REQUIRES: native_cpu_ock
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -O0 -mllvm -sycl-native-cpu-no-vecz -mllvm -sycl-native-dump-device-ir %s | FileCheck %s

// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -fno-inline -Xclang -sycl-std=2020 -mllvm -sycl-opt -S -emit-llvm  %s -o - | FileCheck %s --check-prefix=CHECK-DEV

// check that builtins are defined

// CHECK-NOT: {{.*}}__spirv_GenericCastToPtrExplicit
// CHECK-DEV: {{.*}}__spirv_GenericCastToPtrExplicit

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

#define DefTestCast(FName, Space, PType)\
SYCL_EXTERNAL auto FName(PType p) {\
  return dynamic_address_cast<Space>(p);\
}

#define DefTestCastForSpace(PType)\
DefTestCast(to_local, access::address_space::local_space, PType)\
DefTestCast(to_global, access::address_space::global_space, PType)\
DefTestCast(to_private, access::address_space::private_space, PType)\
DefTestCast(to_generic, access::address_space::generic_space, PType)

DefTestCastForSpace(int*)
DefTestCastForSpace(const int*)
DefTestCastForSpace(volatile int*)
DefTestCastForSpace(const volatile int*)

int main(){}
// check that the generated module has the is-native-cpu module flag set
// CHECK: !{{[0-9]*}} = !{i32 1, !"is-native-cpu", i32 1}
