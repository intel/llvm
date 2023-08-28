// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-MLIR-DAG: func.func private @func_const() {{.*}} attributes {{{.*}}passthrough = [{{.*}}"nounwind", {{.*}}"willreturn", {{.*}}["memory", "0"]{{.*}}]}
// CHECK-MLIR-DAG: func.func private @func_pure() {{.*}} attributes {{{.*}}passthrough = [{{.*}}"nounwind", {{.*}}"willreturn", {{.*}}["memory", "21"]{{.*}}]}

// CHECK-LLVM-DAG: declare spir_func i32 @func_const() #[[FUNCATTRS1:[0-9]+]]
// CHECK-LLVM-DAG: declare spir_func i32 @func_pure() #[[FUNCATTRS2:[0-9]+]]

// CHECK-LLVM-DAG: attributes #[[FUNCATTRS1]] = { {{.*}}nounwind {{.*}}willreturn memory(none){{.*}} }
// CHECK-LLVM-DAG: attributes #[[FUNCATTRS2]] = { {{.*}}nounwind {{.*}}willreturn memory(read){{.*}} }

#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL [[gnu::const]] int func_const();
extern "C" SYCL_EXTERNAL [[gnu::pure]] int func_pure();

extern "C" SYCL_EXTERNAL int foo() {
  return func_const() + func_pure();
}
