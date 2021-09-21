// RUN: %clang -fsycl-device-only %s -S -emit-llvm -O0 -g -o - | FileCheck %s
//
// Verify the SYCL kernel routine is marked artificial and has the
// expected source correlation.
//
// In order to placate the profiling tools, which can't cope with instructions
// mapped to line 0, we've made the change so that the artificial code in a
// SYCL kernel gets the source line info for the kernel caller function (the
// 'kernel' template function on line 15 in this file).
//

#include "Inputs/sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::sampler Sampler;
  kernel<class use_kernel_for_test>([=]() {
    Sampler.use();
  });
  return 0;
}

// CHECK: define{{.*}} spir_kernel {{.*}}19use_kernel_for_test({{.*}}){{.*}} !dbg [[KERNEL:![0-9]+]] {{.*}}{
// CHECK: getelementptr inbounds %class.{{.*}}.anon{{.*}} !dbg [[LINE_A0:![0-9]+]]
// CHECK: call spir_func void {{.*}}6__init{{.*}} !dbg [[LINE_A0]]
// CHECK: call spir_func void @_ZZ4mainENKUlvE_clEv{{.*}} !dbg [[LINE_B0:![0-9]+]]
// CHECK: ret void, !dbg [[LINE_C0:![0-9]+]]
// CHECK: [[KERNEL]] = {{.*}}!DISubprogram(name: "{{.*}}19use_kernel_for_test"
// CHECK-SAME: scope: [[FILE:![0-9]+]],
// CHECK-SAME: file: [[FILE]],
// CHECK-SAME: flags: DIFlagArtificial | DIFlagPrototyped
// CHECK: [[FILE]] = !DIFile(filename: "{{.*}}debug-info-srcpos-kernel.cpp"{{.*}})
// CHECK: [[LINE_A0]] = !DILocation(line: 15,{{.*}}scope: [[KERNEL]]
// CHECK: [[LINE_B0]] = !DILocation(line: 16,{{.*}}scope: [[BLOCK:![0-9]+]]
// CHECK: [[BLOCK]] = distinct !DILexicalBlock(scope: [[KERNEL]]
// CHECK: [[LINE_C0]] = !DILocation(line: 17,{{.*}}scope: [[KERNEL]]
