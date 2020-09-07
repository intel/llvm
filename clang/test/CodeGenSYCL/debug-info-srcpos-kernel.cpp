// RUN: %clang -fsycl-device-only %s -S -emit-llvm -O0 -I %S/Inputs -g -o - | FileCheck %s
//
// Verify the SYCL kernel routine is marked artificial and has no source
// correlation.
//
// In order to placate the profiling tools, which can't cope with instructions
// mapped to line 0, we've made the change so that the artificial code in a
// SYCL kernel gets the source line info for the kernel caller function (the
// 'kernel' template function on line 15 in this file).
//

#include <sycl.hpp>

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
// CHECK: getelementptr inbounds %"class.{{.*}}.anon"{{.*}} !dbg [[LINE_A0:![0-9]+]]
// CHECK: call spir_func void {{.*}}6__init{{.*}} !dbg [[LINE_A0]]
// CHECK: call spir_func void @"_ZZ4mainENK3$_0clEv"{{.*}} !dbg [[LINE_B0:![0-9]+]]
// CHECK: ret void
// CHECK: [[FILE:![0-9]+]] = !DIFile(filename: "{{.*}}debug-info-srcpos-kernel.cpp"{{.*}})
// CHECK: [[KERNEL]] = {{.*}}!DISubprogram(name: "{{.*}}19use_kernel_for_test"
// CHECK-SAME: scope: [[FILE]]
// CHECK-SAME: file: [[FILE]]
// CHECK-SAME: flags: DIFlagArtificial | DIFlagPrototyped
// CHECK: [[LINE_A0]] = !DILocation(line: 15,{{.*}}scope: [[KERNEL]]
// CHECK: [[LINE_B0]] = !DILocation(line: 0

// TODO: [[LINE_B0]] should be mapped to line 15 as well. That said,
// this 'line 0' assignment is less problematic as the lambda function
// call would be inlined in most cases.
// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
