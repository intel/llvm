// RUN: %clang --sycl %s -S -I %S/Inputs -emit-llvm -g -o - | FileCheck %s
//
// Verify the SYCL kernel routine is marked artificial and has no source
// correlation.
//
// The SYCL kernel should have no source correlation of its own, so it needs
// to be marked artificial or it will inherit source correlation from the
// surrounding code.
//

#include <sycl.hpp>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
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
// CHECK: ret void, !dbg [[LINE_A0]]
// CHECK: [[FILE:![0-9]+]] = !DIFile(filename: "{{.*}}debug-info-srcpos-kernel.cpp"{{.*}})
// CHECK: [[KERNEL]] = {{.*}}!DISubprogram(name: "{{.*}}19use_kernel_for_test"
// CHECK-SAME: scope: [[FILE]]
// CHECK-SAME: file: [[FILE]]
// CHECK-SAME: flags: DIFlagArtificial | DIFlagPrototyped
// CHECK: [[LINE_A0]] = !DILocation(line: 0
// CHECK: [[LINE_B0]] = !DILocation(line: 0

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
