// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc -DDEVICE
// RUN: %clang_cc1 -verify -triple spirv64-intel -aux-triple x86_64-unknown-unknown -fopenmp -fopenmp-is-target-device \
// RUN:-fopenmp-host-ir-file-path %t-host.bc -nogpulib %s -emit-llvm -DDEVICE -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc -DTARGET
// RUN: %clang_cc1 -verify -triple spirv64-intel -aux-triple x86_64-unknown-unknown -fopenmp -fopenmp-is-target-device \
// RUN: -fopenmp-host-ir-file-path %t-host.bc -nogpulib %s -emit-llvm -DTARGET -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc -DTARGET_KIND
// RUN: %clang_cc1 -verify -triple spirv64-intel -aux-triple x86_64-unknown-unknown -fopenmp -fopenmp-is-target-device \
// RUN: -fopenmp-host-ir-file-path %t-host.bc -nogpulib %s -emit-llvm -DTARGET_KIND -o - | FileCheck %s --check-prefix=TDK_GPU

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -triple spirv64-intel -aux-triple x86_64-unknown-unknown -fopenmp -fopenmp-is-target-device \
// RUN: -fopenmp-host-ir-file-path %t-host.bc -nogpulib %s -emit-llvm -o - \
// RUN: | FileCheck %s --check-prefix=DK_GPU


// expected-no-diagnostics

#pragma omp declare target
int foo() { return 0; }

#ifdef DEVICE
#pragma omp begin declare variant match(device = {arch(spirv64)})
#elif defined(TARGET)
#pragma omp begin declare variant match(target_device = {arch(spirv64)})
#elif defined(TARGET_KIND)
#pragma omp begin declare variant match(target_device = {kind(gpu)})
#else
#pragma omp begin declare variant match(device = {kind(gpu)})
#endif

int foo() { return 1; }
#pragma omp end declare variant
#pragma omp end declare target

// CHECK-DAG: define{{.*}}  @_Z3foov()

// CHECK-DAG: call {{.*}}@_Z3foov()
// TDK_GPU-DAG: call {{.*}}@"_Z25foo$ompvariant$S3$s8$Pgpuv"()
// DK_GPU-DAG: call {{.*}}@"_Z25foo$ompvariant$S2$s6$Pgpuv"()

int main() {
  int res;
#pragma omp target map(from \
                       : res)
  res = foo();
  return res;
}
