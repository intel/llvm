// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-as %t/device.ll -o %t/in.bc
// RUN: %clang -c -o %t/in.o %t/host.c
// RUN: clang-offload-bundler -type=o -targets=openmp-spir64_gen,host-x86_64-unknown-linux-gnu -input=%t/in.bc -input=%t/in.o -output=%t/out.o
// RUN: llvm-readobj --string-dump=.tgtsym %t/out.o | FileCheck %s

// CHECK-NOT: __AsanKernelMetadata
// CHECK-NOT: __MsanKernelMetadata
// CHECK-NOT: __TsanKernelMetadata
// CHECK-NOT: __AsanDeviceGlobalMetadata
// CHECK-NOT: __MsanDeviceGlobalMetadata
// CHECK-NOT: __TsanDeviceGlobalMetadata
// CHECK: not_skipping
// CHECK: __another_global

//--- host.c
int main() {return 0;}

//--- device.ll
@__AsanKernelMetadata = global i64 0
@__MsanKernelMetadata = global i64 0
@__TsanKernelMetadata = global i64 0
@__AsanDeviceGlobalMetadata = global i64 0
@__MsanDeviceGlobalMetadata = global i64 0
@__TsanDeviceGlobalMetadata = global i64 0

@not_skipping = global i64 0

@__another_global = global i64 0
