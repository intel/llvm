// RUN: %clang_cc1 -fcuda-is-device -fdenormal-fp-math-f32=preserve-sign \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefixes=FTZ32,PTXFTZ32 %s

// RUN: %clang_cc1 -fcuda-is-device -fdenormal-fp-math=preserve-sign \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefixes=FTZ,PTXFTZ %s

// CHECK-LABEL: define void @_Z3foov() #0
void foo() {}

// FTZ32: attributes #0 = {{.*}} "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// PTXFTZ32:!llvm.module.flags = !{{{.*}}, [[MODFLAG:![0-9]+]], {{.*}}}
// PTXFTZ32:[[MODFLAG]] = !{i32 7, !"nvvm-reflect-ftz", i32 1}

// FTZ: attributes #0 = {{.*}} "denormal-fp-math"="preserve-sign,preserve-sign"
// PTXFTZ:!llvm.module.flags = !{{{.*}}, [[MODFLAG:![0-9]+]], {{.*}}}
// PTXFTZ:[[MODFLAG]] = !{i32 7, !"nvvm-reflect-ftz", i32 1}
