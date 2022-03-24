// This test is intended to check a certain approach of compiling libraries and
// application, when both regular SYCL and ESIMD are used.
//
// We used to have a bug, when under some circumstances compiler created empty
// device images, but at the same time it stated that they contain some kernels.
// More details can be found in intel/llvm#4927.
//
// REQUIRES: linux,gpu
// UNSUPPORTED: cuda || hip
// TODO/DEBUG Segmentation fault occurs with esimd_emulator backend
// XFAIL: esimd_emulator
//
// RUN: %clangxx -fsycl -fPIC -O3 %S/Inputs/complex-lib-sycl.cpp -c -o %t-lib-sycl.o
// RUN: %clangxx -fsycl -fPIC -O3 %S/Inputs/complex-lib-esimd.cpp -c -o %t-lib-esimd.o
// RUN: %clangxx -fsycl -fPIC -O3 %S/Inputs/complex-lib-test.cpp -c -o %t-test.o
//
// RUN: ar crsv %t-lib-sycl.a %t-lib-sycl.o
// RUN: ar crsv %t-lib-esimd.a %t-lib-esimd.o
//
// One shared library is built using static libraries
//
// RUN: %clangxx -fsycl -shared %t-lib-sycl.a %t-lib-esimd.a \
// RUN:  -fsycl-device-code-split=per_kernel -Wl,--whole-archive \
// RUN:  %t-lib-sycl.a %t-lib-esimd.a -Wl,--no-whole-archive -Wl,-soname,%S -o %t-lib-a.so
//
// And another one is constructed directly from object files
//
// RUN: %clangxx -fsycl -shared %t-lib-sycl.o %t-lib-esimd.o \
// RUN:  -fsycl-device-code-split=per_kernel -Wl,-soname,%S -o %t-lib-o.so
//
// RUN: %clangxx -fsycl %t-test.o %t-lib-a.so -o %t-a.run
// RUN: %clangxx -fsycl %t-test.o %t-lib-o.so -o %t-o.run
//
// FIXME: is there better way to handle libraries loading than LD_PRELOAD?
// There is no LIT substitution, which would point to a directory, where
// temporary files are located. There is %T, but it is marked as "deprecated,
// do not use"
// RUN: %GPU_RUN_PLACEHOLDER LD_PRELOAD=%t-lib-a.so %t-a.run
// RUN: %GPU_RUN_PLACEHOLDER LD_PRELOAD=%t-lib-o.so %t-o.run
