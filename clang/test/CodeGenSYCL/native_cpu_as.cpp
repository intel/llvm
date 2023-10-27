// This test is temporarily disabled for SYCL Native CPU on Windows
// UNSUPPORTED: system-windows
// Checks that name mangling matches between SYCL Native CPU and OpenCL when -fsycl-is-native-cpu is set
// RUN: %clang_cc1 -DCPP -fsycl-is-device -S -emit-llvm -internal-isystem %S/Inputs -fsycl-is-native-cpu -o %t_sycl.ll %s 
// RUN: FileCheck -input-file=%t_sycl.ll %s 

// RUN: %clang_cc1 -x cl -DOCL -S -emit-llvm -internal-isystem %S/Inputs -fsycl-is-native-cpu -o %t_ocl.ll %s 
// RUN: FileCheck -input-file=%t_ocl.ll %s 

#ifdef CPP
#define AS_LOCAL __attribute((address_space(3)))
#define AS_GLOBAL __attribute((address_space(1)))
#define AS_PRIVATE __attribute((address_space(0)))
#define ATTRS [[intel::device_indirectly_callable]]
#define ATTRS2 SYCL_EXTERNAL
#else 
#ifdef OCL
#define AS_LOCAL __local
#define AS_GLOBAL __global
#define AS_PRIVATE __private
#define ATTRS __attribute((overloadable))
#define ATTRS2 __attribute((overloadable))
#endif
#endif


ATTRS2 void use_private(int *p);
ATTRS  void func(AS_LOCAL int *p1, AS_GLOBAL int *p2, AS_PRIVATE int *p3){
  int private_var;
  use_private(&private_var);
}
// CHECK: define dso_local void @_Z4funcPU3AS3iPU3AS1iPi(
// CHECK: call void @_Z11use_privatePi(



