/// Verify object removal when the offload compilation fails.

// REQUIRES: system-linux

// RUN: touch %t.o
// RUN: not %clangxx -fsycl -Xsycl-target-frontend -DCOMPILE_HOST_FAIL=1 -o %t.o %s
// RUN: not ls %t.o

// RUN: touch %t.o
// RUN: not %clangxx --offload-new-driver -fsycl -Xsycl-target-frontend -DCOMPILE_HOST_FAIL=1 -o %t.o %s
// RUN: not ls %t.o

void func(){};
#ifdef COMPILE_HOST_FAIL
#error FAIL
#endif // COMPILE_HOST_FAIL
