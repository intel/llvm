/// Verify object removal when the offload compilation fails.

// REQUIRES: system-linux

// DEFINE: %{clang_install_dir} = %/S/Inputs/SYCL/bin

/// Force failure during device compilation
// RUN: touch %t.o
// RUN: not %clangxx -fsycl -Xsycl-target-frontend -DCOMPILE_FAIL=1 -c -o %t.o %s
// RUN: not ls %t.o

// RUN: touch %t.o
// RUN: not %clangxx --offload-new-driver -ccc-install-dir %{clang_install_dir} -fsycl -Xsycl-target-frontend -DCOMPILE_FAIL=1 -c -o %t.o %s
// RUN: not ls %t.o

/// Force failure during compilation
// RUN: touch %t.o
// RUN: not %clangxx -fsycl -DCOMPILE_FAIL=1 -c -o %t.o %s
// RUN: not ls %t.o

// RUN: touch %t.o
// RUN: not %clangxx --offload-new-driver -ccc-install-dir %{clang_install_dir} -fsycl -DCOMPILE_FAIL=1 -c -o %t.o %s
// RUN: not ls %t.o

void func(){};
#ifdef COMPILE_FAIL
#error FAIL
#endif // COMPILE_FAIL
