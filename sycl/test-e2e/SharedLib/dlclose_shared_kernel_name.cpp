// Regression test for use-after-free in ProgramManager::removeImages when two
// DSOs define kernels with the same mangled name. After dlclose of the first
// DSO, the kernel name storage is unmapped. A subsequent dlclose of the second
// DSO would segfault while accessing those dangling pointers in
// runtime's data structures (which used string_view keys before the fix).
//
// REQUIRES: linux
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %t.dir/libA_%basename_t.so
// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %t.dir/libB_%basename_t.so
// RUN: %{build} -DFNAME=%basename_t -ldl -Wl,-rpath,%t.dir -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <dlfcn.h>
#include <sycl/detail/core.hpp>

#ifdef BUILD_LIB
// Both DSOs get the same SYCL mangled kernel name.
extern "C" void run_kernel() {
  sycl::queue q;
  sycl::buffer<int, 1> b(1);
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor acc(b, cgh);
     cgh.single_task<class SharedKernel>([=]() { acc[0] = 1; });
   }).wait();
}
#else
#define STRINGIFY_HELPER(A) #A
#define STRINGIFY(A) STRINGIFY_HELPER(A)

int main() {
  void *hA = dlopen("libA_" STRINGIFY(FNAME) ".so", RTLD_NOW | RTLD_LOCAL);
  assert(hA && "failed to dlopen libA");
  void *hB = dlopen("libB_" STRINGIFY(FNAME) ".so", RTLD_NOW | RTLD_LOCAL);
  assert(hB && "failed to dlopen libB");

  auto *fnA = (void (*)())dlsym(hA, "run_kernel");
  auto *fnB = (void (*)())dlsym(hB, "run_kernel");
  assert(fnA && fnB && "failed to dlsym run_kernel");

  fnA();
  fnB();

  // First dlclose unmaps libA's offload table, making any string_view keys
  // pointing into it dangle. Second dlclose triggers the crash on an
  // unpatched runtime because removeImages() hashes/compares those keys.
  dlclose(hA);
  dlclose(hB);

  return 0;
}
#endif
