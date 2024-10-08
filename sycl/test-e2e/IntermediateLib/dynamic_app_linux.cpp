// REQUIRES: level_zero && linux

// build shared library
// RUN: %clangxx -fsycl -fPIC -shared -o %T/simple_lib.so %S/Inputs/simple_lib.cpp

// build app
// RUN: %clangxx -DSO_PATH="%T/simple_lib.so" -o %t.out %s

// RUN: %{run} %t.out
// RUN: env UR_L0_LEAKS_DEBUG=1 %{run} %t.out

// In these tests we are building an intermediate library which uses SYCL and an
// app that employs that intermediate library, using both static and dynamic
// linking, and delayed release. This is to test that release and shutdown are
// working correctly.

/*
    //library
    clang++ -fsycl  -fPIC -shared -o simple_lib.so Inputs/simple_lib.cpp

    //app
    clang++ -DSO_PATH="simple_lib.so" -o dynamic_app.bin dynamic_app_linux.cpp

    UR_L0_LEAKS_DEBUG=1 ./dynamic_app.bin

*/

#include "Inputs/simple_lib.h"
#include <assert.h>
#include <dlfcn.h>
#include <iostream>

void *handle = nullptr;

__attribute__((destructor(101))) static void Unload101() {
  std::cout << "app unload - __attribute__((destructor(101)))" << std::endl;
  if (handle) {
    dlclose(handle);
    handle = nullptr;
  }
}

#define STRINGIFY_HELPER(A) #A
#define STRINGIFY(A) STRINGIFY_HELPER(A)
#define SO_FNAME "" STRINGIFY(SO_PATH) ""

int main() {

  handle = dlopen(SO_FNAME, RTLD_NOW);
  if (!handle) {
    std::cout << "failed to load" << std::endl;
    return 1;
  }

  // Function pointer to the exported function
  int (*add_using_device)(int, int) =
      (int (*)(int, int))dlsym(handle, "add_using_device");
  if (!add_using_device) {
    std::cout << "failed to get function" << std::endl;
    return 2;
  }

  int result = add_using_device(3, 4);
  std::cout << "Result: " << result << std::endl;
  assert(result == 7);

  return 0;
}