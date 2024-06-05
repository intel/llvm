// REQUIRES:  level_zero && windows

// DEFINE: %{sharedflag} = %if cl_options %{/clang:-shared%} %else %{-shared%}

// build shared library
// RUN: %clangxx -fsycl -fPIC %{sharedflag} -o simple_lib.dll %S/Inputs/simple_lib.cpp

// build app
// RUN: %{build} -o %t.out

// RUN: %{run} %t.out
// RUN: env UR_L0_LEAKS_DEBUG=1 %{run} %t.out

#include "Inputs/simple_lib.h"
#include <Windows.h>
#include <assert.h>
#include <iostream>

int main() {
  // Load the library (replace with your Windows loading mechanism)
  HMODULE handle = LoadLibraryA("simple_lib.dll");
  if (!handle) {
    std::cout << "failed to load" << std::endl;
    return 1;
  }

  // Function pointer to the exported function
  int (*add_using_device)(int, int) =
      (int (*)(int, int))GetProcAddress(handle, "add_using_device");
  if (!add_using_device) {
    std::cout << "failed to get function" << std::endl;
    FreeLibrary(handle);
    return 2;
  }

  int result = add_using_device(3, 4);
  std::cout << "Result: " << result << std::endl;
  assert(result == 7);

  FreeLibrary(handle);

  return 0;
}