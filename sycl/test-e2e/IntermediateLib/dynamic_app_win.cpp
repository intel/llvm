// REQUIRES:  level_zero && windows

// DEFINE: %{sharedflag} = %if cl_options %{/clang:-shared%} %else %{-shared%}

// build shared library
// RUN: %clangxx -fsycl  %{sharedflag} -o simple_lib.dll %S/Inputs/simple_lib.cpp

// build app
// RUN: %clangxx %s -o %t.out

// RUN: %{run} %t.out
// RUN: env UR_L0_LEAKS_DEBUG=1 %{run} %t.out

/*
  clang++ -fsycl -shared -o simple_lib.dll ./Inputs/simple_lib.cpp

  clang++ -o dynamic_app_win.exe dynamic_app_win.cpp

*/

#include "Inputs/simple_lib.h"
#include <Windows.h>
#include <assert.h>
#include <iostream>

int main() {
  HMODULE handle = LoadLibraryA("simple_lib.dll");
  if (!handle) {
    std::cout << "failed to load" << std::endl;
    return 1;
  }

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
