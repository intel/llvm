// REQUIRES: level_zero && linux

// DEFINE: %{fPIC_flag} =  %if windows %{%} %else %{-fPIC%}
// build static library
// RUN: %clangxx -fsycl -c  %{fPIC_flag} -o simple_lib.o %S/Inputs/simple_lib.cpp

// build app
// RUN:  %clangxx -fsycl -o %t.out %s simple_lib.o

// RUN: %{run} %t.out
// RUN: env UR_L0_LEAKS_DEBUG=1 %{run} %t.out

// In these tests we are building an intermediate library which uses SYCL and an
// app that employs that intermediate library, using both static and dynamic
// linking, and delayed release. This is to test that release and shutdown are
// working correctly.

/*
    //library
    clang++ -fsycl -c -fPIC -o simple_lib.o Inputs/simple_lib.cpp

    //app
    clang++ -fsycl -o static_app.bin static_app.cpp simple_lib.o

    UR_L0_LEAKS_DEBUG=1 ./simple_app.bin

*/

#include "Inputs/simple_lib.h"
#include <assert.h>
#include <iostream>

int main() {
  int result = add_using_device(3, 4);
  std::cout << "result: " << result << std::endl;
  assert(result == 7);
}
