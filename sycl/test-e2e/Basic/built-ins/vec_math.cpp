// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out

#include "vec_math.hpp"

int main() {
  run_test<sycl::vec>();
  return 0;
}
