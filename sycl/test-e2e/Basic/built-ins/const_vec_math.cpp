// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out

#include "vec_math.hpp"

// Alias is needed to const-qualify vec without template args.
template <typename T, int NumElems>
using ConstVec = const sycl::vec<T, NumElems>;

int main() {
  run_test<ConstVec>();
  return 0;
}
