// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -D__SYCL_USE_LIBSYCL8_VEC_IMPL=1 %{mathflags} -o %t2.out
// RUN: %{run} %t2.out

#include "vec_math.hpp"

// Alias is needed to const-qualify vec without template args.
template <typename T, int NumElems>
using ConstVec = const sycl::vec<T, NumElems>;

int main() {
  run_test<ConstVec>();
  return 0;
}
