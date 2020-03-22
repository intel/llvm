// RUN: %clangxx -fsycl %s -c

// Regression tests for https://github.com/intel/llvm/issues/1011
// Checks that SYCL headers call internal templated function 'printArgs'
// with fully quilified name to not confuse ADL.
//

template <typename TArg0, typename... TArgs>
auto printArgs(TArg0 arg, TArgs... args) {
}

#include <CL/sycl.hpp>

int main() {
  return 0;
}
