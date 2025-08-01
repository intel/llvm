// RUN: %clangxx -fsycl -fsycl-host-compiler=g++ -fsycl-host-compiler-options='-std=c++17' %s -o %t.out
// UNSUPPORTED: windows
// XFAIL: libcxx
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19616

#include "bit_cast.hpp"
