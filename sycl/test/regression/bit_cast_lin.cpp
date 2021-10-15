// RUN: %clangxx -fsycl -fsycl-host-compiler=g++ -fsycl-host-compiler-options='-std=c++17' %s -o %t.out
// UNSUPPORTED: windows

#include "bit_cast.hpp"
