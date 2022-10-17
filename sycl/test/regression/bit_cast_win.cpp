// RUN: %clangxx -fsycl -fsycl-host-compiler=cl -fsycl-host-compiler-options='/std:c++17 /Zc:__cplusplus' %s -o %t.out
// UNSUPPORTED: linux

#include "bit_cast.hpp"
