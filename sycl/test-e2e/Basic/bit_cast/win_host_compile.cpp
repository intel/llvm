
// REQUIRES: windows

// RUN: %clangxx -fsycl -fsycl-host-compiler=cl -fsycl-host-compiler-options='/std:c++17 /Zc:__cplusplus'  -o %t.out  %p/bit_cast.cpp
// RUN: %{run} %t.out

// This test depends on its neighbor 'bit_cast.cpp'.
