// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -D_HAS_STD_BYTE=0 %s -Xclang -verify-ignore-unexpected=note,warning
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics
#include <CL/sycl.hpp>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif

int main() { return 0; }
