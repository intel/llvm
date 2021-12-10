// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -D_HAS_STD_BYTE=0 %s -Xclang -verify-ignore-unexpected=note,warning
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics
#include <CL/sycl.hpp>
