// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s -c -o %t.out
// RUN: %clangxx -fsycl -std=c++14 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s -c -o %t.out
// RUN: %clangxx -fsycl -std=c++17 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s -c -o %t.out
// RUN: %clangxx -fsycl -std=c++20 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s -c -o %t.out
// expected-no-diagnostics

#include <CL/sycl.hpp>
