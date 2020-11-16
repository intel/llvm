// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s -c -o %t.out
// RUN: %clangxx -std=c++14 -fsyntax-only -Xclang -verify -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning %s -o -c %t.out
// RUN: %clangxx -std=c++17 -fsyntax-only -Xclang -verify -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning %s -o -c %t.out
// RUN: %clangxx -std=c++20 -fsyntax-only -Xclang -verify -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning %s -o -c %t.out
// expected-no-diagnostics

#include <CL/sycl.hpp>
