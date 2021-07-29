// RUN: %clangxx -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// RUN: %clangxx -std=c++14 -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// RUN: %clangxx -std=c++17 -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// RUN: %clangxx -std=c++20 -fsycl --no-system-header-prefix=CL/sycl -Wall -pedantic -Wno-c99-extensions -Wno-deprecated -fsyntax-only -Xclang -verify %s -c -o %t.out
// expected-no-diagnostics

#include <CL/sycl.hpp>
