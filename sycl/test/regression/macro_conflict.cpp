// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -o %t.out -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics
//
//===----------------------------------------------------------------------===//
// This test checks if the user-defined macros SUCCESS is
// conflicting with the symbols defined in SYCL header files.
// This test only checks compilation error, so the main function is omitted.
//===----------------------------------------------------------------------===//

#define SUCCESS 0

#include <CL/sycl.hpp>
