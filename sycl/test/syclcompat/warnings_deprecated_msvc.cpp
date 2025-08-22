// Test to verify that syclcompat namespace and APIs generate deprecation warnings.

// REQUIRES: windows
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s -Wall -Wextra

// expected-warning@+1{{warning: syclcompat is deprecated and the deprecation warnings are ignored unless compiled with /W3 or above.}}
#include <syclcompat/syclcompat.hpp>
