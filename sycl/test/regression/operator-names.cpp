// RUN: %clangxx -fsycl -fsyntax-only %s -Wno-deprecated -fno-operator-names
//
//===----------------------------------------------------------------------===//
// This test checks if any SYCL header files use C++ operator name keywords
// e.g. and, or, not
//
// This test does not use -fsyntax-only because it does not appear to respect
// the -fno-operator-names option
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
