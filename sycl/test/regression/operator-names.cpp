// RUN: %clangxx -fsycl -fsyntax-only %s -Wno-deprecated -fno-operator-names
//
//===----------------------------------------------------------------------===//
// This test checks if any SYCL header files use C++ operator name keywords
// e.g. and, or, not
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
