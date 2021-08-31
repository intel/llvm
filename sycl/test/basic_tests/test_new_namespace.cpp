// RUN: %clangxx -fsycl -fsyntax-only %s

// The test checks that compilation with SYCL 2020 style namespaces (sycl
// instead of cl::sycl) works fine
//
// This test is temporary one which should make sure that such a compilation
// mode is not broken until complete transition happens.

#define __SYCL_DISABLE_SYCL121_NAMESPACE

#include <sycl/sycl.hpp>

int main() {}
