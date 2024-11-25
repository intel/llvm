// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// The purpose of this test is to ensure that the header containing
// sycl::handler class definition is self-contained, i.e. we can use handler
// and no extra headers are needed.
//
// TODO: the test should be expanded to use various methods of the class. Due
// to their template nature we may not test all code paths until we trigger
// instantiation of a corresponding method.

#include <sycl/handler.hpp>

class kernel_name;

void foo(sycl::handler &h) {}
