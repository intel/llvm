// RUN: %clangxx -DPOSITIVE_TESTING -DSYCL121_HEADER -fsycl %s
// RUN: %clangxx -DPOSITIVE_TESTING -fsycl %s
// RUN: %clangxx -fsycl %s -Xclang -verify -fsyntax-only

#ifdef POSITIVE_TESTING

#ifdef SYCL121_HEADER
#include <CL/sycl.hpp>

int main() {
  cl::sycl::queue q;
  sycl::buffer<int, 1> A{sycl::range<1>{4}};
}

#else // SYCL121_HEADER

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  sycl::buffer<int, 1> A{sycl::range<1>{4}};
}

#endif // SYCL121_HEADER

#else // POSITIVE_TESTING

#include <sycl/sycl.hpp>

int main() {
  cl::sycl::queue q; // expected-error {{use of undeclared identifier 'cl'}}
}

#endif // POSITIVE_TESTING
