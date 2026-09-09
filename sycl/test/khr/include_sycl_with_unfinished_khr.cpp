// Including <sycl/sycl.hpp> with the unfinished KHR extensions enabled must
// compile cleanly. This guards against namespace shadowing where KHR headers
// in namespace sycl::khr use bare `detail::Foo` intending sycl::detail::Foo,
// but resolve to a sycl::khr::detail defined by another KHR header.
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>
int main() {}
