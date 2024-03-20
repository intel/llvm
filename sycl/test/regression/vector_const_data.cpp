// RUN: %clangxx -fsycl %s -fsyntax-only
#include <sycl/sycl.hpp>
// TODO: There are some spec discussions currently on hold about whether
// sycl::vec should even be allowed to be instantiated with a const-qualified
// type. If that discussion eventually resolves to the answer no, then this test
// will probably fail compilation at line 13 and will need to be deleted.
// The reason it is a test for now is that it verifies that overload resolution
// succeeds for the function template member load of sycl::vec when instantiated
// with a const-qualified type.
int main() {
  sycl::vec<int, 16> my_vec{};
  sycl::vec<const int, 16> my_const_vec{};
  return 0;
}
