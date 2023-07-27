// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

int main() {
  using AccT = sycl::accessor<int, 1, sycl::access::mode::read>;

  // empty accessor doesn't throw when passed to handler::require().
  {
    AccT acc;
    assert(acc.empty());
    sycl::queue q;
    {
      try {
        q.submit([&](sycl::handler &cgh) {
          cgh.require(acc);
        });
        q.wait_and_throw();
      } catch (sycl::exception &e) {
        assert("Unexpected exception");
      } catch (...) {
        std::cout << "Some other unexpected exception (line " << __LINE__ << ")"
                  << std::endl;
        return 1;
      }
    }
  }

  return 0;
}
