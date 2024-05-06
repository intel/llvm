// RUN: %{build} -I . -o %t.out
// RUN: %{run} %t.out

#include <cstdlib>
#include <sycl/sycl.hpp>
#include <vector>

void checkExceptionFields(const sycl::exception &e) {
  assert(e.code() == sycl::errc::invalid && "Invalid error code");
  assert(std::string(e.what()) ==
             "Calls to sycl::queue::submit cannot be nested. Command group "
             "function objects should use the sycl::handler API instead." &&
         "Invalid e.what() string");
}

void nestedSubmitParallelFor(sycl::queue &q) {
  uint32_t n = 1024;
  std::vector<float> array(n);
  {
    sycl::buffer<float> buf(array.data(), sycl::range<1>{n});
    q.submit([&](sycl::handler &h) {
      auto acc = buf.get_access<sycl::access::mode::write>(h);
      q.parallel_for<class zero>(sycl::range<1>{n},
                                 [=](sycl::id<1> i) { acc[i] = float(0.0); });
    });
  }
}

void nestedSubmitMemset(sycl::queue &q) {
  uint32_t n = 1024;
  int *data = sycl::malloc_shared<int>(n, q);
  try {
    q.submit([&](sycl::handler &h) { q.memset(data, 0, n * sizeof(int)); });
  } catch (...) {
    sycl::free(data, q);
    throw;
  }
  sycl::free(data, q);
}

void doValidCall(sycl::queue &q) {
  q.submit([&](sycl::handler &h) {});
}

template <typename CommandSubmitterT> void test(CommandSubmitterT QueueSubmit) {
  sycl::queue q{};
  bool ExceptionHappened = false;
  try {
    QueueSubmit(q);
  } catch (const sycl::exception &e) {
    checkExceptionFields(e);
    ExceptionHappened = true;
  }
  assert(ExceptionHappened);
  // Checks that queue is in a valid state and we could submit new commands.
  doValidCall(q);
  q.wait();
}

int main() {
  test(nestedSubmitParallelFor);
  // All shortcut functions has a common part where nested call detection
  // happens. Testing only one of them is enough.
  test(nestedSubmitMemset);

  return EXIT_SUCCESS;
}
