// RUN: %{build} -I . -o %t.out
// RUN: %{run} %t.out

#include <string>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
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
  int *data = sycl::malloc_device<int>(n, q);
  try {
    q.submit([&](sycl::handler &h) { q.memset(data, 0, n * sizeof(int)); });
  } catch (...) {
    sycl::free(data, q);
    throw;
  }
  sycl::free(data, q);
}

template <typename CommandSubmitterT>
void test(sycl::queue &Queue, CommandSubmitterT QueueSubmit) {
  bool ExceptionHappened = false;
  try {
    QueueSubmit(Queue);
  } catch (const sycl::exception &e) {
    checkExceptionFields(e);
    ExceptionHappened = true;
  }
  assert(ExceptionHappened);
  // Checks that queue is in a valid state: nested call tracker was cleaned up
  // after exception and does not prevent from submission of new commands.
  Queue.submit([&](sycl::handler &h) {});
  Queue.wait();
}

int main() {
  sycl::queue q{};
  test(q, nestedSubmitParallelFor);
  // All shortcut functions has a common part where nested call detection
  // happens. Testing only one of them is enough.
  if (q.get_device().get_info<sycl::info::device::usm_device_allocations>())
    test(q, nestedSubmitMemset);

  return EXIT_SUCCESS;
}
