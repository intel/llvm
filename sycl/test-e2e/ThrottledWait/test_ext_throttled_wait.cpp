// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/ext/oneapi/experimental/throttled_wait.hpp>
#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

// a very big N for looping in long running kernel
constexpr uint64_t N = 1000000000;

void test_wait_and_throw(sycl::queue &q) {
  try {
    sycl::event e = q.submit([&](sycl::handler &CGH) {
      CGH.host_task([=]() {
        throw std::runtime_error("Exception thrown from host_task.");
      });
    });
    syclex::ext_oneapi_throttled_wait_and_throw(e,
                                                std::chrono::milliseconds(100));

    assert(false &&
           "We should not be here. Exception should have been thrown.");
  } catch (std::runtime_error &e) {
    assert(std::string(e.what()) == "Exception thrown from host_task.");
    std::cout << "Caught exception: " << e.what() << std::endl;
  }
}

void test_wait(sycl::queue &q) {
  // fast kernel
  sycl::event fast =
      q.submit([&](sycl::handler &cgh) { cgh.single_task([=]() {}); });
  syclex::ext_oneapi_throttled_wait(fast, std::chrono::milliseconds(100));

  // slow kernel
  uint64_t a = 0;
  {
    sycl::buffer<uint64_t, 1> buf(&a, sycl::range<1>(1));

    sycl::event slow = q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc(buf, cgh, sycl::read_write);
      cgh.single_task<class hello_world>([=]() {
        for (long i = 0; i < N; i++) {
          acc[0] = acc[0] + 1;
        }
      });
    });
    syclex::ext_oneapi_throttled_wait(slow, std::chrono::milliseconds(100));
  } // buffer goes out of scope, data copied back to 'a'.

  std::cout << "a: " << a << std::endl;
  assert(a == N);

  // Ensure compatible with discarded events.
  auto DiscardedEvent = q.ext_oneapi_submit_barrier();
  syclex::ext_oneapi_throttled_wait(DiscardedEvent,
                                    std::chrono::milliseconds(100));
}

std::vector<sycl::event> create_event_list(sycl::queue &q) {
  std::vector<sycl::event> events;
  sycl::event slow = q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
      for (long i = 0; i < N; i++) {
      }
    });
  });
  events.push_back(slow);

  sycl::event fast =
      q.submit([&](sycl::handler &cgh) { cgh.single_task([=]() {}); });
  events.push_back(fast);

  sycl::event DiscardedEvent = q.ext_oneapi_submit_barrier();
  events.push_back(DiscardedEvent);

  return events;
}

void test_wait_event_list(sycl::queue &q) {
  auto events = create_event_list(q);
  syclex::ext_oneapi_throttled_wait(events, std::chrono::milliseconds(100));
}

void test_wait_and_throw_event_list(sycl::queue &q) {
  auto events = create_event_list(q);
  syclex::ext_oneapi_throttled_wait_and_throw(events,
                                              std::chrono::milliseconds(100));
}

int main() {
  auto asyncHandler = [](sycl::exception_list el) {
    for (auto &e : el) {
      std::rethrow_exception(e);
    }
  };
  sycl::queue q(asyncHandler);

#ifdef SYCL_EXT_ONEAPI_THROTTLED_WAIT
  test_wait(q);
  test_wait_and_throw(q);
  test_wait_event_list(q);
  test_wait_and_throw_event_list(q);
#else
  assert(false &&
         "SYCL_EXT_ONEAPI_THROTTLED_WAIT feature test macro not defined");
#endif

  return 0;
}