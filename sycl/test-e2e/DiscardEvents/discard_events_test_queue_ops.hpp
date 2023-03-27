
#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

void CheckArray(sycl::queue Q, int *x, size_t buffer_size, int expected) {
  Q.wait();
  for (size_t i = 0; i < buffer_size; ++i)
    assert(x[i] == expected);
}

static constexpr size_t BUFFER_SIZE = 16;

void TestQueueOperations(sycl::queue Q) {
  sycl::range<1> Range(BUFFER_SIZE);
  auto Dev = Q.get_device();
  auto Ctx = Q.get_context();
  int *x = sycl::malloc_shared<int>(BUFFER_SIZE, Q);
  assert(x != nullptr);
  int *y = sycl::malloc_shared<int>(BUFFER_SIZE, Q);
  assert(y != nullptr);

  Q.memset(x, 0, BUFFER_SIZE * sizeof(int));
  CheckArray(Q, x, BUFFER_SIZE, 0);

  Q.memcpy(y, x, BUFFER_SIZE * sizeof(int));
  CheckArray(Q, y, BUFFER_SIZE, 0);

  Q.fill(y, 1, BUFFER_SIZE);
  CheckArray(Q, y, BUFFER_SIZE, 1);

  Q.copy(y, x, BUFFER_SIZE);
  CheckArray(Q, x, BUFFER_SIZE, 1);

  Q.prefetch(y, BUFFER_SIZE * sizeof(int));
  Q.mem_advise(y, BUFFER_SIZE * sizeof(int), 0);
  Q.ext_oneapi_submit_barrier();

  Q.single_task([=] {
    for (auto i = 0u; i < BUFFER_SIZE; ++i)
      y[i] *= 2;
  });
  CheckArray(Q, y, BUFFER_SIZE, 2);

  Q.parallel_for(Range,
                 [=](sycl::item<1> itemID) { y[itemID.get_id(0)] *= 3; });
  CheckArray(Q, y, BUFFER_SIZE, 6);

  // Creates new queue with the same context/device, but without discard_events
  // property. This queue returns a normal event, not a discarded one.
  sycl::queue RegularQ(Ctx, Dev, sycl::property::queue::in_order{});
  int *x1 = sycl::malloc_shared<int>(BUFFER_SIZE, RegularQ);
  assert(x1 != nullptr);
  auto event = RegularQ.memset(x1, 0, BUFFER_SIZE * sizeof(int));

  Q.memcpy(y, x, 0, event);
  CheckArray(Q, y, BUFFER_SIZE, 6);

  Q.wait();
  free(x, Q);
  free(y, Q);
  free(x1, RegularQ);
}

void TestQueueOperationsViaSubmit(sycl::queue Q) {
  sycl::range<1> Range(BUFFER_SIZE);
  auto Dev = Q.get_device();
  auto Ctx = Q.get_context();
  int *x = sycl::malloc_shared<int>(BUFFER_SIZE, Q);
  assert(x != nullptr);
  int *y = sycl::malloc_shared<int>(BUFFER_SIZE, Q);
  assert(y != nullptr);

  Q.submit(
      [&](sycl::handler &CGH) { CGH.memset(x, 0, BUFFER_SIZE * sizeof(int)); });
  CheckArray(Q, x, BUFFER_SIZE, 0);

  Q.submit(
      [&](sycl::handler &CGH) { CGH.memcpy(y, x, BUFFER_SIZE * sizeof(int)); });
  CheckArray(Q, y, BUFFER_SIZE, 0);

  Q.submit([&](sycl::handler &CGH) { CGH.fill(y, 1, BUFFER_SIZE); });
  CheckArray(Q, y, BUFFER_SIZE, 1);

  Q.submit([&](sycl::handler &CGH) { CGH.copy(y, x, BUFFER_SIZE); });
  CheckArray(Q, x, BUFFER_SIZE, 1);

  Q.submit(
      [&](sycl::handler &CGH) { CGH.prefetch(y, BUFFER_SIZE * sizeof(int)); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.mem_advise(y, BUFFER_SIZE * sizeof(int), 0);
  });
  Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier(); });

  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=] {
      for (auto i = 0u; i < BUFFER_SIZE; ++i)
        y[i] *= 2;
    });
  });
  CheckArray(Q, y, BUFFER_SIZE, 2);

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(Range,
                     [=](sycl::item<1> itemID) { y[itemID.get_id(0)] *= 3; });
  });
  CheckArray(Q, y, BUFFER_SIZE, 6);

  // Creates new queue with the same context/device, but without discard_events
  // property. This queue returns a normal event, not a discarded one.
  sycl::queue RegularQ(Ctx, Dev, sycl::property::queue::in_order{});
  int *x1 = sycl::malloc_shared<int>(BUFFER_SIZE, RegularQ);
  assert(x1 != nullptr);
  auto event = RegularQ.memset(x1, 0, BUFFER_SIZE * sizeof(int));

  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(event);
    CGH.memcpy(y, x, 0);
  });
  CheckArray(Q, y, BUFFER_SIZE, 6);

  Q.wait();
  free(x, Q);
  free(y, Q);
  free(x1, RegularQ);
}
