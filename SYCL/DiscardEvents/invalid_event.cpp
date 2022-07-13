// FIXME: Fails on HIP and OpenCL accelerator
// UNSUPPORTED: hip, (opencl && accelerator)
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// The test checks that each PI call to the queue returns a discarded event
// with the status "ext_oneapi_unknown"

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace cl::sycl;
static constexpr size_t BUFFER_SIZE = 16;

void QueueAPIsReturnDiscardedEvent(sycl::queue Q) {
  sycl::range<1> range(BUFFER_SIZE);

  auto Dev = Q.get_device();
  const int MemAdvice = ((Dev.get_backend() == sycl::backend::ext_oneapi_cuda)
                             ? PI_MEM_ADVICE_CUDA_SET_READ_MOSTLY
                             : PI_MEM_ADVICE_UNKNOWN);
  int *x = sycl::malloc_shared<int>(BUFFER_SIZE, Q);
  assert(x != nullptr);
  int *y = sycl::malloc_shared<int>(BUFFER_SIZE, Q);
  assert(y != nullptr);

  sycl::event DiscardedEvent;

  DiscardedEvent = Q.memset(x, 0, BUFFER_SIZE * sizeof(int));
  assert(
      DiscardedEvent.get_info<sycl::info::event::command_execution_status>() ==
      sycl::info::event_command_status::ext_oneapi_unknown);

  DiscardedEvent = Q.memcpy(y, x, BUFFER_SIZE * sizeof(int));
  assert(
      DiscardedEvent.get_info<sycl::info::event::command_execution_status>() ==
      sycl::info::event_command_status::ext_oneapi_unknown);

  DiscardedEvent = Q.fill(y, 1, BUFFER_SIZE);
  assert(
      DiscardedEvent.get_info<sycl::info::event::command_execution_status>() ==
      sycl::info::event_command_status::ext_oneapi_unknown);

  DiscardedEvent = Q.copy(y, x, BUFFER_SIZE);
  assert(
      DiscardedEvent.get_info<sycl::info::event::command_execution_status>() ==
      sycl::info::event_command_status::ext_oneapi_unknown);

  DiscardedEvent = Q.prefetch(y, BUFFER_SIZE * sizeof(int));
  assert(
      DiscardedEvent.get_info<sycl::info::event::command_execution_status>() ==
      sycl::info::event_command_status::ext_oneapi_unknown);

  DiscardedEvent = Q.mem_advise(y, BUFFER_SIZE * sizeof(int), MemAdvice);
  assert(
      DiscardedEvent.get_info<sycl::info::event::command_execution_status>() ==
      sycl::info::event_command_status::ext_oneapi_unknown);

  DiscardedEvent = Q.single_task([=] {});
  assert(
      DiscardedEvent.get_info<sycl::info::event::command_execution_status>() ==
      sycl::info::event_command_status::ext_oneapi_unknown);

  DiscardedEvent = Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(range, [=](sycl::item<1> itemID) {});
  });
  assert(
      DiscardedEvent.get_info<sycl::info::event::command_execution_status>() ==
      sycl::info::event_command_status::ext_oneapi_unknown);

  DiscardedEvent = Q.ext_oneapi_submit_barrier();
  assert(
      DiscardedEvent.get_info<sycl::info::event::command_execution_status>() ==
      sycl::info::event_command_status::ext_oneapi_unknown);

  Q.wait();
  free(x, Q);
  free(y, Q);
}

int main(int Argc, const char *Argv[]) {
  sycl::property_list Props1{
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue OOO_Queue(Props1);
  QueueAPIsReturnDiscardedEvent(OOO_Queue);

  sycl::property_list Props2{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue Inorder_Queue(Props2);
  QueueAPIsReturnDiscardedEvent(Inorder_Queue);

  std::cout << "The test passed." << std::endl;
  return 0;
}
