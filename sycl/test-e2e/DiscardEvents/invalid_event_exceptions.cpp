// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// The test checks 3 things:
// 1. An attempt to construct a queue with both properties(discard_events and
// enable_profiling) throws an exception.
// 2. Checks the APIs for discarded event that should throw an exception that
// they do it.
// 3. An attempt to pass discarded event into depends_on throws an exception.

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

void DiscardedEventWaitExceptionHelper(
    const std::function<void()> &FunctionToTry) {
  try {
    FunctionToTry();
    assert(false && "No exception was thrown.");
  } catch (const sycl::exception &e) {
    assert(e.code().value() == static_cast<int>(sycl::errc::invalid) &&
           "sycl::exception code was not the expected sycl::errc::invalid.");
  } catch (...) {
    assert(false &&
           "Unexpected exception was thrown in kernel invocation function.");
  }
}

void DependsOnDiscardedEventException(sycl::queue Q) {
  auto DiscardedEvent =
      Q.submit([&](sycl::handler &CGH) { CGH.single_task([] {}); });

  Q.submit([&](sycl::handler &CGH) {
    try {
      CGH.depends_on(DiscardedEvent);
      assert(false && "No exception was thrown.");
    } catch (const sycl::exception &e) {
      assert(e.code().value() == static_cast<int>(sycl::errc::invalid) &&
             "sycl::exception code was not the expected sycl::errc::invalid.");
    } catch (...) {
      assert(false &&
             "Unexpected exception was thrown in kernel invocation function.");
    }
    CGH.single_task([] {});
  });

  sycl::event e1, e2;
  Q.submit([&](sycl::handler &CGH) {
    try {
      CGH.depends_on({e1, DiscardedEvent, e2});
      assert(false && "No exception was thrown.");
    } catch (const sycl::exception &e) {
      assert(e.code().value() == static_cast<int>(sycl::errc::invalid) &&
             "sycl::exception code was not the expected sycl::errc::invalid.");
    } catch (...) {
      assert(false &&
             "Unexpected exception was thrown in kernel invocation function.");
    }
    CGH.single_task([] {});
  });

  sycl::queue RegularQ;
  RegularQ.submit([&](sycl::handler &CGH) {
    try {
      CGH.depends_on(DiscardedEvent);
      assert(false && "No exception was thrown.");
    } catch (const sycl::exception &e) {
      assert(e.code().value() == static_cast<int>(sycl::errc::invalid) &&
             "sycl::exception code was not the expected sycl::errc::invalid.");
    } catch (...) {
      assert(false &&
             "Unexpected exception was thrown in kernel invocation function.");
    }
    CGH.single_task([] {});
  });

  RegularQ.submit([&](sycl::handler &CGH) {
    try {
      CGH.depends_on({e1, DiscardedEvent, e2});
      assert(false && "No exception was thrown.");
    } catch (const sycl::exception &e) {
      assert(e.code().value() == static_cast<int>(sycl::errc::invalid) &&
             "sycl::exception code was not the expected sycl::errc::invalid.");
    } catch (...) {
      assert(false &&
             "Unexpected exception was thrown in kernel invocation function.");
    }
    CGH.single_task([] {});
  });
}

void CheckDiscardedEventAPIException(sycl::queue Q) {
  DiscardedEventWaitExceptionHelper([&]() {
    auto DiscardedEvent =
        Q.submit([&](sycl::handler &CGH) { CGH.single_task([] {}); });
    DiscardedEvent.wait();
  });

  DiscardedEventWaitExceptionHelper([&]() {
    auto DiscardedEvent =
        Q.submit([&](sycl::handler &CGH) { CGH.single_task([] {}); });
    sycl::event::wait({DiscardedEvent});
  });

  DiscardedEventWaitExceptionHelper([&]() {
    auto DiscardedEvent =
        Q.submit([&](sycl::handler &CGH) { CGH.single_task([] {}); });
    DiscardedEvent.wait_and_throw();
  });

  DiscardedEventWaitExceptionHelper([&]() {
    auto DiscardedEvent =
        Q.submit([&](sycl::handler &CGH) { CGH.single_task([] {}); });
    sycl::event::wait_and_throw({DiscardedEvent});
  });

  DiscardedEventWaitExceptionHelper([&]() {
    auto DiscardedEvent =
        Q.submit([&](sycl::handler &CGH) { CGH.single_task([] {}); });
    DiscardedEvent.get_wait_list();
  });
}

void CreatingEnableProfilingQueueException(sycl::property_list Props) {
  try {
    sycl::queue Q{Props};
    assert(false && "No exception was thrown.");
  } catch (const sycl::exception &e) {
    assert(e.code().value() == static_cast<int>(sycl::errc::invalid) &&
           "sycl::exception code was not the expected sycl::errc::invalid.");
  } catch (...) {
    assert(false &&
           "Unexpected exception was thrown in kernel invocation function.");
  }
}

int main(int Argc, const char *Argv[]) {
  sycl::property_list Props1{
      sycl::property::queue::enable_profiling{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  CreatingEnableProfilingQueueException(Props1);

  sycl::property_list Props2{
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue OOO_Queue(Props2);
  DependsOnDiscardedEventException(OOO_Queue);
  CheckDiscardedEventAPIException(OOO_Queue);

  sycl::property_list Props3{
      sycl::property::queue::in_order{},
      sycl::property::queue::enable_profiling{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  CreatingEnableProfilingQueueException(Props3);

  sycl::property_list Props4{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue Inorder_Queue(Props4);
  DependsOnDiscardedEventException(Inorder_Queue);
  CheckDiscardedEventAPIException(Inorder_Queue);

  std::cout << "The test passed." << std::endl;
  return 0;
}
