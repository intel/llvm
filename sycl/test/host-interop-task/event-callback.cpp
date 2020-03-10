// RUN: %clangxx -fsycl %s -o %t.out %threads_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <thread>

#include <CL/sycl.hpp>
#include <CL/sycl/detail/event_impl.hpp>

namespace S = cl::sycl;

struct Context {
  std::atomic_bool Flag;
  S::queue &Queue;
  std::string Message;
  S::buffer<int, 1> Buf;
  std::mutex Mutex;
  std::condition_variable CV;
};

void Thread1Fn(Context &Ctx) {
  // T1.1. submit device-side kernel K1
  Ctx.Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::global_buffer> GeneratorAcc(Ctx.Buf, CGH);

    auto GeneratorKernel = [GeneratorAcc] () {
      for (size_t Idx = 0; Idx < GeneratorAcc.get_count(); ++Idx)
        GeneratorAcc[Idx] = Idx;
    };
    CGH.single_task<class GeneratorTaskC>(GeneratorKernel);
  })
  // T1.2. submit host task using event of K1 as a lock with callback to set
  //       flag F = true
  .when_complete([&Ctx] () {
    bool Expected = false;
    bool Desired = true;
    assert(Ctx.Flag.compare_exchange_strong(Expected, Desired));

    // let's employ some locking here
    {
      std::lock_guard<std::mutex> Lock(Ctx.Mutex);
      Ctx.CV.notify_all();
    }
  });
}

void Thread2Fn(Context &Ctx) {
  std::unique_lock<std::mutex> Lock(Ctx.Mutex);

  // T2.1. Wait until flag F is set eq true.
  Ctx.CV.wait(Lock, [&Ctx] { return Ctx.Flag.load(); });

  assert(Ctx.Flag.load());

  // T2.2. print some "hello, world" message
  Ctx.Message = "Hello, world";
}

void test() {
  auto EH = [] (S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

  // optional
  Queue.set_event_cb_thread_pool_size(4);

  Context Ctx{{false}, Queue, "", {10}};

  // 0. setup: thread 1 T1: exec smth; thread 2 T2: waits; init flag F = false
  std::thread Thread1(Thread1Fn, std::reference_wrapper<Context>(Ctx));
  std::thread Thread2(Thread2Fn, std::reference_wrapper<Context>(Ctx));

  Thread1.join();
  Thread2.join();

  assert(Ctx.Flag.load());
  assert(Ctx.Message == "Hello, world");
}

int main(void) {
  test();

  return 0;
}
