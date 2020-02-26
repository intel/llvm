// RUN: %clangxx -fsycl %s -o %t.out
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

void ThreadA1Fn(Context &Ctx) {
  // T1.1. submit device-side kernel K1
  auto Event = Ctx.Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::global_buffer> GeneratorAcc(Ctx.Buf, CGH);

    auto GeneratorKernel = [GeneratorAcc] () {
      for (size_t Idx = 0; Idx < GeneratorAcc.get_count(); ++Idx)
        GeneratorAcc[Idx] = Idx;
    };
    CGH.single_task<class GeneratorTaskA>(GeneratorKernel);
  });

  // T1.2. submit host task using event of K1 as a lock with callback to set
  //       flag F = true
  Ctx.Queue.submit([&](S::handler &CGH) {
    CGH.depends_on(Event);

    auto Callback = [&Ctx, Event] (const std::vector<S::event> &E) -> void {
      assert(E.size() == 1);

      // operator== of cl::sycl::event will only compare pointers to impls.
      // We want to compare underlying impl objects, though.
      assert(S::detail::getSyclObjImpl(Event)->get() == S::detail::getSyclObjImpl(E[0])->get());

      bool Expected = false;
      bool Desired = true;
      assert(Ctx.Flag.compare_exchange_strong(Expected, Desired));

      // let's employ some locking here
      {
        std::lock_guard<std::mutex> Lock(Ctx.Mutex);
        Ctx.CV.notify_all();
      }
    };

    // The Callback is run on Queue-internal thread-pool or in backend's thread
    // if thread pool size is explicitly set to 0
    CGH.host_task(Callback);
  });
}

void ThreadB1Fn(Context &Ctx) {
  // T1.1. submit device-side kernel K1
  Ctx.Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::global_buffer> GeneratorAcc(Ctx.Buf, CGH);

    auto GeneratorKernel = [GeneratorAcc] () {
      for (size_t Idx = 0; Idx < GeneratorAcc.get_count(); ++Idx)
        GeneratorAcc[Idx] = Idx;
    };
    CGH.single_task<class GeneratorTaskB>(GeneratorKernel);
  });

  // T1.2. submit host task using event of K1 as a lock with callback to set
  //       flag F = true
  Ctx.Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::read,
                S::access::target::host_buffer> TestAcc(Ctx.Buf, CGH);

    auto Callback = [&Ctx, TestAcc] (const std::vector<S::event> &E) -> void {
      assert(E.size() == 1);

      for (size_t Idx = 0; Idx < TestAcc.get_count(); ++Idx)
        assert(Idx == TestAcc[Idx]);

      bool Expected = false;
      bool Desired = true;
      assert(Ctx.Flag.compare_exchange_strong(Expected, Desired));

      // let's employ some locking here
      {
        std::lock_guard<std::mutex> Lock(Ctx.Mutex);
        Ctx.CV.notify_all();
      }
    };

    // The Callback is run on Queue-internal thread-pool or in backend's thread
    // if thread pool size is explicitly set to 0
    CGH.host_task(Callback);
  });
}

void ThreadC1Fn(Context &Ctx) {
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
  .when_complete([&Ctx] (const S::event &E) {
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

void testA() {
  auto EH = [] (S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

  // optional
//  Queue.set_host_task_thread_pool_size(4);

  Context Ctx{{false}, Queue, "", {10}};

  // 0. setup: thread 1 T1: exec smth; thread 2 T2: waits; init flag F = false
  std::thread Thread1(ThreadA1Fn, std::reference_wrapper<Context>(Ctx));
  std::thread Thread2(Thread2Fn, std::reference_wrapper<Context>(Ctx));

  Thread1.join();
  Thread2.join();

  std::cout << "Msg = " << Ctx.Message << std::endl;

  assert(Ctx.Flag.load());
  assert(Ctx.Message == "Hello, world");
}

void testB() {
  auto EH = [] (S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

  // optional
//  Queue.set_host_task_thread_pool_size(4);

  Context Ctx{{false}, Queue, "", {10}};

  // 0. setup: thread 1 T1: exec smth; thread 2 T2: waits; init flag F = false
  std::thread Thread1(ThreadB1Fn, std::reference_wrapper<Context>(Ctx));
  std::thread Thread2(Thread2Fn, std::reference_wrapper<Context>(Ctx));

  Thread1.join();
  Thread2.join();

  assert(Ctx.Flag.load());
  assert(Ctx.Message == "Hello, world");
}

void testC() {
  auto EH = [] (S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

  // optional
  Queue.set_host_task_thread_pool_size(4);

  Context Ctx{{false}, Queue, "", {10}};

  // 0. setup: thread 1 T1: exec smth; thread 2 T2: waits; init flag F = false
  std::thread Thread1(ThreadC1Fn, std::reference_wrapper<Context>(Ctx));
  std::thread Thread2(Thread2Fn, std::reference_wrapper<Context>(Ctx));

  Thread1.join();
  Thread2.join();

  assert(Ctx.Flag.load());
  assert(Ctx.Message == "Hello, world");
}

int main(void) {
  testA();
//  testB();
//  testC();

  return 0;
}
