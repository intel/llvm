// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %threads_lib
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 %t.out 2>&1 %ACC_CHECK_PLACEHOLDER

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

#include <CL/sycl.hpp>

namespace S = cl::sycl;

struct Context {
  std::atomic_bool Flag;
  S::queue &Queue;
  S::buffer<int, 1> Buf1;
  S::buffer<int, 1> Buf2;
  S::buffer<int, 1> Buf3;
  std::mutex Mutex;
  std::condition_variable CV;
};

void Thread1Fn(Context *Ctx) {
  // 0. initialize resulting buffer with apriori wrong result
  {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::host_buffer>
        Acc(Ctx->Buf1);

    for (size_t Idx = 0; Idx < Acc.get_count(); ++Idx)
      Acc[Idx] = -1;
  }

  {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::host_buffer>
        Acc(Ctx->Buf2);

    for (size_t Idx = 0; Idx < Acc.get_count(); ++Idx)
      Acc[Idx] = -2;
  }

  {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::host_buffer>
        Acc(Ctx->Buf3);

    for (size_t Idx = 0; Idx < Acc.get_count(); ++Idx)
      Acc[Idx] = -3;
  }

  // 1. submit task writing to buffer 1
  Ctx->Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::global_buffer>
        GeneratorAcc(Ctx->Buf1, CGH);

    auto GeneratorKernel = [GeneratorAcc] {
      for (size_t Idx = 0; Idx < GeneratorAcc.get_count(); ++Idx)
        GeneratorAcc[Idx] = Idx;
    };

    CGH.single_task<class GeneratorTask>(GeneratorKernel);
  });

  // 2. submit host task writing from buf 1 to buf 2
  auto HostTaskEvent = Ctx->Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::read,
                S::access::target::host_buffer>
        CopierSrcAcc(Ctx->Buf1, CGH);
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::host_buffer>
        CopierDstAcc(Ctx->Buf2, CGH);

    auto CopierHostTask = [CopierSrcAcc, CopierDstAcc, &Ctx] {
      for (size_t Idx = 0; Idx < CopierDstAcc.get_count(); ++Idx)
        CopierDstAcc[Idx] = CopierSrcAcc[Idx];

      bool Expected = false;
      bool Desired = true;
      assert(Ctx->Flag.compare_exchange_strong(Expected, Desired));

      {
        std::lock_guard<std::mutex> Lock(Ctx->Mutex);
        Ctx->CV.notify_all();
      }
    };

    CGH.codeplay_host_task(CopierHostTask);
  });

  // 3. submit simple task to move data between two buffers
  Ctx->Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::read,
                S::access::target::global_buffer>
        SrcAcc(Ctx->Buf2, CGH);
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::global_buffer>
        DstAcc(Ctx->Buf3, CGH);

    CGH.depends_on(HostTaskEvent);

    auto CopierKernel = [SrcAcc, DstAcc] {
      for (size_t Idx = 0; Idx < DstAcc.get_count(); ++Idx)
        DstAcc[Idx] = SrcAcc[Idx];
    };

    CGH.single_task<class CopierTask>(CopierKernel);
  });

  // 4. check data in buffer #3
  {
    S::accessor<int, 1, S::access::mode::read,
                S::access::target::host_buffer>
        Acc(Ctx->Buf3);

    bool Failure = false;

    for (size_t Idx = 0; Idx < Acc.get_count(); ++Idx) {
      fprintf(stderr, "Third buffer [%3zu] = %i\n", Idx, Acc[Idx]);

      Failure |= (Acc[Idx] != Idx);
    }

    assert(!Failure && "Invalid data in third buffer");
  }
}

void Thread2Fn(Context *Ctx) {
  std::unique_lock<std::mutex> Lock(Ctx->Mutex);

  // T2.1. Wait until flag F is set eq true.
  Ctx->CV.wait(Lock, [Ctx] { return Ctx->Flag.load(); });

  assert(Ctx->Flag.load());
}

void test() {
  auto EH = [](S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

  Context Ctx{{false}, Queue, {10}, {10}, {10}, {}, {}};

  // 0. setup: thread 1 T1: exec smth; thread 2 T2: waits; init flag F = false
  auto A1 = std::async(std::launch::async, Thread1Fn, &Ctx);
  auto A2 = std::async(std::launch::async, Thread2Fn, &Ctx);

  A1.get();
  A2.get();

  assert(Ctx.Flag.load());

  // 3. check via host accessor that buf 2 contains valid data
  {
    S::accessor<int, 1, S::access::mode::read,
                S::access::target::host_buffer>
        ResultAcc(Ctx.Buf2);

    bool Failure = false;
    for (size_t Idx = 0; Idx < ResultAcc.get_count(); ++Idx) {
      fprintf(stderr, "Second buffer [%3zu] = %i\n", Idx, ResultAcc[Idx]);

      Failure |= (ResultAcc[Idx] != Idx);
    }

    assert(!Failure && "Invalid data in result buffer");
  }
}

int main() {
  test();

  return 0;
}

// launch of GeneratorTask kernel
// CHECK:---> piKernelCreate(
// CHECK: GeneratorTask
// CHECK:---> piEnqueueKernelLaunch(
// prepare for host task
// CHECK:---> piEnqueueMemBufferMap(
// launch of CopierTask kernel
// CHECK:---> piKernelCreate(
// CHECK: CopierTask
// CHECK:---> piEnqueueKernelLaunch(
// TODO need to check for piEventsWait as "wait on dependencies of host task".
// At the same time this piEventsWait may occur anywhere after
// piEnqueueMemBufferMap ("prepare for host task").
