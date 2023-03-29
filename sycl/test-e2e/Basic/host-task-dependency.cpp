// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %threads_lib
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 %t.out 2>&1 %ACC_CHECK_PLACEHOLDER
//
// TODO: Behaviour is unstable for level zero on Windows. Enable when fixed.
// TODO: The test is sporadically fails on CUDA. Enable when fixed.
// UNSUPPORTED: (windows && level_zero) || hip_nvidia

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <sycl/sycl.hpp>

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

namespace S = sycl;

template <typename T> class NameGen;

struct Context {
  std::atomic_bool Flag;
  S::queue &Queue;
  S::buffer<int, 1> Buf1;
  S::buffer<int, 1> Buf2;
  S::buffer<int, 1> Buf3;
  std::mutex Mutex;
  std::condition_variable CV;
};

S::event HostTask_CopyBuf1ToBuf2(Context *Ctx) {
  S::event Event = Ctx->Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::read, S::access::target::host_buffer>
        CopierSrcAcc(Ctx->Buf1, CGH);
    S::accessor<int, 1, S::access::mode::write, S::access::target::host_buffer>
        CopierDstAcc(Ctx->Buf2, CGH);

    auto CopierHostTask = [=] {
      for (size_t Idx = 0; Idx < CopierDstAcc.size(); ++Idx)
        CopierDstAcc[Idx] = CopierSrcAcc[Idx];

      bool Expected = false;
      bool Desired = true;
      assert(Ctx->Flag.compare_exchange_strong(Expected, Desired));

      {
        std::lock_guard<std::mutex> Lock(Ctx->Mutex);
        Ctx->CV.notify_all();
      }
    };

    CGH.host_task(CopierHostTask);
  });
  return Event;
}

void Thread1Fn(Context *Ctx) {
  // 0. initialize resulting buffer with apriori wrong result
  {
    S::accessor<int, 1, S::access::mode::write, S::access::target::host_buffer>
        Acc(Ctx->Buf1);

    for (size_t Idx = 0; Idx < Acc.size(); ++Idx)
      Acc[Idx] = -1;
  }

  {
    S::accessor<int, 1, S::access::mode::write, S::access::target::host_buffer>
        Acc(Ctx->Buf2);

    for (size_t Idx = 0; Idx < Acc.size(); ++Idx)
      Acc[Idx] = -2;
  }

  {
    S::accessor<int, 1, S::access::mode::write, S::access::target::host_buffer>
        Acc(Ctx->Buf3);

    for (size_t Idx = 0; Idx < Acc.size(); ++Idx)
      Acc[Idx] = -3;
  }

  // 1. submit task writing to buffer 1
  Ctx->Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::write, S::access::target::device>
        GeneratorAcc(Ctx->Buf1, CGH);

    auto GeneratorKernel = [GeneratorAcc] {
      for (size_t Idx = 0; Idx < GeneratorAcc.size(); ++Idx)
        GeneratorAcc[Idx] = Idx;
    };

    CGH.single_task<NameGen<class Gen>>(GeneratorKernel);
  });

  // 2. submit host task writing from buf 1 to buf 2
  S::event HostTaskEvent = HostTask_CopyBuf1ToBuf2(Ctx);

  // 3. submit simple task to move data between two buffers
  Ctx->Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::read, S::access::target::device>
        SrcAcc(Ctx->Buf2, CGH);
    S::accessor<int, 1, S::access::mode::write, S::access::target::device>
        DstAcc(Ctx->Buf3, CGH);

    CGH.depends_on(HostTaskEvent);

    auto CopierKernel = [SrcAcc, DstAcc] {
      for (size_t Idx = 0; Idx < DstAcc.size(); ++Idx)
        DstAcc[Idx] = SrcAcc[Idx];
    };

    CGH.single_task<NameGen<class Copier>>(CopierKernel);
  });

  // 4. check data in buffer #3
  {
    S::accessor<int, 1, S::access::mode::read, S::access::target::host_buffer>
        Acc(Ctx->Buf3);

    bool Failure = false;

    for (size_t Idx = 0; Idx < Acc.size(); ++Idx) {
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
    S::accessor<int, 1, S::access::mode::read, S::access::target::host_buffer>
        ResultAcc(Ctx.Buf2);

    bool Failure = false;
    for (size_t Idx = 0; Idx < ResultAcc.size(); ++Idx) {
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

// launch of Gen kernel
// CHECK:---> piKernelCreate(
// CHECK: NameGen
// CHECK:---> piEnqueueKernelLaunch(
// prepare for host task
// CHECK:---> piEnqueueMemBuffer{{Map|Read}}(
// launch of Copier kernel
// CHECK:---> piKernelCreate(
// CHECK: Copier
// CHECK:---> piEnqueueKernelLaunch(

// CHECK:Third buffer [  0] = 0
// CHECK:Third buffer [  1] = 1
// CHECK:Third buffer [  2] = 2
// CHECK:Third buffer [  3] = 3
// CHECK:Third buffer [  4] = 4
// CHECK:Third buffer [  5] = 5
// CHECK:Third buffer [  6] = 6
// CHECK:Third buffer [  7] = 7
// CHECK:Third buffer [  8] = 8
// CHECK:Third buffer [  9] = 9
// CHECK:Second buffer [  0] = 0
// CHECK:Second buffer [  1] = 1
// CHECK:Second buffer [  2] = 2
// CHECK:Second buffer [  3] = 3
// CHECK:Second buffer [  4] = 4
// CHECK:Second buffer [  5] = 5
// CHECK:Second buffer [  6] = 6
// CHECK:Second buffer [  7] = 7
// CHECK:Second buffer [  8] = 8
// CHECK:Second buffer [  9] = 9

// TODO need to check for piEventsWait as "wait on dependencies of host task".
// At the same time this piEventsWait may occur anywhere after
// piEnqueueMemBufferMap ("prepare for host task").
