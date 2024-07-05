// RUN: %{build} -o %t.out

// RUN: %{run} %t.out

// RUN: %{run} %t.out 10

#include <chrono>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/detail/host_task_impl.hpp>
#include <thread>

using namespace sycl;
using namespace sycl::access;

static constexpr size_t BUFFER_SIZE = 1024;

static auto EH = [](exception_list EL) {
  for (const std::exception_ptr &E : EL) {
    throw E;
  }
};

// Host-task depending on another host-task via handler::depends_on() only
// should not hang. A bit more complicated case with kernels depending on
// host-task being involved.
void test(size_t Count) {
  queue Q(EH);

  static constexpr size_t BufferSize = 10 * 1024;

  buffer<int, 1> B0{range<1>{BufferSize}};
  buffer<int, 1> B1{range<1>{BufferSize}};
  buffer<int, 1> B2{range<1>{BufferSize}};
  buffer<int, 1> B3{range<1>{BufferSize}};
  buffer<int, 1> B4{range<1>{BufferSize}};
  buffer<int, 1> B5{range<1>{BufferSize}};

  using namespace std::chrono_literals;
  constexpr auto SleepFor = 1s;

  for (size_t Idx = 1; Idx <= Count; ++Idx) {
    // This host task should be submitted without hesitation
    Q.submit([&](handler &CGH) {
      std::cout << "Submit HT-1" << std::endl;

      auto Acc0 = B0.get_host_access();

      CGH.host_task([=] {
        std::this_thread::sleep_for(SleepFor);
        Acc0[0] = 1 * Idx;
      });
    });

    Q.submit([&](handler &CGH) {
      std::cout << "Submit Kernel-1" << std::endl;

      auto Acc0 = B0.get_access<mode::read_write>(CGH);

      CGH.single_task<class Test5_Kernel1>([=] { Acc0[1] = 1 * Idx; });
    });

    Q.submit([&](handler &CGH) {
      std::cout << "Submit Kernel-2" << std::endl;

      auto Acc1 = B1.get_access<mode::read_write>(CGH);

      CGH.single_task<class Test5_Kernel2>([=] { Acc1[2] = 1 * Idx; });
    });

    Q.submit([&](handler &CGH) {
      std::cout << "Submit HT-2" << std::endl;

      auto Acc2 = B2.get_host_access();

      CGH.host_task([=] {
        std::this_thread::sleep_for(SleepFor);
        Acc2[3] = 1 * Idx;
      });
    });

    // This host task is going to depend on blocked empty node of the second
    // host-task (via buffer #0). Still this one should be enqueued.
    event EHT3 = Q.submit([&](handler &CGH) {
      std::cout << "Submit HT-3" << std::endl;

      auto Acc0 = B0.get_host_access();
      auto Acc1 = B1.get_host_access();
      auto Acc2 = B2.get_host_access();

      CGH.host_task([=] {
        std::this_thread::sleep_for(SleepFor);
        Acc0[4] = 1 * Idx;
        Acc1[4] = 2 * Idx;
        Acc2[4] = 3 * Idx;
      });
    });

    // This host-task only depends on the third host-task via
    // handler::depends_on(). This one should not hang and should be executed
    // after host-task #3.
    Q.submit([&](handler &CGH) {
      std::cout << "Submit HT-4" << std::endl;

      CGH.depends_on(EHT3);

      auto Acc5 = B5.get_host_access();

      CGH.host_task([=] { Acc5[5] = 1 * Idx; });
    });
  }

  Q.wait_and_throw();
}

int main(int Argc, const char *Argv[]) {
  size_t Count = 1;
  if (Argc > 1)
    Count = std::stoi(Argv[1]);

  test(Count);
  return 0;
}
