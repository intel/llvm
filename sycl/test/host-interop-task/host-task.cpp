// RUN: %clangxx -std=c++14 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out 1
// RUN: %GPU_RUN_PLACEHOLDER %t.out 1
// RUN: %ACC_RUN_PLACEHOLDER %t.out 1

// RUN: %CPU_RUN_PLACEHOLDER %t.out 2
// RUN: %GPU_RUN_PLACEHOLDER %t.out 2
// RUN: %ACC_RUN_PLACEHOLDER %t.out 2

// RUN: %CPU_RUN_PLACEHOLDER %t.out 3
// RUN: %GPU_RUN_PLACEHOLDER %t.out 3
// RUN: %ACC_RUN_PLACEHOLDER %t.out 3

// RUNx: %CPU_RUN_PLACEHOLDER %t.out 4
// RUNx: %GPU_RUN_PLACEHOLDER %t.out 4
// RUNx: %ACC_RUN_PLACEHOLDER %t.out 4

#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace cl::sycl;
using namespace cl::sycl::access;

static constexpr size_t BUFFER_SIZE = 1024;

static auto EH = [](exception_list EL) {
  for (const std::exception_ptr &E : EL) {
    throw E;
  }
};

// Check that a single host-task with a buffer will work
void test1() {
  buffer<int, 1> Buffer{BUFFER_SIZE};

  queue Q(EH);

  Q.submit([&](handler &CGH) {
    auto Acc = Buffer.get_access<mode::write>(CGH);
    CGH.codeplay_host_task([=] {
      // A no-op
    });
  });

  Q.wait_and_throw();
}

// Check that a host task after the kernel (deps via buffer) will work
void test2() {
  buffer<int, 1> Buffer1{BUFFER_SIZE};
  buffer<int, 1> Buffer2{BUFFER_SIZE};

  queue Q(EH);

  Q.submit([&](handler &CGH) {
    auto Acc = Buffer1.template get_access<mode::write>(CGH);

    auto Kernel = [=](item<1> Id) { Acc[Id] = 123; };
    CGH.parallel_for<class Test6Init>(Acc.get_count(), Kernel);
  });

  Q.submit([&](handler &CGH) {
    auto AccSrc = Buffer1.template get_access<mode::read>(CGH);
    auto AccDst = Buffer2.template get_access<mode::write>(CGH);

    CGH.codeplay_host_task([=] {
      for (size_t Idx = 0; Idx < AccDst.get_count(); ++Idx)
        AccDst[Idx] = AccSrc[Idx];
    });
  });

  {
    auto Acc = Buffer2.get_access<mode::read>();

    for (size_t Idx = 0; Idx < Acc.get_count(); ++Idx) {
      std::cout << "Second buffer [" << Idx << "] = " << Acc[Idx] << std::endl;
      assert(Acc[Idx] == 123);
    }
  }

  Q.wait_and_throw();
}

// Host-task depending on another host-task via both buffers and
// handler::depends_on() should not hang
void test3() {
  queue Q(EH);

  static constexpr size_t BufferSize = 10 * 1024;

  buffer<int, 1> B0{range<1>{BufferSize}};
  buffer<int, 1> B1{range<1>{BufferSize}};
  buffer<int, 1> B2{range<1>{BufferSize}};
  buffer<int, 1> B3{range<1>{BufferSize}};
  buffer<int, 1> B4{range<1>{BufferSize}};
  buffer<int, 1> B5{range<1>{BufferSize}};
  buffer<int, 1> B6{range<1>{BufferSize}};
  buffer<int, 1> B7{range<1>{BufferSize}};
  buffer<int, 1> B8{range<1>{BufferSize}};
  buffer<int, 1> B9{range<1>{BufferSize}};

  std::vector<event> Deps;

  using namespace std::chrono_literals;
  static constexpr size_t Count = 10;

  auto Start = std::chrono::steady_clock::now();
  for (size_t Idx = 0; Idx < Count; ++Idx) {
    event E = Q.submit([&](handler &CGH) {
      CGH.depends_on(Deps);

      std::cout << "Submit: " << Idx << std::endl;

      auto Acc0 = B0.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc1 = B1.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc2 = B2.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc3 = B3.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc4 = B4.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc5 = B5.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc6 = B6.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc7 = B7.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc8 = B8.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc9 = B9.get_access<mode::read_write, target::host_buffer>(CGH);

      CGH.codeplay_host_task([=] {
        uint64_t X = 0;

        X ^= reinterpret_cast<uint64_t>(&Acc0[Idx + 0]);
        X ^= reinterpret_cast<uint64_t>(&Acc1[Idx + 1]);
        X ^= reinterpret_cast<uint64_t>(&Acc2[Idx + 2]);
        X ^= reinterpret_cast<uint64_t>(&Acc3[Idx + 3]);
        X ^= reinterpret_cast<uint64_t>(&Acc4[Idx + 4]);
        X ^= reinterpret_cast<uint64_t>(&Acc5[Idx + 5]);
        X ^= reinterpret_cast<uint64_t>(&Acc6[Idx + 6]);
        X ^= reinterpret_cast<uint64_t>(&Acc7[Idx + 7]);
        X ^= reinterpret_cast<uint64_t>(&Acc8[Idx + 8]);
        X ^= reinterpret_cast<uint64_t>(&Acc9[Idx + 9]);
      });
    });

    Deps = {E};
  }

  auto Start = std::chrono::steady_clock::now();
  Q.wait_and_throw();
  auto End = std::chrono::steady_clock::now();

  constexpr auto Threshold = 2s;

  assert(End - Start < Threshold && "Host tasks were waiting for too long");
}

// Host-task depending on another host-task via handler::depends_on() only
// should not hang
void test4() {
  queue Q(EH);

  static constexpr size_t BufferSize = 10 * 1024;

  buffer<int, 1> B0{range<1>{BufferSize}};
  buffer<int, 1> B1{range<1>{BufferSize}};
  buffer<int, 1> B2{range<1>{BufferSize}};
  buffer<int, 1> B3{range<1>{BufferSize}};
  buffer<int, 1> B4{range<1>{BufferSize}};
  buffer<int, 1> B5{range<1>{BufferSize}};

  // This host task should be submitted without hesitation
  event E1 = Q.submit([&](handler &CGH) {
    std::cout << "Submit 1" << std::endl;

    auto Acc0 = B0.get_access<mode::read_write, target::host_buffer>(CGH);
    auto Acc1 = B1.get_access<mode::read_write, target::host_buffer>(CGH);
    auto Acc2 = B2.get_access<mode::read_write, target::host_buffer>(CGH);

    CGH.codeplay_host_task([=] {
      Acc0[0] = 1;
      Acc1[0] = 2;
      Acc2[0] = 3;
    });
  });

  // This host task is going to depend on blocked empty node of the first
  // host-task (via buffer #2). Still this one should be enqueued.
  event E2 = Q.submit([&](handler &CGH) {
    std::cout << "Submit 2" << std::endl;

    auto Acc2 = B2.get_access<mode::read_write, target::host_buffer>(CGH);
    auto Acc3 = B3.get_access<mode::read_write, target::host_buffer>(CGH);

    CGH.codeplay_host_task([=] {
      Acc2[1] = 1;
      Acc3[1] = 2;
    });
  });

  // This host-task only depends on the second host-task via
  // handler::depends_on(). This one should not hang and should be enqueued
  // after host-task #2.
  event E3 = Q.submit([&](handler &CGH) {
    CGH.depends_on(E2);

    std::cout << "Submit 3" << std::endl;

    auto Acc4 = B4.get_access<mode::read_write, target::host_buffer>(CGH);
    auto Acc5 = B5.get_access<mode::read_write, target::host_buffer>(CGH);

    CGH.codeplay_host_task([=] {
      Acc4[2] = 1;
      Acc5[2] = 2;
    });
  });

  Q.wait_and_throw();
}

int main(int Argc, const char *Argv[]) {
  if (Argc < 2)
    return 1;

  int TestIdx = std::stoi(Argv[1]);

  switch (TestIdx) {
  case 1:
    test1();
    break;
  case 2:
    test2();
    break;
  case 3:
    test3();
    break;
  case 4:
    test4();
    break;
  default:
    return 1;
  }

  return 0;
}
