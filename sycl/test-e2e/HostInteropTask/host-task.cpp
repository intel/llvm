// RUN: %{build} -o %t.out
// RUN: %{run} %t.out 1

// RUN: %{run} %t.out 2

// RUN: %{run} %t.out 3

#include <chrono>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <vector>

using namespace sycl;
using namespace sycl::access;

static constexpr size_t BUFFER_SIZE = 1024;

static auto EH = [](exception_list EL) {
  for (const std::exception_ptr &E : EL) {
    throw E;
  }
};

template <typename T, bool B> class NameGen;

// Check that a single host-task with a buffer will work
void test1(queue &Q) {
  buffer<int, 1> Buffer{BUFFER_SIZE};

  Q.submit([&](handler &CGH) {
    auto Acc = Buffer.get_access<mode::write>(CGH);
    CGH.host_task([=] { /* A no-op */ });
  });

  Q.wait_and_throw();
}

// Check that a host task after the kernel (deps via buffer) will work
void test2(queue &Q) {
  buffer<int, 1> Buffer1{BUFFER_SIZE};
  buffer<int, 1> Buffer2{BUFFER_SIZE};

  Q.submit([&](handler &CGH) {
    auto Acc = Buffer1.template get_access<mode::write>(CGH);

    auto Kernel = [=](item<1> Id) { Acc[Id] = 123; };
    CGH.parallel_for<NameGen<class Test6Init, true>>(Acc.size(), Kernel);
  });

  Q.submit([&](handler &CGH) {
    auto AccSrc = Buffer1.template get_access<mode::read>(CGH);
    auto AccDst = Buffer2.template get_access<mode::write>(CGH);

    auto Func = [=] {
      for (size_t Idx = 0; Idx < AccDst.size(); ++Idx)
        AccDst[Idx] = AccSrc[Idx];
    };
    CGH.host_task(Func);
  });

  {
    auto Acc = Buffer2.get_host_access();

    for (size_t Idx = 0; Idx < Acc.size(); ++Idx) {
      std::cout << "Second buffer [" << Idx << "] = " << Acc[Idx] << std::endl;
      assert(Acc[Idx] == 123);
    }
  }

  Q.wait_and_throw();
}

// Host-task depending on another host-task via both buffers and
// handler::depends_on() should not hang
void test3(queue &Q) {
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

  static constexpr size_t Count = 10;

  auto Start = std::chrono::steady_clock::now();
  for (size_t Idx = 0; Idx < Count; ++Idx) {
    event E = Q.submit([&](handler &CGH) {
      CGH.depends_on(Deps);

      std::cout << "Submit: " << Idx << std::endl;

      auto Acc0 = B0.get_host_access();
      auto Acc1 = B1.get_host_access();
      auto Acc2 = B2.get_host_access();
      auto Acc3 = B3.get_host_access();
      auto Acc4 = B4.get_host_access();
      auto Acc5 = B5.get_host_access();
      auto Acc6 = B6.get_host_access();
      auto Acc7 = B7.get_host_access();
      auto Acc8 = B8.get_host_access();
      auto Acc9 = B9.get_host_access();

      auto Func = [=] {
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
      };
      CGH.host_task(Func);
    });

    Deps = {E};
  }

  Q.wait_and_throw();
  auto End = std::chrono::steady_clock::now();

  using namespace std::chrono_literals;
  constexpr auto Threshold = 2s;

  assert(End - Start < Threshold && "Host tasks were waiting for too long");
}

int main(int Argc, const char *Argv[]) {
  if (Argc < 2)
    return 1;

  int TestIdx = std::stoi(Argv[1]);

  queue Q(EH);
  switch (TestIdx) {
  case 1:
    test1(Q);
    break;
  case 2:
    test2(Q);
    break;
  case 3:
    test3(Q);
    break;
  default:
    return 1;
  }

  return 0;
}
