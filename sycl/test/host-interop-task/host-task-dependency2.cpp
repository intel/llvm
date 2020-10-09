// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUN: %CPU_RUN_PLACEHOLDER %t.out 10
// RUN: %GPU_RUN_PLACEHOLDER %t.out 10
// RUN: %ACC_RUN_PLACEHOLDER %t.out 10

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;
using namespace cl::sycl::access;

static constexpr size_t BUFFER_SIZE = 1024;

static auto EH = [](exception_list EL) {
  for (const std::exception_ptr &E : EL) {
    throw E;
  }
};

// Host-task depending on another host-task via handler::depends_on() only
// should not hang
void test(size_t Count) {
  queue Q(EH);

  static constexpr size_t BufferSize = 10 * 1024;

  buffer<int, 1> B0{range<1>{BufferSize}};
  buffer<int, 1> B1{range<1>{BufferSize}};
  buffer<int, 1> B2{range<1>{BufferSize}};
  buffer<int, 1> B3{range<1>{BufferSize}};
  buffer<int, 1> B4{range<1>{BufferSize}};
  buffer<int, 1> B5{range<1>{BufferSize}};

  for (size_t Idx = 1; Idx <= Count; ++Idx) {
    // This host task should be submitted without hesitation
    event E1 = Q.submit([&](handler &CGH) {
      std::cout << "Submit 1" << std::endl;

      auto Acc0 = B0.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc1 = B1.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc2 = B2.get_access<mode::read_write, target::host_buffer>(CGH);

      CGH.codeplay_host_task([=] {
        Acc0[0] = 1 * Idx;
        Acc1[0] = 2 * Idx;
        Acc2[0] = 3 * Idx;
      });
    });

    // This host task is going to depend on blocked empty node of the first
    // host-task (via buffer #2). Still this one should be enqueued.
    event E2 = Q.submit([&](handler &CGH) {
      std::cout << "Submit 2" << std::endl;

      auto Acc2 = B2.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc3 = B3.get_access<mode::read_write, target::host_buffer>(CGH);

      CGH.codeplay_host_task([=] {
        Acc2[1] = 1 * Idx;
        Acc3[1] = 2 * Idx;
      });
    });

    // This host-task only depends on the second host-task via
    // handler::depends_on(). This one should not hang and should be eexecuted
    // after host-task #2.
    event E3 = Q.submit([&](handler &CGH) {
      CGH.depends_on(E2);

      std::cout << "Submit 3" << std::endl;

      auto Acc4 = B4.get_access<mode::read_write, target::host_buffer>(CGH);
      auto Acc5 = B5.get_access<mode::read_write, target::host_buffer>(CGH);

      CGH.codeplay_host_task([=] {
        Acc4[2] = 1 * Idx;
        Acc5[2] = 2 * Idx;
      });
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
