// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %threads_lib -lOpenCL
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

using namespace cl::sycl;
using namespace cl::sycl::access;

static constexpr size_t BUFFER_SIZE = 1024;

// Check that a single host-task with a buffer will work
void test1() {
  buffer<int, 1> Buffer{BUFFER_SIZE};

  queue Q;

  Q.submit([&](handler &CGH) {
    auto Acc = Buffer.get_access<mode::write>(CGH);
    CGH.codeplay_host_task([=] {
      // A no-op
    });
  });
}

void test2() {
  buffer<int, 1> Buffer1{BUFFER_SIZE};
  buffer<int, 1> Buffer2{BUFFER_SIZE};

  queue Q;

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
}

int main() {
  test1();
  test2();
  return 0;
}
