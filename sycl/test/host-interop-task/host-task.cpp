// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

namespace S = cl::sycl;

void test() {
  auto EH = [] (S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

#define DATA_SIZE 10
  S::buffer<int, 1> Buf1(DATA_SIZE);
  S::buffer<int, 1> Buf2(DATA_SIZE);

  // 0. initialize resulting buffer with apriori wrong result
  {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::host_buffer> Acc(Buf2);

    for (size_t Idx = 0; Idx < Acc.get_count(); ++Idx)
      Acc[Idx] = -1;
  }

  // 1. submit task writing to buffer 1
  Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::global_buffer> GeneratorAcc(Buf1, CGH);

    auto GeneratorKernel = [GeneratorAcc] () {
      for (size_t Idx = 0; Idx < GeneratorAcc.get_count(); ++Idx)
        GeneratorAcc[Idx] = Idx;
    };

    CGH.single_task<class GeneratorTask>(GeneratorKernel);
  });

  // 2. submit host task writing from buf 1 to buf 2
  Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::read,
                S::access::target::host_buffer> CopierSrcAcc(Buf1, CGH);
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::host_buffer> CopierDstAcc(Buf2, CGH);

    auto CopierKernel = [CopierSrcAcc, CopierDstAcc] () {
      for (size_t Idx = 0; Idx < CopierDstAcc.get_count(); ++Idx)
        CopierDstAcc[Idx] = CopierSrcAcc[Idx];
    };

    CGH.codeplay_host_task(CopierKernel);
  });

  // 3. check via host accessor that buf 2 contains valid data
  {
    S::accessor<int, 1, S::access::mode::read,
                S::access::target::host_buffer> ResultAcc(Buf2);

    for (size_t Idx = 0; Idx < ResultAcc.get_count(); ++Idx) {
      assert(ResultAcc[Idx] == Idx && "Invalid data in result buffer");
    }
  }
}

int main() {
  test();

  return 0;
}
