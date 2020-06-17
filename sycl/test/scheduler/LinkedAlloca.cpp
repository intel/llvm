// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %sycl_source_dir %s -o %t.out
// RUN: env SYCL_PRINT_EXECUTION_GRAPH=always %t.out
// RUN: cat graph_7after_addHostAccessor.dot
// RUN: cat graph_7after_addHostAccessor.dot | FileCheck %s

#include <CL/sycl.hpp>

namespace S = cl::sycl;

void test() {
  auto EH = [](S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

  static const size_t BSIZE = 10;
  S::buffer<int, 1> Buf{BSIZE};

  // 0. submit kernel
  Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::global_buffer>
        GeneratorAcc(Buf, CGH);

    auto GeneratorKernel = [GeneratorAcc]() {
      for (size_t Idx = 0; Idx < GeneratorAcc.get_count(); ++Idx)
        GeneratorAcc[Idx] = BSIZE - Idx;
    };

    CGH.single_task<class GeneratorTask0>(GeneratorKernel);
  });
  // 1. create host-accessor
  {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::host_buffer>
        Acc(Buf);

    for (size_t Idx = 0; Idx < Acc.get_count(); ++Idx)
      Acc[Idx] = -1;
  }

  // 2. submit kernel
  Queue.submit([&](S::handler &CGH) {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::global_buffer>
        GeneratorAcc(Buf, CGH);

    auto GeneratorKernel = [GeneratorAcc]() {
      for (size_t Idx = 0; Idx < GeneratorAcc.get_count(); ++Idx)
        GeneratorAcc[Idx] = Idx;
    };

    CGH.single_task<class GeneratorTask>(GeneratorKernel);
  });

  // 3. create host-accessor
  {
    S::accessor<int, 1, S::access::mode::write,
                S::access::target::host_buffer>
        Acc(Buf);

    bool Failure = false;
    for (size_t Idx = 0; Idx < Acc.get_count(); ++Idx) {
      fprintf(stderr, "Buffer [%03zu] = %i\n", Idx, Acc[Idx]);
      Failure |= (Acc[Idx] != Idx);
    }

    assert(!Failure || "Invalid data in buffer");
  }
}

int main(void) {
  test();

  return 0;
}

// CHECK: {{^}}"[[MAP_OP:0x[0-9a-fA-F]+]]"{{.*}} MAP ON
// CHECK: "[[MAP_OP]]" -> "[[MAP_DEP_1:0x[0-9a-fA-F]+]]"
// CHECK: "[[MAP_OP]]" -> "[[MAP_DEP_2:0x[0-9a-fA-F]+]]"
// CHECK: {{^}}"[[MAP_DEP_2]]"{{.*}}\nEXEC CG ON
// CHECK: {{^}}"[[MAP_DEP_1]]"{{.*}}\nALLOCA ON HOST
