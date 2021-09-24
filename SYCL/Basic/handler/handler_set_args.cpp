// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Memory access fault on AMD
// XFAIL: hip_amd
//==--------------- handler_set_args.cpp -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>

constexpr bool UseOffset = true;
constexpr bool NoOffset = false;
const cl::sycl::range<1> Range = 1;

using AccessorT = cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                                     cl::sycl::access::target::global_buffer>;

struct SingleTaskFunctor {
  SingleTaskFunctor(AccessorT acc) : MAcc(acc) {}

  void operator()() const { MAcc[0] = 10; }

  AccessorT MAcc;
};

template <bool useOffset> struct ParallelForRangeIdFunctor {
  ParallelForRangeIdFunctor(AccessorT acc) : MAcc(acc) {}

  void operator()(cl::sycl::id<1> id) const { MAcc[0] = 10; }

  AccessorT MAcc;
};

template <bool useOffset> struct ParallelForRangeItemFunctor {
  ParallelForRangeItemFunctor(AccessorT acc) : MAcc(acc) {}

  void operator()(cl::sycl::item<1> item) const { MAcc[0] = 10; }

  AccessorT MAcc;
};

struct ParallelForNdRangeFunctor {
  ParallelForNdRangeFunctor(AccessorT acc) : MAcc(acc) {}

  void operator()(cl::sycl::nd_item<1> ndItem) const { MAcc[0] = 10; }

  AccessorT MAcc;
};

template <class kernel_name>
cl::sycl::kernel getPrebuiltKernel(cl::sycl::queue &queue) {
  cl::sycl::program program(queue.get_context());
  program.build_with_kernel_type<kernel_name>();
  return program.get_kernel<kernel_name>();
}

template <class kernel_wrapper>
void checkApiCall(cl::sycl::queue &queue, kernel_wrapper &&kernelWrapper) {
  int result = 0;
  {
    auto buf = cl::sycl::buffer<int, 1>(&result, Range);
    queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      kernelWrapper(cgh, acc);
    });
  }
  assert(result == 10);
}

int main() {
  cl::sycl::queue Queue;
  const cl::sycl::id<1> Offset(0);
  const cl::sycl::nd_range<1> NdRange(Range, Range);

  checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
    cgh.single_task(SingleTaskFunctor(acc));
  });

  checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
    cgh.parallel_for(Range, ParallelForRangeIdFunctor<NoOffset>(acc));
  });

  checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
    cgh.parallel_for(Range, Offset, ParallelForRangeIdFunctor<UseOffset>(acc));
  });

  checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
    cgh.parallel_for(Range, ParallelForRangeItemFunctor<NoOffset>(acc));
  });

  checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
    cgh.parallel_for(Range, Offset,
                     ParallelForRangeItemFunctor<UseOffset>(acc));
  });

  checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
    cgh.parallel_for(NdRange, ParallelForNdRangeFunctor(acc));
  });

  {
    auto preBuiltKernel = getPrebuiltKernel<SingleTaskFunctor>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.single_task(preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel =
        getPrebuiltKernel<ParallelForRangeIdFunctor<NoOffset>>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for(Range, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel =
        getPrebuiltKernel<ParallelForRangeIdFunctor<UseOffset>>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for(Range, Offset, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel =
        getPrebuiltKernel<ParallelForRangeItemFunctor<NoOffset>>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for(Range, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel =
        getPrebuiltKernel<ParallelForRangeItemFunctor<UseOffset>>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for(Range, Offset, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel = getPrebuiltKernel<ParallelForNdRangeFunctor>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for(NdRange, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel = getPrebuiltKernel<SingleTaskFunctor>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.single_task<class OtherKernelName1>(preBuiltKernel,
                                              [=]() { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel =
        getPrebuiltKernel<ParallelForRangeIdFunctor<NoOffset>>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class OtherKernelName2>(
          preBuiltKernel, Range, [=](cl::sycl::id<1> id) { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel =
        getPrebuiltKernel<ParallelForRangeIdFunctor<UseOffset>>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class OtherKernelName3>(
          preBuiltKernel, Range, Offset,
          [=](cl::sycl::id<1> id) { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel =
        getPrebuiltKernel<ParallelForRangeItemFunctor<NoOffset>>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class OtherKernelName4>(
          preBuiltKernel, Range, [=](cl::sycl::item<1> item) { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel =
        getPrebuiltKernel<ParallelForRangeItemFunctor<UseOffset>>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class OtherKernelName5>(
          preBuiltKernel, Range, Offset,
          [=](cl::sycl::item<1> item) { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel = getPrebuiltKernel<ParallelForNdRangeFunctor>(Queue);

    checkApiCall(Queue, [&](cl::sycl::handler &cgh, AccessorT acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class OtherKernelName6>(
          preBuiltKernel, NdRange,
          [=](cl::sycl::nd_item<1> ndItem) { acc[0] = 10; });
    });
  }

  return 0;
}
