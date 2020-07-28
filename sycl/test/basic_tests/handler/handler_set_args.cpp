// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==--------------- handler_set_args.cpp -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>

struct use_offset {
  static const int no = 0;
  static const int yes = 1;
};

using accessor_t =
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer>;

struct single_task_functor {
  single_task_functor(accessor_t acc) : acc(acc) {}

  void operator()() { acc[0] = 10; }

  accessor_t acc;
};

struct single_task_new_functor {
  single_task_new_functor(accessor_t acc) : acc(acc) {}

  void operator()() { acc[0] = 10; }

  accessor_t acc;
};

template <int useOffset> struct parallel_for_range_id_functor {
  parallel_for_range_id_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::id<1> id) { acc[0] = 10; }

  accessor_t acc;
};

template <int useOffset> struct parallel_for_range_item_functor {
  parallel_for_range_item_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::item<1> item) { acc[0] = 10; }

  accessor_t acc;
};

struct parallel_for_nd_range_functor {
  parallel_for_nd_range_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::nd_item<1> ndItem) { acc[0] = 10; }

  accessor_t acc;
};

template <class kernel_name>
cl::sycl::kernel get_prebuilt_kernel(cl::sycl::queue &queue) {
  cl::sycl::program program(queue.get_context());
  program.build_with_kernel_type<kernel_name>();
  return program.get_kernel<kernel_name>();
}

const cl::sycl::range<1> range = 1;

template <class kernel_wrapper>
void check_api_call(cl::sycl::queue &queue, kernel_wrapper &&kernelWrapper) {
  int result = 0;
  {
    auto buf = cl::sycl::buffer<int, 1>(&result, range);
    queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      kernelWrapper(cgh, acc);
    });
  }
  assert(result == 10);
}

int main() {
  cl::sycl::queue queue;
  const cl::sycl::id<1> offset(0);

  const cl::sycl::nd_range<1> ndRange(range, range);

  check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
    cgh.single_task(single_task_functor(acc));
  });

  check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
    cgh.parallel_for(range, parallel_for_range_id_functor<use_offset::no>(acc));
  });

  check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
    cgh.parallel_for(range, offset,
                     parallel_for_range_id_functor<use_offset::yes>(acc));
  });

  check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
    cgh.parallel_for(range,
                     parallel_for_range_item_functor<use_offset::no>(acc));
  });

  check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
    cgh.parallel_for(range, offset,
                     parallel_for_range_item_functor<use_offset::yes>(acc));
  });

  check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
    cgh.parallel_for(ndRange, parallel_for_nd_range_functor(acc));
  });

  {
    auto preBuiltKernel = get_prebuilt_kernel<single_task_functor>(queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.single_task(preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_range_id_functor<use_offset::no>>(
            queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for(range, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_range_id_functor<use_offset::yes>>(
            queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for(range, offset, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_range_item_functor<use_offset::no>>(
            queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for(range, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_range_item_functor<use_offset::yes>>(
            queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for(range, offset, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_nd_range_functor>(queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for(ndRange, preBuiltKernel);
    });
  }

  {
    auto preBuiltKernel = get_prebuilt_kernel<single_task_functor>(queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.single_task<class other_kernel_name1>(preBuiltKernel,
                                                [=]() { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_range_id_functor<use_offset::no>>(
            queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class other_kernel_name2>(
          preBuiltKernel, range, [=](cl::sycl::id<1> id) { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_range_id_functor<use_offset::yes>>(
            queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class other_kernel_name3>(
          preBuiltKernel, range, offset,
          [=](cl::sycl::id<1> id) { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_range_item_functor<use_offset::no>>(
            queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class other_kernel_name4>(
          preBuiltKernel, range, [=](cl::sycl::item<1> item) { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_range_item_functor<use_offset::yes>>(
            queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class other_kernel_name5>(
          preBuiltKernel, range, offset,
          [=](cl::sycl::item<1> item) { acc[0] = 10; });
    });
  }

  {
    auto preBuiltKernel =
        get_prebuilt_kernel<parallel_for_nd_range_functor>(queue);

    check_api_call(queue, [&](cl::sycl::handler &cgh, accessor_t acc) {
      cgh.set_args(acc);
      cgh.parallel_for<class other_kernel_name6>(
          preBuiltKernel, ndRange,
          [=](cl::sycl::nd_item<1> ndItem) { acc[0] = 10; });
    });
  }

  return 0;
}
