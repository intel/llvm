// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %threads_lib %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: level_zero, cuda
// REQUIRES: opencl, opencl_icd

#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>
#include <CL/sycl/detail/cl.h>

using namespace cl::sycl;
using namespace cl::sycl::access;

static constexpr size_t BUFFER_SIZE = 1024;

template <typename T> class Modifier;

template <typename T> class Init;

template <typename BufferT, typename ValueT>
void checkBufferValues(BufferT Buffer, ValueT Value) {
  auto Acc = Buffer.template get_access<mode::read>();
  for (size_t Idx = 0; Idx < Acc.get_count(); ++Idx) {
    if (Acc[Idx] != Value) {
      std::cerr << "buffer[" << Idx << "] = " << Acc[Idx]
                << ", expected val = " << Value << std::endl;
      assert(0 && "Invalid data in the buffer");
    }
  }
}

template <typename DataT>
void copy(buffer<DataT, 1> &Src, buffer<DataT, 1> &Dst, queue &Q) {
  Q.submit([&](handler &CGH) {
    auto SrcA = Src.template get_access<mode::read>(CGH);
    auto DstA = Dst.template get_access<mode::write>(CGH);

    CGH.codeplay_host_task([=](interop_handle IH) {
      auto NativeQ = IH.get_native_queue();
      auto SrcMem = IH.get_native_mem(SrcA);
      auto DstMem = IH.get_native_mem(DstA);
      cl_event Event;

      int RC = clEnqueueCopyBuffer(NativeQ, SrcMem, DstMem, 0, 0,
                                   sizeof(DataT) * SrcA.get_count(), 0, nullptr,
                                   &Event);

      if (RC != CL_SUCCESS)
        throw runtime_error("Can't enqueue buffer copy", RC);

      RC = clWaitForEvents(1, &Event);

      if (RC != CL_SUCCESS)
        throw runtime_error("Can't wait for event on buffer copy", RC);

      if (Q.get_backend() != IH.get_backend())
        throw runtime_error(
            "interop_handle::get_backend() returned a wrong value",
            CL_INVALID_VALUE);
    });
  });
}

template <typename DataT> void modify(buffer<DataT, 1> &B, queue &Q) {
  Q.submit([&](handler &CGH) {
    auto Acc = B.template get_access<mode::read_write>(CGH);

    auto Kernel = [=](item<1> Id) { Acc[Id] += 1; };

    CGH.parallel_for<Modifier<DataT>>(Acc.get_count(), Kernel);
  });
}

template <typename DataT, DataT B1Init, DataT B2Init>
void init(buffer<DataT, 1> &B1, buffer<DataT, 1> &B2, queue &Q) {
  Q.submit([&](handler &CGH) {
    auto Acc1 = B1.template get_access<mode::write>(CGH);
    auto Acc2 = B2.template get_access<mode::write>(CGH);

    CGH.parallel_for<Init<DataT>>(BUFFER_SIZE, [=](item<1> Id) {
      Acc1[Id] = -1;
      Acc2[Id] = -2;
    });
  });
}

// A test that uses OpenCL interop to copy data from buffer A to buffer B, by
// getting cl_mem objects and calling the clEnqueueBufferCopy. Then run a SYCL
// kernel that modifies the data in place for B, e.g. increment one, then copy
// back to buffer A. Run it on a loop, to ensure the dependencies and the
// reference counting of the objects is not leaked.
void test1() {
  static constexpr int COUNT = 4;
  queue Q;
  buffer<int, 1> Buffer1{BUFFER_SIZE};
  buffer<int, 1> Buffer2{BUFFER_SIZE};

  // init the buffer with a'priori invalid data
  init<int, -1, -2>(Buffer1, Buffer2, Q);

  // Repeat a couple of times
  for (size_t Idx = 0; Idx < COUNT; ++Idx) {
    copy(Buffer1, Buffer2, Q);
    modify(Buffer2, Q);
    copy(Buffer2, Buffer1, Q);
  }

  checkBufferValues(Buffer1, COUNT - 1);
  checkBufferValues(Buffer2, COUNT - 1);
}

// Same as above, but performing each command group on a separate SYCL queue
// (on the same or different devices). This ensures the dependency tracking
// works well but also there is no accidental side effects on other queues.
void test2() {
  static constexpr int COUNT = 4;
  buffer<int, 1> Buffer1{BUFFER_SIZE};
  buffer<int, 1> Buffer2{BUFFER_SIZE};

  // init the buffer with a'priori invalid data
  {
    queue Q;
    init<int, -1, -2>(Buffer1, Buffer2, Q);
  }

  // Repeat a couple of times
  for (size_t Idx = 0; Idx < COUNT; ++Idx) {
    queue Q;
    copy(Buffer1, Buffer2, Q);
    modify(Buffer2, Q);
    copy(Buffer2, Buffer1, Q);
  }
  checkBufferValues(Buffer1, COUNT - 1);
  checkBufferValues(Buffer2, COUNT - 1);
}

// Same as above but with queue constructed out of context
void test2_1() {
  static constexpr int COUNT = 4;
  buffer<int, 1> Buffer1{BUFFER_SIZE};
  buffer<int, 1> Buffer2{BUFFER_SIZE};

  auto Device = default_selector().select_device();
  auto Context = context(Device);
  // init the buffer with a'priori invalid data
  {
    queue Q(Context, Device);
    init<int, -1, -2>(Buffer1, Buffer2, Q);
  }

  // Repeat a couple of times
  for (size_t Idx = 0; Idx < COUNT; ++Idx) {
    queue Q(Context, Device);
    copy(Buffer1, Buffer2, Q);
    modify(Buffer2, Q);
    copy(Buffer2, Buffer1, Q);
  }
  checkBufferValues(Buffer1, COUNT - 1);
  checkBufferValues(Buffer2, COUNT - 1);
}

// A test that does a clEnqueueWait inside the interop scope, for an event
// captured outside the command group. The OpenCL event can be set after the
// command group finishes. Must not deadlock according to implementation and
// proposal
void test3() {
  // Want some large buffer for operation to take long
  buffer<int, 1> Buffer{BUFFER_SIZE * 128};

  queue Q;

  event Event = Q.submit([&](handler &CGH) {
    auto Acc1 = Buffer.get_access<mode::write>(CGH);

    CGH.parallel_for<class Init3>(BUFFER_SIZE,
                                  [=](item<1> Id) { Acc1[Id] = 123; });
  });

  Q.submit([&](handler &CGH) {
    CGH.codeplay_host_task([=](interop_handle IH) {
      cl_event Ev = Event.get();

      int RC = clWaitForEvents(1, &Ev);

      if (RC != CL_SUCCESS)
        throw runtime_error("Can't wait for events", RC);
    });
  });
}

// Check that a single host-interop-task with a buffer will work
void test4() {
  buffer<int, 1> Buffer{BUFFER_SIZE};

  queue Q;

  Q.submit([&](handler &CGH) {
    auto Acc = Buffer.get_access<mode::write>(CGH);
    CGH.codeplay_host_task([=](interop_handle IH) {
      // A no-op
    });
  });
}

void test5() {
  buffer<int, 1> Buffer1{BUFFER_SIZE};
  buffer<int, 1> Buffer2{BUFFER_SIZE};

  queue Q;

  Q.submit([&](handler &CGH) {
    auto Acc = Buffer1.template get_access<mode::write>(CGH);

    auto Kernel = [=](item<1> Id) { Acc[Id] = 123; };
    CGH.parallel_for<class Test5Init>(Acc.get_count(), Kernel);
  });

  copy(Buffer1, Buffer2, Q);

  checkBufferValues(Buffer2, static_cast<int>(123));
}

// The test checks that an exception which is thrown from host_task body
// is reported as asynchronous.
void test6() {
  queue Queue([](sycl::exception_list ExceptionList) {
    if (ExceptionList.size() != 1) {
      std::cerr << "Should be one exception in exception list" << std::endl;
      std::abort();
    }
    std::rethrow_exception(*ExceptionList.begin());
  });

  try {
    size_t size = 1;
    buffer<int, 1> Buf{size};
    Queue.submit([&](sycl::handler &CGH) {
      auto acc = Buf.get_access<mode::write, target::host_buffer>(CGH);
      CGH.codeplay_host_task(
          [=](interop_handle IH) { (void)IH.get_native_mem(acc); });
    });
    Queue.wait_and_throw();
    assert(!"Expected exception was not caught");
  } catch (sycl::exception &ExpectedException) {
    assert(std::string(ExpectedException.what())
                   .find("memory object out of accessor for specified target "
                         "is not allowed") != std::string::npos &&
           "Unexpected error was caught!");
  }
}

int main() {
  test1();
  test2();
  test2_1();
  test3();
  test4();
  test5();
  test6();
  std::cout << "Test PASSED" << std::endl;
  return 0;
}
