// RUN: %{build} -o %t.out -lcuda
// RUN: %{run} %t.out
// REQUIRES: cuda

#include <iostream>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/detail/host_task_impl.hpp>

#include <cuda.h>

using namespace sycl;
using namespace sycl::access;

static constexpr size_t BUFFER_SIZE = 1024;

template <typename T> class Modifier;

template <typename T> class Init;

template <typename BufferT, typename ValueT>
void checkBufferValues(BufferT Buffer, ValueT Value) {
  auto Acc = Buffer.get_host_access();
  for (size_t Idx = 0; Idx < Acc.size(); ++Idx) {
    if (Acc[Idx] != Value) {
      std::cerr << "buffer[" << Idx << "] = " << Acc[Idx]
                << ", expected val = " << Value << '\n';
      exit(1);
    }
  }
}

template <typename DataT>
void copy(buffer<DataT, 1> &Src, buffer<DataT, 1> &Dst, queue &Q) {
  Q.submit([&](handler &CGH) {
    auto SrcA = Src.template get_access<mode::read>(CGH);
    auto DstA = Dst.template get_access<mode::write>(CGH);

    auto Func = [=](interop_handle IH) {
      auto Stream = IH.get_native_queue<backend::ext_oneapi_cuda>();
      auto SrcMem = IH.get_native_mem<backend::ext_oneapi_cuda>(SrcA);
      auto DstMem = IH.get_native_mem<backend::ext_oneapi_cuda>(DstA);

      if (cuMemcpyAsync(DstMem, SrcMem, sizeof(DataT) * SrcA.size(), Stream) !=
          CUDA_SUCCESS) {
        throw;
      }

      if (cuStreamSynchronize(Stream) != CUDA_SUCCESS) {
        throw;
      }

      if (Q.get_backend() != IH.get_backend())
        throw;
    };
    CGH.host_task(Func);
  });
}

template <typename DataT> void modify(buffer<DataT, 1> &B, queue &Q) {
  Q.submit([&](handler &CGH) {
    auto Acc = B.template get_access<mode::read_write>(CGH);

    auto Kernel = [=](item<1> Id) { Acc[Id] += 1; };

    CGH.parallel_for<Modifier<DataT>>(Acc.size(), Kernel);
  });
}

template <typename DataT, DataT B1Init, DataT B2Init>
void init(buffer<DataT, 1> &B1, buffer<DataT, 1> &B2, queue &Q) {
  Q.submit([&](handler &CGH) {
    auto Acc1 = B1.template get_access<mode::write>(CGH);
    auto Acc2 = B2.template get_access<mode::write>(CGH);

    CGH.parallel_for<Init<DataT>>(BUFFER_SIZE, [=](item<1> Id) {
      Acc1[Id] = B1Init;
      Acc2[Id] = B2Init;
    });
  });
}

// Check that a single host-interop-task with a buffer will work.
void test_ht_buffer(queue &Q) {
  buffer<int, 1> Buffer{BUFFER_SIZE};

  Q.submit([&](handler &CGH) {
    auto Acc = Buffer.get_access<mode::write>(CGH);
    auto Func = [=](interop_handle IH) { /*A no-op */ };
    CGH.host_task(Func);
  });
}

// A test that uses CUDA interop to copy data from buffer A to buffer B, by
// getting CUDA ptrs and calling the cuMemcpyWithAsync. Then run a SYCL
// kernel that modifies the data in place for B, e.g. increment one, then copy
// back to buffer A. Run it on a loop, to ensure the dependencies and the
// reference counting of the objects is not leaked.
void test_ht_kernel_dependencies(queue &Q) {
  static constexpr int COUNT = 4;
  buffer<int, 1> Buffer1{BUFFER_SIZE};
  buffer<int, 1> Buffer2{BUFFER_SIZE};

  // Init the buffer with a'priori invalid data.
  init<int, -1, -2>(Buffer1, Buffer2, Q);

  // Repeat a couple of times.
  for (size_t Idx = 0; Idx < COUNT; ++Idx) {
    copy(Buffer1, Buffer2, Q);
    modify(Buffer2, Q);
    copy(Buffer2, Buffer1, Q);
  }

  checkBufferValues(Buffer1, COUNT - 1);
  checkBufferValues(Buffer2, COUNT - 1);
}

void tests(queue &Q) {
  test_ht_buffer(Q);
  test_ht_kernel_dependencies(Q);
}

int main() {
  queue Q;
  tests(Q);
  std::cout << "Test PASSED" << std::endl;
  return 0;
}
