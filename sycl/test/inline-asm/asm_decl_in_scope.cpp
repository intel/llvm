// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out

#include "include/asmcheck.h"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;

constexpr int LIST_SIZE = 1024 * 1024;
using arr_t = std::vector<cl::sycl::cl_int>;

// class is used for kernel name
template <typename T>
class decl_in_scope_kernel;

class ocl_ctx_t {
  cl::sycl::queue deviceQueue;

public:
  ocl_ctx_t(const cl::sycl::device_selector &sel) : deviceQueue(sel) {}

  template <typename T>
  void process_buffers(T const *pa, T const *pb, T *pc, size_t sz);
};

int main() {
  arr_t A(LIST_SIZE), B(LIST_SIZE), C(LIST_SIZE);

  try {
    cl::sycl::gpu_selector gpsel;
    ocl_ctx_t ct{gpsel};

    cl::sycl::queue deviceQueue(gpsel);
    sycl::device Device = deviceQueue.get_device();

    if (!isInlineASMSupported(Device) || !Device.has_extension("cl_intel_required_subgroup_size")) {
      std::cout << "Skipping test\n";
      return 0;
    }

    for (int i = 0; i < LIST_SIZE; i++) {
      A[i] = i;
      B[i] = 2;
    }

    ct.process_buffers(A.data(), B.data(), C.data(), LIST_SIZE);

    for (int i = 0; i < LIST_SIZE; ++i)
      if (C[i] != A[i] * B[i]) {
        std::cerr << "At index: " << i << ". ";
        std::cerr << C[i] << " != " << A[i] * B[i] << "\n";
        abort();
      }

    std::cout << "Everything is correct" << std::endl;
  } catch (cl::sycl::exception const &err) {
    std::cerr << "ERROR: " << err.what() << ":\n";
    return -1;
  }
}

template <typename T>
void ocl_ctx_t::process_buffers(T const *pa, T const *pb, T *pc, size_t sz) {
  cl::sycl::range<1> numOfItems{sz};
  cl::sycl::buffer<T, 1> bufferA(pa, numOfItems);
  cl::sycl::buffer<T, 1> bufferB(pb, numOfItems);
  cl::sycl::buffer<T, 1> bufferC(pc, numOfItems);

  bufferA.set_final_data(nullptr);
  bufferB.set_final_data(nullptr);

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto A = bufferA.template get_access<sycl_read>(cgh);
    auto B = bufferB.template get_access<sycl_read>(cgh);
    auto C = bufferC.template get_access<sycl_write>(cgh);

    auto kern = [ A, B, C ](cl::sycl::id<1> wiID) [[cl::intel_reqd_sub_group_size(16)]] {

      // declaration of temp within and outside the scope
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
      asm("{\n"
          ".decl temp v_type=G type=d num_elts=16 align=GRF\n"
          "mov (M1, 16) temp(0, 0)<1> %1(0, 0)<1;1,0>\n"
          "mov (M1, 16) %0(0, 0)<1>  temp(0, 0)<1;1,0>\n"
          "}\n"
          ".decl temp v_type=G type=d num_elts=16 align=GRF\n"
          "mul (M1, 16) temp(0, 0)<1> %2(0, 0)<1;1,0> %0(0, 0)<1;1,0>\n"
          "mov (M1, 16) %0(0, 0)<1>  temp(0, 0)<1;1,0>\n"
          : "+rw"(C[wiID])
          : "rw"(A[wiID]), "rw"(B[wiID]));
#else
      C[wiID] = A[wiID];
      C[wiID] *= B[wiID];
#endif
    };
    cgh.parallel_for<class decl_in_scope_kernel<T>>(numOfItems, kern);
  });
}
