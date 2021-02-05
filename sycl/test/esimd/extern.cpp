// RUN: %clangxx -fsycl-explicit-simd -fsycl -fsycl-device-only -c %s -o %t.o

// This test references an external function vadd. It ensures that
// simd buffer arguments to vadd are converted to native vector
// types, although callee definition is not visible.

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;

constexpr unsigned VL = 16;

extern SYCL_EXTERNAL sycl::INTEL::gpu::simd<float, VL>
  vadd(sycl::INTEL::gpu::simd<float, VL> a,
       sycl::INTEL::gpu::simd<float, VL> b);

int main(void) {
    constexpr unsigned Size = 16; //1024 * 128;

    float *A = new float[Size];
    float *B = new float[Size];
    float *C = new float[Size];

    for (unsigned i = 0; i < Size; ++i) {
        A[i] = B[i] = i;
        C[i] = 0.0f;
    }

    auto asyncHandler = [](cl::sycl::exception_list ExceptionList) {
        for (auto &Exception : ExceptionList) {
            std::rethrow_exception(Exception);
        }
    };

    {
        buffer<float, 1> bufa(A, range<1>(Size));
        buffer<float, 1> bufb(B, range<1>(Size));
        buffer<float, 1> bufc(C, range<1>(Size));

        // We need that many workgroups
        cl::sycl::range<1> GlobalRange{Size / VL};

        // We need that many threads in each group
        cl::sycl::range<1> LocalRange{1};

        queue q(gpu_selector{}, asyncHandler);

        auto dev = q.get_device();
        std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

        auto e = q.submit([&](handler &cgh) {
        auto PA = bufa.get_access<access::mode::read>(cgh);
        auto PB = bufb.get_access<access::mode::read>(cgh);
        auto PC = bufc.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class Test>(
            GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
                using namespace sycl::INTEL::gpu;
                unsigned int offset = i * VL * sizeof(float);
                simd<float, VL> va = block_load<float, VL>(PA, offset);
                simd<float, VL> vb = block_load<float, VL>(PB, offset);
                simd<float, VL> vc = vadd(va, vb);
                block_store(PC, offset, vc);
            });
        });
        e.wait();
    }

    int err_cnt = 0;

    for (unsigned i = 0; i < Size; ++i) {
        std::cout << "C[" << i << "] = " << C[i] << " = " << A[i]
                    << " + " << B[i] << "\n";
        if (A[i] + B[i] != C[i]) {
        if (++err_cnt < 10) {
            std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                    << " + " << B[i] << "\n";
        }
        }
    }
    if (err_cnt > 0) {
        std::cout << "  pass rate: "
                << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
                << (Size - err_cnt) << "/" << Size << ")\n";
    }

    delete[] A;
    delete[] B;
    delete[] C;

    std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
    return err_cnt > 0 ? 1 : 0;
}
