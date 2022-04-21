// RUN: %clangxx -fsycl-device-only -fsycl-early-optimizations -fsycl-dead-args-optimization -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ -S -emit-llvm -o - %s | FileCheck %s

#include <CL/sycl.hpp>

inline constexpr int size = 100;

int main() {
    {
        sycl::buffer<int> a{sycl::range{size}};
        sycl::buffer<int> b{sycl::range{size}};

        sycl::queue q;

        q.submit([&](sycl::handler &cgh) {
            sycl::ext::oneapi::accessor_property_list PL{sycl::ext::oneapi::no_offset, sycl::no_init};
            sycl::accessor acc_a(a, cgh, sycl::write_only, PL);
            sycl::accessor acc_b{b, cgh, sycl::read_only};
            // CHECK: define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlT_E_(i32 addrspace(1)* {{.*}}, i32 addrspace(1)* noundef readonly {{.*}}, %"class.cl::sycl::id"* noundef byval(%"class.cl::sycl::id") align 8 {{.*}})
            cgh.parallel_for(size, [=](auto i) {
                acc_a[i] = acc_b[i];
            });
        });

        q.wait();
    }

    {
        sycl::buffer<int> a{sycl::range{size}};
        sycl::buffer<int> b{sycl::range{size}};

        sycl::queue q;

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor acc_a(a, cgh, sycl::write_only);
            sycl::accessor acc_b{b, cgh, sycl::read_only};
            // CHECK: define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE0_clES2_EUlT_E_(i32 addrspace(1)* {{.*}}, %"class.cl::sycl::id"* noundef byval(%"class.cl::sycl::id") align 8 {{.*}}, i32 addrspace(1)* noundef readonly {{.*}}, %"class.cl::sycl::id"* noundef byval(%"class.cl::sycl::id") align 8 {{.*}})
            cgh.parallel_for(size, [=](auto i) {
                acc_a[i] = acc_b[i];
            });
        });

        q.wait();
    }

    return 0;
}
