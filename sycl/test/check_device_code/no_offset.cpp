// RUN: %clangxx -fsycl-device-only -fsycl-early-optimizations -fsycl-dead-args-optimization -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ -S -emit-llvm -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

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
            // CHECK: define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlT_E_(ptr addrspace(1) {{.*}}, ptr addrspace(1) noundef readonly {{.*}}, ptr noundef byval(%"class.sycl::_V1::id") align 8 {{.*}})
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
            // CHECK: define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_EUlT_E_(ptr addrspace(1) {{.*}}, ptr noundef byval(%"class.sycl::_V1::id") align 8 {{.*}}, ptr addrspace(1) noundef readonly {{.*}}, ptr noundef byval(%"class.sycl::_V1::id") align 8 {{.*}})
            cgh.parallel_for(size, [=](auto i) {
                acc_a[i] = acc_b[i];
            });
        });

        q.wait();
    }

    return 0;
}
