// REQUIRES: hip

// RUN: %clangxx -fsycl-device-only -fsycl-targets=amd_gpu_gfx90a -S -Xclang -emit-llvm %s -o -| FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {

  buffer<double, 1> bufA(nullptr, range<1>(1));
  buffer<double, 1> bufB(nullptr, range<1>(1));
  buffer<double, 1> bufC(nullptr, range<1>(1));
  buffer<double, 1> bufD(nullptr, range<1>(1));

  queue q;

  q.submit([&](handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accA(bufA, cgh);
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accB(bufB, cgh);
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accC(bufC, cgh);
    sycl::accessor<double, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accD(bufD, cgh);

    cgh.parallel_for<class row_row_m16n16k4>(
        nd_range<2>({1, 64}, {1, 64}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 64)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, double, use::accumulator, 16, 16> sub_c{};
          joint_matrix<sub_group, double, use::a, 16, 4, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, double, use::b, 4, 16, layout::row_major>
              sub_b{};

          // CHECK: tail call <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double %{{.*}}, double %{{.*}}, <4 x double> zeroinitializer, i32 0, i32 0, i32 0)
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              16, layout::row_major);
        });
  });

  return 0;
};
