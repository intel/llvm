// FIXME: Investigate OS-agnostic failures
// REQUIRES: TEMPORARY_DISABLED

// UNSUPPORTED: cuda || hip
// CUDA does not support SYCL 1.2.1 images.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/accessor.hpp>
#include <sycl/sycl.hpp>
using namespace sycl;
#define N 4   // dimensin
#define M 128 // dimension
#define C 4   // 4 channel
#define L 2   // 2 images

void try_1D(queue &Q) {
  int X = -55;
  buffer<int, 1> BX{&X, 1};
  int *host_array = new int[M * C];
  image im1(host_array, image_channel_order::rgba,
            image_channel_type::unsigned_int8, range{M});

  Q.submit([&](handler &h) {
    accessor<int4, 1, access::mode::read, access::target::image> acs1(im1, h);
    accessor ABX{BX, h};
    auto R = acs1.get_range();
    std::cout << "Host acs1.get_range()=" << R[0] << "\n";
    assert(R[0] == M);
    h.parallel_for(nd_range{range{M}, range{N}}, [=](nd_item<1> it) {
      int idx = it.get_global_linear_id();
      if (idx == 0) {
        auto R = acs1.get_range();
        ABX[0] = R[0];
      }
    });
  });
  {
    host_accessor HABX{BX, read_only};
    std::cout << "From Device acs1.get_range()=" << X << "\n";
    assert(X == M);
  }
}

void try_2D(queue &Q) {
  range<2> X = {55, 66};
  buffer<range<2>, 1> BX{&X, 1};
  int *host_array = new int[M * N * C];
  image im2(host_array, image_channel_order::rgba,
            image_channel_type::unsigned_int8, range{M, N});

  Q.submit([&](handler &h) {
    accessor<int4, 2, access::mode::read, access::target::image> acs2(im2, h);
    accessor ABX{BX, h};
    auto R = acs2.get_range();
    std::cout << "Host acs2.get_range()=" << R[0] << "," << R[1] << "\n";
    assert(R[0] == M);
    assert(R[1] == N);
    h.parallel_for(nd_range{range{M, N}, range{N, N}}, [=](nd_item<2> it) {
      int idx = it.get_global_linear_id();
      if (idx == 0) {
        ABX[0] = acs2.get_range();
      }
    });
  });
  {
    host_accessor HABX{BX, read_only};
    std::cout << "From Device acs2.get_range()=" << HABX[0][0] << ","
              << HABX[0][1] << "\n";
    assert(HABX[0][0] == M);
    assert(HABX[0][1] == N);
  }
}

void try_3D(queue &Q) {
  range<3> X{55, 66, 77};
  buffer<range<3>, 1> BX{&X, 1};
  int *host_array3_2 = malloc_host<int>(N * M * C * L, Q);
  image im3(host_array3_2, image_channel_order::rgba,
            image_channel_type::unsigned_int8, range{M, N, L});

  Q.submit([&](handler &h) {
    accessor<int4, 2, access::mode::read, access::target::image_array> acs3(im3,
                                                                            h);
    accessor ABX{BX, h};
    auto R = acs3.get_range();
    std::cout << "Host acs3.get_range()=" << R[0] << "," << R[1] << "," << R[2]
              << "\n";
    assert(R[0] == M);
    assert(R[1] == N);
    assert(R[2] == L);
    h.parallel_for(nd_range{range{M, N, L}, range{N, N, L}},
                   [=](nd_item<3> it) {
                     int idx = it.get_global_linear_id();
                     if (idx == 0) {
                       ABX[0] = acs3.get_range();
                     }
                   });
  });
  {
    host_accessor HABX{BX, read_only};
    std::cout << "From Device acs3.get_range()=" << HABX[0][0] << ","
              << HABX[0][1] << "," << HABX[0][2] << "\n";
    assert(HABX[0][0] == M);
    assert(HABX[0][1] == N);
    assert(HABX[0][2] == L);
  }
}

int main() {
  queue Q;

  try_1D(Q);
  try_2D(Q);
  try_3D(Q);

  return 0;
}
