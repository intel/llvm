// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -sycl-std=2017 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple amdgcn-amd-amdhsa -disable-llvm-passes -sycl-std=2017 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx-nvidia-cuda -disable-llvm-passes -sycl-std=2017 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -disable-llvm-passes -sycl-std=2017 -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace sycl;
queue q;

class Functor32 {
public:
  [[sycl::reqd_work_group_size(32)]] void operator()() const {}
};
class Functor32x16 {
public:
  [[sycl::reqd_work_group_size(32, 16)]] void operator()() const {}
};
class Functor32x16x16 {
public:
  [[sycl::reqd_work_group_size(32, 16, 16)]] void operator()() const {}
};
class CLFunctor32x16x16 {
public:
  [[cl::reqd_work_group_size(32, 16, 16)]] void operator()() const {}
};

[[sycl::reqd_work_group_size(8)]] void f8() {}
[[sycl::reqd_work_group_size(8, 1)]] void f8x1() {}
[[sycl::reqd_work_group_size(8, 1, 1)]] void f8x1x1() {}
[[cl::reqd_work_group_size(8, 1, 1)]] void clf8x1x1() {}

class Functor1D {
public:
  void operator()() const {
    f8();
  }
};
class Functor2D {
public:
  void operator()() const {
    f8x1();
  }
};
class Functor3D {
public:
  void operator()() const {
    f8x1x1();
  }
};
class CLFunctor3D {
public:
  void operator()() const {
    clf8x1x1();
  }
};

template <int SIZE>
class FunctorTemp1D {
public:
  [[sycl::reqd_work_group_size(SIZE)]] void operator()() const {}
};
template <int SIZE, int SIZE1>
class FunctorTemp2D {
public:
  [[sycl::reqd_work_group_size(SIZE, SIZE1)]] void operator()() const {}
};
template <int SIZE, int SIZE1, int SIZE2>
class FunctorTemp3D {
public:
  [[sycl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};
template <int SIZE, int SIZE1, int SIZE2>
class CLFunctorTemp3D {
public:
  [[cl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};

template <int N>
[[sycl::reqd_work_group_size(N)]] void func1D() {}
template <int N, int N1>
[[sycl::reqd_work_group_size(N, N1)]] void func2D() {}
template <int N, int N1, int N2>
[[sycl::reqd_work_group_size(N, N1, N2)]] void func3D() {}
template <int N, int N1, int N2>
[[cl::reqd_work_group_size(N, N1, N2)]] void clfunc3D() {}

int main() {
  q.submit([&](handler &h) {
    CLFunctor32x16x16 clf32x16x16;
    h.single_task<class kernel_name1>(clf32x16x16);

    CLFunctor3D clf3d;
    h.single_task<class kernel_name2>(clf3d);

    h.single_task<class kernel_name3>(
        []() [[cl::reqd_work_group_size(8, 8, 8)]]{});

    CLFunctorTemp3D<2, 2, 2> clft3d;
    h.single_task<class kernel_name4>(clft3d);

    h.single_task<class kernel_name5>([]() {
      clfunc3D<8, 4, 4>();
    });

    h.single_task<class kernel_name6>(
        []() [[cl::reqd_work_group_size(1, 8, 2)]]{});

    Functor32x16x16 f32x16x16;
    h.single_task<class kernel_name7>(f32x16x16);

    Functor3D f3d;
    h.single_task<class kernel_name8>(f3d);

    h.single_task<class kernel_name9>(
        []() [[sycl::reqd_work_group_size(8, 8, 8)]]{});

    FunctorTemp3D<2, 2, 2> ft3d;
    h.single_task<class kernel_name10>(ft3d);

    h.single_task<class kernel_name11>([]() {
      func3D<8, 4, 4>();
    });

    h.single_task<class kernel_name12>(
        []() [[sycl::reqd_work_group_size(1, 8, 2)]]{});

    Functor32x16 f32x16;
    h.single_task<class kernel_name13>(f32x16);

    Functor2D f2d;
    h.single_task<class kernel_name14>(f2d);

    h.single_task<class kernel_name15>(
        []() [[sycl::reqd_work_group_size(8, 8)]]{});

    FunctorTemp2D<2, 2> ft2d;
    h.single_task<class kernel_name16>(ft2d);

    h.single_task<class kernel_name17>([]() {
      func2D<8, 4>();
    });

    h.single_task<class kernel_name18>(
        []() [[sycl::reqd_work_group_size(1, 8)]]{});

    Functor32 f32;
    h.single_task<class kernel_name19>(f32);

    Functor1D f1d;
    h.single_task<class kernel_name20>(f1d);

    h.single_task<class kernel_name21>(
        []() [[sycl::reqd_work_group_size(8)]]{});

    FunctorTemp1D<2> ft1d;
    h.single_task<class kernel_name22>(ft1d);

    h.single_task<class kernel_name23>([]() {
      func1D<8>();
    });

    h.single_task<class kernel_name24>(
        []() [[sycl::reqd_work_group_size(1)]]{});
  });
  return 0;
}

// CHECK: define {{.*}} void @{{.*}}kernel_name1() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D32:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name2() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D8:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name3() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D88:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name4() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D22:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name5() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D44:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name6() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D2:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name7() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D32]]
// CHECK: define {{.*}} void @{{.*}}kernel_name8() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D8]]
// CHECK: define {{.*}} void @{{.*}}kernel_name9() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D88]]
// CHECK: define {{.*}} void @{{.*}}kernel_name10() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D22]]
// CHECK: define {{.*}} void @{{.*}}kernel_name11() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D44]]
// CHECK: define {{.*}} void @{{.*}}kernel_name12() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D2]]
// CHECK: define {{.*}} void @{{.*}}kernel_name13() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D32:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name14() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D8:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name15() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D88:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name16() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D22:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name17() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D44:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name18() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D2_or_WGSIZE1D8:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name19() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE1D32:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name20() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D2_or_WGSIZE1D8]]
// CHECK: define {{.*}} void @{{.*}}kernel_name21() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D2_or_WGSIZE1D8]]
// CHECK: define {{.*}} void @{{.*}}kernel_name22() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE1D22:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name23() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D2_or_WGSIZE1D8]]
// CHECK: define {{.*}} void @{{.*}}kernel_name24() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE1D2:[0-9]+]]

// CHECK: ![[NDRWGS3D]] = !{i32 3}
// CHECK: ![[WGSIZE3D32]] = !{i32 16, i32 16, i32 32}
// CHECK: ![[WGSIZE3D8]] = !{i32 1, i32 1, i32 8}
// CHECK: ![[WGSIZE3D88]] = !{i32 8, i32 8, i32 8}
// CHECK: ![[WGSIZE3D22]] = !{i32 2, i32 2, i32 2}
// CHECK: ![[WGSIZE3D44]] = !{i32 4, i32 4, i32 8}
// CHECK: ![[WGSIZE3D2]] = !{i32 2, i32 8, i32 1}
// CHECK: ![[NDRWGS2D]] = !{i32 2}
// CHECK: ![[WGSIZE2D32]] = !{i32 16, i32 32, i32 1}
// CHECK: ![[WGSIZE2D8]] = !{i32 1, i32 8, i32 1}
// CHECK: ![[WGSIZE2D88]] = !{i32 8, i32 8, i32 1}
// CHECK: ![[WGSIZE2D22]] = !{i32 2, i32 2, i32 1}
// CHECK: ![[WGSIZE2D44]] = !{i32 4, i32 8, i32 1}
// CHECK: ![[WGSIZE2D2_or_WGSIZE1D8]] = !{i32 8, i32 1, i32 1}
// CHECK: ![[NDRWGS1D]] = !{i32 1}
// CHECK: ![[WGSIZE1D32]] = !{i32 32, i32 1, i32 1}
// CHECK: ![[WGSIZE1D22]] = !{i32 2, i32 1, i32 1}
// CHECK: ![[WGSIZE1D2]] = !{i32 1, i32 1, i32 1}
