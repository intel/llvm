// RUN: %clang_cc1 -fsycl-is-device %s -emit-llvm -triple nvptx64-nvidia-cuda-sycldevice -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel<class kernel_no_reqd_work_size>([]() {});
  // CHECK: define dso_local void @{{.*}}kernel_no_reqd_work_size()
  // CHECK-NOT: define dso_local void @{{.*}}kernel_no_reqd_work_size() {{.*}} !reqd_work_group_size ![[WGSIZE1D:[0-9]+]]

  kernel<class kernel_reqd_work_size_1d>(
      []() [[intel::reqd_work_group_size(32)]]{});
  // CHECK: define dso_local void @{{.*}}kernel_reqd_work_size_1d() {{.*}} !reqd_work_group_size ![[WGSIZE1D:[0-9]+]]

  kernel<class kernel_reqd_work_size_2d>(
      []() [[intel::reqd_work_group_size(64, 32)]]{});
  // CHECK: define dso_local void @{{.*}}kernel_reqd_work_size_2d() {{.*}} !reqd_work_group_size ![[WGSIZE2D:[0-9]+]]

  kernel<class kernel_reqd_work_size_3d>(
      []() [[intel::reqd_work_group_size(128, 64, 32)]]{});
  // CHECK: define dso_local void @{{.*}}kernel_reqd_work_size_3d() {{.*}} !reqd_work_group_size ![[WGSIZE3D:[0-9]+]]
}

// CHECK-NOT: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_no_reqd_work_size, !"reqntidx", i32 !{{[0-9]+}}}
// CHECK-NOT: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_no_reqd_work_size, !"reqntidy", i32 !{{[0-9]+}}}
// CHECK-NOT: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_no_reqd_work_size, !"reqntidz", i32 !{{[0-9]+}}}

// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_reqd_work_size_1d, !"reqntidx", i32 1}
// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_reqd_work_size_1d, !"reqntidy", i32 1}
// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_reqd_work_size_1d, !"reqntidz", i32 32}

// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_reqd_work_size_2d, !"reqntidx", i32 1}
// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_reqd_work_size_2d, !"reqntidy", i32 32}
// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_reqd_work_size_2d, !"reqntidz", i32 64}

// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_reqd_work_size_3d, !"reqntidx", i32 32}
// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_reqd_work_size_3d, !"reqntidy", i32 64}
// CHECK: !{{[0-9]+}} = !{void ()* @{{.*}}kernel_reqd_work_size_3d, !"reqntidz", i32 128}

// CHECK: ![[WGSIZE1D]] = !{i32 1, i32 1, i32 32}
// CHECK: ![[WGSIZE2D]] = !{i32 1, i32 32, i32 64}
// CHECK: ![[WGSIZE3D]] = !{i32 32, i32 64, i32 128}
