// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -S -emit-llvm  -o %t_temp.ll %s
// RUN: %clangxx -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s

// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -fno-inline -Xclang -sycl-std=2020 -mllvm -sycl-opt -S -emit-llvm  -o %t_temp.ll %s
// RUN: %clangxx -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s --check-prefix=CHECK-TL

// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -Xclang -fenable-sycl-dae -mllvm -sycl-opt -mllvm -inline-threshold=500 -S -emit-llvm %s -o %t_temp.ll
// RUN: %clangxx -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s

// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -Xclang -fenable-sycl-dae -mllvm -sycl-opt -fno-inline -S -emit-llvm %s -o %t_temp.ll
// RUN: %clangxx -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s --check-prefix=CHECK-TL

// Check that builtins are emitted as expected
// RUN: %clangxx -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s --check-prefix=CHECK-BT

// check that we added the state struct as a function argument, and that we
// inject the calls to our builtins.

// CHECK: %struct.__nativecpu_state = type { [3 x i64], [3 x i64], [3 x i64], [3 x i64], [3 x i64], [3 x i64], [3 x i64] }
#include "sycl.hpp"
class Test1;
class Test2;
class Test3;
int main() {
  sycl::queue deviceQueue;
  sycl::accessor<int, 1, sycl::access::mode::write> acc;
  sycl::range<1> r(1);
  deviceQueue.submit([&](sycl::handler &h) {
    h.parallel_for<Test1>(r, [=](sycl::id<1> id) { acc[id[0]] = 42; });
    // CHECK: @_ZTS5Test1.NativeCPUKernel(ptr {{.*}}, ptr {{.*}}, ptr addrspace(1){{.*}})
    // CHECK: call{{.*}}__dpcpp_nativecpu_get_global_id(i32 0, ptr addrspace(1) %2)
    // CHECK-NOT: @llvm.threadlocal

    // CHECK-TL:      %[[VAL1:.*]] = call ptr addrspace(1) @llvm.threadlocal.address.p1(ptr addrspace(1) @_ZL28nativecpu_thread_local_state)
    // CHECK-TL-NEXT  %[[VAL2:.*]] = load ptr addrspace(1), ptr addrspace(1) %VAL1, align 8
    // CHECK-TL-NEXT  %{{.*}} = call i64 @__dpcpp_nativecpu_get_wg_size(i32 0, ptr addrspace(1) %VAL2)

    // CHECK-TL:      %{{.*}} = call ptr addrspace(1) @llvm.threadlocal.address.p1(ptr addrspace(1) @_ZL28nativecpu_thread_local_state)
    // CHECK-TL-DAG: store ptr addrspace(1) %{{.*}}, ptr addrspace(1) %{{.*}}, align 8
    // CHECK-TL-DAG: call void @_ZTS5Test1.NativeCPUKernel(ptr addrspace(1) %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
  });
  sycl::nd_range<2> r2({1, 1}, {
                                   1,
                                   1,
                               });
  deviceQueue.submit([&](sycl::handler &h) {
    h.parallel_for<Test2>(r2, [=](sycl::id<2> id) { acc[id[1]] = 42; });
    // CHECK: @_ZTS5Test2.NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, ptr addrspace(1) %2)
    // CHECK: call{{.*}}__dpcpp_nativecpu_get_global_id(i32 1, ptr addrspace(1) %2)
    // CHECK: call{{.*}}__dpcpp_nativecpu_get_global_id(i32 0, ptr addrspace(1) %2)
  });
  sycl::nd_range<3> r3({1, 1, 1}, {1, 1, 1});
  deviceQueue.submit([&](sycl::handler &h) {
    h.parallel_for<Test3>(
        r3, [=](sycl::item<3> item) { acc[item[2]] = item.get_range(0); });
    // CHECK: @_ZTS5Test3.NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, ptr addrspace(1) %2)
    // CHECK-DAG: call{{.*}}__dpcpp_nativecpu_get_global_range(i32 2, ptr addrspace(1) %2)
    // CHECK-DAG: call{{.*}}__dpcpp_nativecpu_get_global_range(i32 1, ptr addrspace(1) %2)
    // CHECK-DAG: call{{.*}}__dpcpp_nativecpu_get_global_range(i32 0, ptr addrspace(1) %2)
    // CHECK-DAG: call{{.*}}__dpcpp_nativecpu_get_global_id(i32 2, ptr addrspace(1) %2)
    // CHECK-DAG: call{{.*}}__dpcpp_nativecpu_get_global_id(i32 1, ptr addrspace(1) %2)
    // CHECK-DAG: call{{.*}}__dpcpp_nativecpu_get_global_id(i32 0, ptr addrspace(1) %2)
  });

  const size_t dim = 2;
  using dataT = std::tuple<size_t, size_t, size_t>;
  sycl::range<3> NumOfWorkItems{2 * dim, 2 * (dim + 1), 2 * (dim + 2)};
  sycl::range<3> LocalSizes{dim, dim + 1, dim + 2};
  sycl::buffer<dataT, 3> Buffer(NumOfWorkItems);

  sycl::queue Queue;

  Queue.submit([&](sycl::handler &cgh) {
    sycl::accessor Accessor{Buffer, cgh, sycl::write_only};
    sycl::nd_range<3> TheRange{NumOfWorkItems, LocalSizes};
    cgh.parallel_for<class FillBuffer>(TheRange, [=](sycl::nd_item<3> id) {
      auto localX = id.get_local_id(0);
      auto localY = id.get_local_id(1);
      auto localZ = id.get_local_id(2);

      auto groupX = id.get_group(0);
      auto groupY = id.get_group(1);
      auto groupZ = id.get_group(2);

      auto rangeX = id.get_local_range(0);
      auto rangeY = id.get_local_range(1);
      auto rangeZ = id.get_local_range(2);
      Accessor[groupX * rangeX + localX][groupY * rangeY + localY]
              [groupZ * rangeZ + localZ] = {rangeX, rangeY, rangeZ};

      // CHECK-DAG: call{{.*}}__dpcpp_nativecpu_get_local_id(i32 0, ptr addrspace(1) %{{[0-9]*}})
      // CHECK-DAG: call{{.*}}__dpcpp_nativecpu_get_wg_size(i32 0, ptr addrspace(1) %{{[0-9]*}})
      // CHECK-DAG: call{{.*}}__dpcpp_nativecpu_get_wg_id(i32 0, ptr addrspace(1) %{{[0-9]*}})
    });
  });
}

// check that builtins are generated as expected
// CHECK-BT: define internal i64 @__dpcpp_nativecpu_get_global_id(i32 %[[DIMARG:.*]], ptr addrspace(1) %[[PTRARG:.*]]) {
// CHECK-BT:   %[[GEP:.*]] = getelementptr %struct.__nativecpu_state, ptr addrspace(1) %[[PTRARG]], i64 0, i32 0, i32 %[[DIMARG]]
// CHECK-BT:   %[[LOAD:.*]] = load i64, ptr addrspace(1) %[[GEP]], align 8
// CHECK-BT:   ret i64 %[[LOAD]]
// CHECK-BT: }

// check that the generated module has the is-native-cpu module flag set
// CHECK: !{{[0-9]*}} = !{i32 1, !"is-native-cpu", i32 1}
