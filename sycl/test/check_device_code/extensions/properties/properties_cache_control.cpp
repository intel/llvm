// RUN: %clangxx -fsycl-device-only -fsycl-targets=spir64 -S -emit-llvm %s -o - |    \
// RUN:    FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

using ST_L1 = cache_control<cache_mode::streaming, cache_level::L1>;
using WB_L1 = cache_control<cache_mode::write_back, cache_level::L1>;
using UC_L1 = cache_control<cache_mode::uncached, cache_level::L1>;
using CA_L1 = cache_control<cache_mode::cached, cache_level::L1>;
using UC_L2 = cache_control<cache_mode::uncached, cache_level::L2>;
using CA_L2 = cache_control<cache_mode::cached, cache_level::L2>;
using UC_L3 = cache_control<cache_mode::uncached, cache_level::L3>;
using CA_L3 = cache_control<cache_mode::cached, cache_level::L3>;

using load_hint = annotated_ptr<
    float, decltype(properties(
               read_hint<cache_control<cache_mode::cached, cache_level::L1>,
                         cache_control<cache_mode::uncached, cache_level::L2,
                                       cache_level::L3>>))>;
using load_assertion = annotated_ptr<
    int,
    decltype(properties(
        read_assertion<cache_control<cache_mode::constant, cache_level::L1>,
                       cache_control<cache_mode::invalidate, cache_level::L2,
                                     cache_level::L3>>))>;
using store_hint = annotated_ptr<
    float,
    decltype(properties(
        write_hint<cache_control<cache_mode::write_through, cache_level::L1>,
                   cache_control<cache_mode::write_back, cache_level::L2,
                                 cache_level::L3>,
                   cache_control<cache_mode::streaming, cache_level::L4>>))>;
using load_store_hint = annotated_ptr<
    float,
    decltype(properties(
        read_hint<cache_control<cache_mode::cached, cache_level::L3>>,
        read_assertion<cache_control<cache_mode::constant, cache_level::L4>>,
        write_hint<
            cache_control<cache_mode::write_through, cache_level::L4>>))>;

template <typename t>
using ap_load_ca_uc_uc =
    annotated_ptr<t, decltype(properties(read_hint<CA_L1, UC_L2, UC_L3>))>;

template <typename t>
using ap_load_st_ca_uc =
    annotated_ptr<t, decltype(properties(read_hint<ST_L1, CA_L2, CA_L3>))>;

template <typename T>
using ap_store_uc_uc_uc =
    annotated_ptr<T, decltype(properties(write_hint<UC_L1, UC_L2, UC_L3>))>;

template <typename T>
using ap_store_wb_uc_uc =
    annotated_ptr<T, decltype(properties(write_hint<WB_L1, UC_L2, UC_L3>))>;

void cache_control_read_hint_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.single_task<>([=]() {
      load_hint src{&ArrayA[0]};
      *src = 55.0f;
    });
  });
}

void cache_control_read_assertion_func() {
  queue q;
  constexpr int N = 10;
  int *ArrayA = malloc_shared<int>(N, q);
  q.submit([&](handler &cgh) {
    cgh.single_task<>([=]() {
      load_assertion src{&ArrayA[0]};
      *src = 66;
    });
  });
}

void cache_control_write_hint_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.single_task<>([=]() {
      store_hint dst{&ArrayA[0]};
      *dst = 77.0f;
    });
  });
}

void cache_control_read_write_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.single_task<>([=]() {
      load_store_hint dst{&ArrayA[0]};
      *dst = 77.0f;
    });
  });
}

void cache_control_load_store_func() {
  queue q(gpu_selector_v);

  constexpr int N = 512;
  double *x_buf = malloc_device<double>(N, q);
  double *y_buf = malloc_device<double>(N, q);
  double *d_buf = malloc_device<double>(1, q);
  double *d_buf_h = malloc_host<double>(1, q);

  q.fill<double>(d_buf, 0.0, 1).wait();

  constexpr int SG_SIZE = 16;

  q.submit([&](handler &cgh) {
     const int nwg = N / SG_SIZE;
     auto x = x_buf;
     auto y = y_buf;
     auto d = d_buf;
     auto d_h = d_buf_h;

     auto kernel =
         [=](nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
           const int global_tid = item.get_global_id(0);
           const int row_st = global_tid * SG_SIZE;

           if (row_st > N)
             return;

           const sub_group sgr = item.get_sub_group();
           const int sgr_tid = sgr.get_local_id();

           ap_store_uc_uc_uc<double> x_s(x);
           ap_store_wb_uc_uc<double> y_s(y);

           x_s[row_st + sgr_tid] = 1.0;
           y_s[row_st + sgr_tid] = 1.0;

           group_barrier(sgr);

           ap_load_ca_uc_uc<double> x_l(x);
           ap_load_st_ca_uc<double> y_l(y);

           const double xVal = x_l[row_st + sgr_tid];
           const double yVal = y_l[row_st + sgr_tid];
           double T = xVal * yVal;
           T = reduce_over_group(sgr, T, 0.0, std::plus<>());

           if (sgr.leader()) {
             atomic_ref<double, memory_order::relaxed, memory_scope::device,
                        access::address_space::global_space>
                 d_atomic(d[0]);
             d_atomic.fetch_add(T);
           }
         };
     cgh.parallel_for<class write_ker>(
         nd_range<2>(range<2>(nwg, SG_SIZE), range<2>(1, SG_SIZE)), kernel);
   }).wait();
}

// Test that annotated pointer parameter functions don't crash.
SYCL_EXTERNAL void annotated_ptr_func_param_test(float *p) {
  *(store_hint{p}) = 42.0f;
}

// CHECK: spir_func{{.*}}annotated_ptr_func_param_test
// CHECK: {{.*}}call ptr addrspace(4) @llvm.ptr.annotation.p4.p1{{.*}}!spirv.Decorations [[WHINT:.*]]
// CHECK: ret void

// CHECK: spir_kernel{{.*}}cache_control_read_hint_func
// CHECK: {{.*}}addrspacecast ptr addrspace(1){{.*}}!spirv.Decorations [[RHINT:.*]]
// CHECK: ret void

// CHECK: spir_kernel{{.*}}cache_control_read_assertion_func
// CHECK: {{.*}}addrspacecast ptr addrspace(1){{.*}}!spirv.Decorations [[RASSERT:.*]]
// CHECK: ret void

// CHECK: spir_kernel{{.*}}cache_control_write_hint_func
// CHECK: {{.*}}addrspacecast ptr addrspace(1){{.*}}!spirv.Decorations [[WHINT]]
// CHECK: ret void

// CHECK: spir_kernel{{.*}}cache_control_read_write_func
// CHECK: {{.*}}addrspacecast ptr addrspace(1){{.*}}!spirv.Decorations [[RWHINT:.*]]
// CHECK: ret void

// CHECK: spir_kernel{{.*}}cache_control_load_store_func
// CHECK: {{.*}}getelementptr{{.*}}addrspace(4){{.*}}!spirv.Decorations [[LDSTHINT_A:.*]]
// CHECK: {{.*}}getelementptr{{.*}}addrspace(4){{.*}}!spirv.Decorations [[LDSTHINT_B:.*]]
// CHECK: ret void

// CHECK: [[WHINT]] = !{[[WHINT1:.*]], [[WHINT2:.*]], [[WHINT3:.*]], [[WHINT4:.*]]}
// CHECK: [[WHINT1]] = !{i32 6443, i32 3, i32 3}
// CHECK: [[WHINT2]] = !{i32 6443, i32 0, i32 1}
// CHECK: [[WHINT3]] = !{i32 6443, i32 1, i32 2}
// CHECK: [[WHINT4]] = !{i32 6443, i32 2, i32 2}

// CHECK: [[RHINT]] = !{[[RHINT1:.*]], [[RHINT2:.*]], [[RHINT3:.*]]}
// CHECK: [[RHINT1]] = !{i32 6442, i32 1, i32 0}
// CHECK: [[RHINT2]] = !{i32 6442, i32 2, i32 0}
// CHECK: [[RHINT3]] = !{i32 6442, i32 0, i32 1}

// CHECK: [[RASSERT]] = !{[[RASSERT1:.*]], [[RASSERT2:.*]], [[RASSERT3:.*]]}
// CHECK: [[RASSERT1]] = !{i32 6442, i32 1, i32 3}
// CHECK: [[RASSERT2]] = !{i32 6442, i32 2, i32 3}
// CHECK: [[RASSERT3]] = !{i32 6442, i32 0, i32 4}

// CHECK: [[RWHINT]] = !{[[RWHINT1:.*]], [[RWHINT2:.*]], [[RWHINT3:.*]]}
// CHECK: [[RWHINT1]] = !{i32 6442, i32 2, i32 1}
// CHECK: [[RWHINT2]] = !{i32 6442, i32 3, i32 4}
// CHECK: [[RWHINT3]] = !{i32 6443, i32 3, i32 1}

// CHECK: [[LDSTHINT_A]] = !{[[RHINT1]], [[RHINT2]], [[RHINT3]], [[LDSTHINT_A1:.*]], [[LDSTHINT_A2:.*]], [[LDSTHINT_A3:.*]]}
// CHECK: [[LDSTHINT_A1]] = !{i32 6443, i32 0, i32 0}
// CHECK: [[LDSTHINT_A2]] = !{i32 6443, i32 1, i32 0}
// CHECK: [[LDSTHINT_A3]] = !{i32 6443, i32 2, i32 0}

// CHECK: [[LDSTHINT_B]] = !{[[LDSTHINT_B1:.*]], [[RWHINT1]], [[LDSTHINT_B2:.*]], [[LDSTHINT_A2]], [[LDSTHINT_A3]], [[LDSTHINT_B3:.*]]}
// CHECK: [[LDSTHINT_B1]] = !{i32 6442, i32 1, i32 1}
// CHECK: [[LDSTHINT_B2]] = !{i32 6442, i32 0, i32 2}
// CHECK: [[LDSTHINT_B3]] = !{i32 6443, i32 0, i32 2}
