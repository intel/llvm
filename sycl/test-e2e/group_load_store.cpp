// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Running like this manually for the time being:
// clang++ -fsycl %s &&  ./a.out && \
// clang++ -fsycl %s -S -emit-llvm -fsycl-device-only -mllvm
// -inline-threshold=5000 -o - | FileCheck %s

#include <iomanip>
#include <sycl/sycl.hpp>

using namespace sycl;

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::property;
using namespace access;

// Using-enum-declaration is C++20.
constexpr auto global_space = address_space::global_space;
constexpr auto local_space = address_space::local_space;
constexpr auto blocked = group_algorithm_data_placement::blocked;
constexpr auto striped = group_algorithm_data_placement::striped;

constexpr int SG_SIZE = 16;
constexpr int N_WGS = 3;

__attribute__((noinline)) __attribute__((optnone)) void
marker(int l = __builtin_LINE()) {
  std::ignore = l;
}

template <typename T>
__attribute__((noinline)) __attribute__((optnone)) void record(T val,
                                                               int *&res_ptr) {
  // Hide pointer arithmetic/type cast from the IR dumps for readability.
  *res_ptr = val;
  res_ptr++;
}

void capture_marker() {
  queue q;
  q.single_task<class MarkerKernel>([]() {
     // CHECK-LABEL: define weak_odr dso_local spir_kernel {{.*}}MarkerKernel
     // Alternatively, we can use FileCheck's -D<..> option and avoid this
     // kernel alltogether, but that would be clumsy.
     marker();
     // CHECK: [[MARKER:call spir_func void @_Z6markeri\(i32 noundef]] [[# @LINE - 1]]
   }).wait();
}

enum Scope { WG, SG };

template <class TestTy> struct Kernel;

constexpr int N_RESULTS = 128;

auto create_results_buf(int global_size) {
  buffer<int, 1> results(N_RESULTS * global_size);
  for (auto &elem : host_accessor{results})
    elem = -1;

  return results;
}

void check_results_buf(buffer<int, 1> results) {
  auto global_size = results.size() / N_RESULTS;
  host_accessor res_acc{results};
  bool success = std::find(res_acc.begin(), res_acc.end(), 0) == res_acc.end();
  if (success)
    return;

  for (int i = 0; i < N_RESULTS; ++i) {
    bool all_same = [&]() {
      auto val = res_acc[i];
      for (int j = 0; j < global_size; ++j) {
        if (val != res_acc[j * N_RESULTS + i])
          return false;
      }
      return true;
    }();
    if (all_same) {
      if (res_acc[i] != -1) // Initial init value, nothing written.
        std::cout << "All: " << res_acc[i] << std::endl;
      continue;
    }
    for (int j = 0; j < global_size; ++j) {
      std::cout << " " << std::setw(3) << res_acc[j * N_RESULTS + i];
      if (j % 8 == 7)
        std::cout << "  |";
    }
    std::cout << std::endl;
  }
  assert(false);
}

void test_ptr_dispatch(size_t wg_size) {
  std::cout << "CheckPtrDispatch" << std::endl;
  // Verify that multi_ptr/generic_ptr are looked through. This is implemented
  // in a single helper so it's enough to verify using sub_group's scalar
  // group_load.
  queue q;
  size_t global_size = wg_size * N_WGS;
  auto results = create_results_buf(global_size);

  // Not greater than 8 vec/array size + gap between WGs.
  constexpr int ELEMS_PER_WI = 8 * 2;

  buffer<int, 1> global_mem_buf(global_size * ELEMS_PER_WI);

  auto *usm_mem_alloc = malloc_device<int>(global_size * ELEMS_PER_WI, q);
  auto Deleter = [=](auto *Ptr) { free(Ptr, q); };
  std::unique_ptr<int, decltype(Deleter)> smart_ptr(usm_mem_alloc, Deleter);

  q.submit([&](handler &cgh) {
    accessor res_acc{results, cgh};
    accessor global_mem_acc{global_mem_buf, cgh};
    local_accessor<int, 1> local_mem_acc{wg_size * ELEMS_PER_WI, cgh};
    cgh.parallel_for<class CheckPtrDispatch>(
        nd_range{range<1>{global_size}, range<1>{wg_size}},
        [=](nd_item<1> ndi) {
          auto *res_ptr = &res_acc[ndi.get_global_id(0) * N_RESULTS];
          auto Record = [&](auto val) { record(val, res_ptr); };

          // TODO: Use accessor::get_pointer once it starts returing raw
          // pointer.
          auto *global_mem =
              global_mem_acc.get_multi_ptr<decorated::no>().get_raw();
          auto *local_mem =
              local_mem_acc.get_multi_ptr<decorated::no>().get_raw();

          // Each WG receives its own "local" sub-region.
          int wg_offset = wg_size * ELEMS_PER_WI * ndi.get_group(0);

          auto sg = ndi.get_sub_group();
          int sg_offset =
              sg.get_max_local_range() * ELEMS_PER_WI * sg.get_group_id();

          auto *usm_mem = usm_mem_alloc + wg_offset + sg_offset;
          global_mem += wg_offset + sg_offset;
          local_mem += sg_offset;

          auto lid = sg.get_local_id();

          int init = ndi.get_global_id(0);

          lid = sg.get_local_id();
          global_mem[lid] = init;
          usm_mem[lid] = init;
          local_mem[lid] = init;

          group_barrier(ndi.get_group());

          auto Check = [&](auto Input, int l = __builtin_LINE())
              __attribute__((always_inline)) {
            marker(l);
            int out;
            group_load(sg, Input, out);
            Record(out == init);
          };

          // CHECK-LABEL: define weak_odr dso_local spir_kernel void @{{.*}}CheckPtrDispatch
          Check(global_mem); // Dynamic address space dispatch.
          // CHECK: [[MARKER]] [[# @LINE - 1]]
          // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
          // CHECK: icmp eq {{.*}}, null
          // CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

          Check(usm_mem); // Dynamic address space dispatch.
          // CHECK: [[MARKER]] [[# @LINE - 1]]
          // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
          // CHECK: icmp eq {{.*}}, null
          // CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

          Check(local_mem);
          // CHECK: [[MARKER]] [[# @LINE - 1]]
          // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
          // CHECK: icmp eq {{.*}}, null
          // CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

          Check(address_space_cast<global_space, decorated::yes>(global_mem));
          // CHECK: [[MARKER]] [[# @LINE - 1]]
          // CHECK-NOT: icmp eq {{.*}}, null
          // CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

          Check(address_space_cast<global_space, decorated::no>(global_mem));
          // CHECK: [[MARKER]] [[# @LINE - 1]]
          // CHECK-NOT: icmp eq {{.*}}, null
          // CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

          Check(address_space_cast<global_space, decorated::yes>(global_mem)
                    .get_decorated());
          // CHECK: [[MARKER]] [[# @LINE - 2]]
          // CHECK-NOT: icmp eq {{.*}}, null
          // CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

          Check(address_space_cast<local_space, decorated::yes>(local_mem));
          // CHECK: [[MARKER]] [[# @LINE - 1]]
          // CHECK-NOT: SubgroupBlockRead

          Check(address_space_cast<local_space, decorated::no>(local_mem));
          // CHECK: [[MARKER]] [[# @LINE - 1]]
          // CHECK-NOT: SubgroupBlockRead

          Check(address_space_cast<local_space, decorated::yes>(local_mem)
                    .get_decorated());
          // CHECK: [[MARKER]] [[# @LINE - 2]]
          // CHECK-NOT: SubgroupBlockRead

          marker(); // CHECK: [[MARKER]] [[# @LINE ]]
        });
  });

  check_results_buf(results);
}

template <typename TestTy> void test(TestTy TestObj, size_t wg_size) {
  size_t global_size = wg_size * N_WGS;
  std::string Name{typeid(TestTy).name()};
  // GCC/clang give mangled name which has the format <NameLength>Name. Strip
  // the leading number.
  Name.erase(Name.begin(),
             std::find_if(Name.begin(), Name.end(),
                          [](unsigned char ch) { return !std::isdigit(ch); }));

  std::cout << Name << std::endl;

  queue q;
  auto results = create_results_buf(global_size);

  // Not greater than 8 vec/array size + gap between WGs.
  constexpr int ELEMS_PER_WI = 8 * 2;

  buffer<int, 1> global_mem_buf(global_size * ELEMS_PER_WI);
  for (auto &elem : host_accessor{global_mem_buf})
    elem = 0;

  q.submit([&](handler &cgh) {
    accessor res_acc{results, cgh};
    accessor global_mem_acc{global_mem_buf, cgh};
    cgh.parallel_for<Kernel<TestTy>>(
        nd_range{range<1>{global_size}, range<1>{wg_size}},
        [=](nd_item<1> ndi) {
          auto *res_ptr = &res_acc[ndi.get_global_id(0) * N_RESULTS];
          auto Record = [&](auto val) { record(val, res_ptr); };

          // TODO: Use accessor::get_pointer once it starts returing raw
          // pointer.
          auto *global_mem =
              global_mem_acc.get_multi_ptr<decorated::no>().get_raw();

          // Each WG receives its own "local" sub-region.
          int wg_offset = wg_size * ELEMS_PER_WI * ndi.get_group(0);

          auto g = ndi.get_group();
          auto sg = ndi.get_sub_group();
          int sg_offset = 0;
          if constexpr (TestTy::Scope == SG)
            sg_offset =
                sg.get_max_local_range() * ELEMS_PER_WI * sg.get_group_id();

          global_mem += wg_offset + sg_offset;

#define KERNEL_OP                                                              \
  template <typename RecordTy, typename DecoratedInt>                          \
  void operator()(nd_item<1> ndi, size_t gid, [[maybe_unused]] size_t lid,     \
                  [[maybe_unused]] size_t sg_lid,                              \
                  [[maybe_unused]] size_t wg_size,                             \
                  [[maybe_unused]] group<1> g, [[maybe_unused]] sub_group sg,  \
                  DecoratedInt *global_mem, RecordTy Record) const

          TestObj(ndi, ndi.get_global_id(0), ndi.get_local_id(0),
                  sg.get_local_id(), g.get_local_range().size(), g, sg,
                  address_space_cast<global_space, decorated::yes>(global_mem)
                      .get_decorated(),
                  Record);
        });
  });

  check_results_buf(results);
}

struct ScalarWGTest {
  static constexpr Scope Scope = WG;
  KERNEL_OP {
    global_mem[lid] = gid;

    group_barrier(g);

    // CHECK-LABEL: define weak_odr dso_local spir_kernel void @{{.*}}Kernel{{.*}}ScalarWGTest
    marker(); // CHECK: [[MARKER]] [[# @LINE ]]

    int out;
    group_load(ndi.get_group(), global_mem, out);
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef
    Record(out == gid);

    marker(); // CHECK: [[MARKER]] [[# @LINE ]]
  }
};

struct ScalarSGTest {
  static constexpr Scope Scope = SG;
  KERNEL_OP {
    global_mem[sg_lid] = gid;

    group_barrier(g);

    // CHECK-LABEL: define weak_odr dso_local spir_kernel void @{{.*}}Kernel{{.*}}ScalarSGTest
    marker(); // CHECK: [[MARKER]] [[# @LINE ]]

    int out;
    group_load(sg, global_mem, out);
    // CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef
    Record(out == gid);

    marker(); // CHECK: [[MARKER]] [[# @LINE ]]
  }
};

struct VecBlockedWGTest {
  static constexpr Scope Scope = SG;
  KERNEL_OP {
    constexpr int VEC_SIZE = 2;

    for (int i = 0; i < VEC_SIZE; ++i)
      global_mem[lid * VEC_SIZE + i] = gid + VEC_SIZE * 2 - i;

    group_barrier(g);

    // CHECK-LABEL: define weak_odr dso_local spir_kernel void @{{.*}}Kernel{{.*}}VecBlockedWGTest
    marker(); // CHECK: [[MARKER]] [[# @LINE ]]

    vec<int, VEC_SIZE> out;
    group_load(g, global_mem, out, properties(data_placement<blocked>));
    // CHECK: call spir_func noundef <2 x i32> @_Z30__spirv_SubgroupBlockReadINTELIDv2_jET_PU3AS1Kj(i32 addrspace(1)* noundef

    bool success = true;
    sycl::detail::dim_loop<VEC_SIZE>(
        [&](size_t i) { success &= (out[i] == gid + VEC_SIZE * 2 - i); });
    Record(success);

    marker(); // CHECK: [[MARKER]] [[# @LINE ]]
  }
};

struct VecStripedWGTest {
  static constexpr Scope Scope = WG;
  KERNEL_OP {
    constexpr int VEC_SIZE = 2;

    for (int i = 0; i < VEC_SIZE; ++i)
      global_mem[lid * VEC_SIZE + i] = gid + VEC_SIZE * 2 - i;

    // clang-format off
    // For int2, g_size = 4
    // ...
    // |  8  7 |  9  8 | 10  9 | 11 10 | <= G2
    // | 12 11 | 13 12 | 14 13 | 15 14 | <= G3
    //
    // blocked:
    //   pairs as written.
    // striped:
    //   ( 8 10), ( 7  9), ( 9 11), ( 8 10)
    //   (12 14), (11 13), (13 15), (12 14)

    // For int3, g_size = 8
    //    *                           *                            *
    // |  6  5  4 |  7  6  5 |  8  7  6 |  9  8  7 | 10  9  8 | 11 10  9 | 12 11 10 | 13 12 11 |
    // | 14 13 12 | 15 14 13 | 16 15 14 | 17 16 15 | 18 17 16 | 19 18 17 | 20 19 18 | 21 20 19 |
    //
    // striped:
    //  ( 6  6 10) ( 5  9  9) ( 4 8 12) ( 7 7 11 ) ( 6 10 10) ( 5  9  13) ( 8 8 12 ) ( 7 11 11)
    //
    // idx = group_local_id + vec_idx * G_SIZE
    // val = group_start + idx / VEC_SIZE - idx % VEC_SIZE
    // clang-format on

    group_barrier(g);

    // CHECK-LABEL: define weak_odr dso_local spir_kernel void @{{.*}}Kernel{{.*}}VecStripedWGTest
    marker(); // CHECK: [[MARKER]] [[# @LINE ]]
    vec<int, VEC_SIZE> out;
    group_load(g, global_mem, out, properties(data_placement<striped>));
    // Two block reads because the stride between vec elements of a single WI is
    // wg_size, not sg_size.
    // CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef
    // CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    bool success = true;
    for (int i = 0; i < VEC_SIZE; ++i) {
      int striped_idx = lid + i * wg_size;
      auto expected = ndi.get_group(0) * wg_size + VEC_SIZE * 2 +
                      striped_idx / VEC_SIZE - striped_idx % VEC_SIZE;
      success &= (out[i] == expected);
    }
    Record(success);

    marker(); // CHECK: [[MARKER]] [[# @LINE ]]
  }
};

struct VecBlockedSGTest {
  static constexpr Scope Scope = SG;
  KERNEL_OP {
    constexpr int VEC_SIZE = 2;

    for (int i = 0; i < VEC_SIZE; ++i)
      global_mem[sg_lid * VEC_SIZE + i] = gid + VEC_SIZE * 2 - i;

    group_barrier(g);

    // CHECK-LABEL: define weak_odr dso_local spir_kernel void @{{.*}}Kernel{{.*}}VecBlockedSGTest
    marker(); // CHECK: [[MARKER]] [[# @LINE ]]

    vec<int, VEC_SIZE> out;
    group_load(sg, global_mem, out, properties(data_placement<blocked>));
    // CHECK: call spir_func noundef <2 x i32> @_Z30__spirv_SubgroupBlockReadINTELIDv2_jET_PU3AS1Kj(i32 addrspace(1)* noundef

    bool success = true;
    for (int i = 0; i < VEC_SIZE; ++i)
      success &= (out[i] == gid + VEC_SIZE * 2 - i);

    Record(success);

    marker(); // CHECK: [[MARKER]] [[# @LINE ]]
  }
};

struct VecStripedSGTest {
  static constexpr Scope Scope = SG;
  KERNEL_OP {
    constexpr int VEC_SIZE = 2;
    int sg_size = sg.get_local_range().size();

    for (int i = 0; i < VEC_SIZE; ++i)
      global_mem[sg_lid * VEC_SIZE + i] = gid + VEC_SIZE * 2 - i;

    // clang-format off
    // For int2, g_size = 4
    // ...
    // |  8  7 |  9  8 | 10  9 | 11 10 | <= G2
    // | 12 11 | 13 12 | 14 13 | 15 14 | <= G3
    //
    // blocked:
    //   pairs as written.
    // striped:
    //   ( 8 10), ( 7  9), ( 9 11), ( 8 10)
    //   (12 14), (11 13), (13 15), (12 14)

    // For int3, g_size = 8
    //    *                           *                            *
    // |  6  5  4 |  7  6  5 |  8  7  6 |  9  8  7 | 10  9  8 | 11 10  9 | 12 11 10 | 13 12 11 |
    // | 14 13 12 | 15 14 13 | 16 15 14 | 17 16 15 | 18 17 16 | 19 18 17 | 20 19 18 | 21 20 19 |
    //
    // striped:
    //  ( 6  6 10) ( 5  9  9) ( 4 8 12) ( 7 7 11 ) ( 6 10 10) ( 5  9  13) ( 8 8 12 ) ( 7 11 11)
    //
    // idx = group_local_id + vec_idx * G_SIZE
    // val = group_start + idx / VEC_SIZE - idx % VEC_SIZE
    // clang-format on

    group_barrier(g);

    // CHECK-LABEL: define weak_odr dso_local spir_kernel void @{{.*}}Kernel{{.*}}VecStripedSGTest
    marker(); // CHECK: [[MARKER]] [[# @LINE ]]

    vec<int, VEC_SIZE> out;
    group_load(sg, global_mem, out, properties(data_placement<striped>));
    // CHECK: call spir_func noundef <2 x i32> @_Z30__spirv_SubgroupBlockReadINTELIDv2_jET_PU3AS1Kj(i32 addrspace(1)* noundef

    bool success = true;
    // Make IR dumps more readable by forcing unrolling.
    sycl::detail::dim_loop<VEC_SIZE>([&](size_t i) {
      int striped_idx = sg_lid + i * sg_size;
      auto expected = ndi.get_group(0) * wg_size +
                      // get_max_local_range() assumes particular splitting of
                      // WG into SG which is implementation-defined when WG
                      // isn't divisible by the SIMD size.
                      sg.get_group_id() * sg.get_max_local_range().size() +
                      VEC_SIZE * 2 + striped_idx / VEC_SIZE -
                      striped_idx % VEC_SIZE;
      success &= (out[i] == expected);
    });
    Record(success);

    marker(); // CHECK: [[MARKER]] [[# @LINE ]]
  }
};

struct SpanBlockedWGTest {
  static constexpr Scope Scope = WG;
  KERNEL_OP {
    // TODO: group_helper with scratchpad
    constexpr int SPAN_SIZE = 2;

    for (int i = 0; i < SPAN_SIZE; ++i)
      global_mem[lid * SPAN_SIZE + i] = gid + SPAN_SIZE * 2 - i;

    group_barrier(g);

    // CHECK-LABEL: define weak_odr dso_local spir_kernel void @{{.*}}Kernel{{.*}}SpanBlockedWGTest
    marker(); // CHECK: [[MARKER]] [[# @LINE ]]

    int out_arr[SPAN_SIZE];
    sycl::span<int, SPAN_SIZE> out(out_arr);
    group_load(g, global_mem, out, properties(data_placement<blocked>));
    // Not optimized yet.
    // CHECK-NOT: SubgroupBlockRead

    bool success = true;
    for (int i = 0; i < SPAN_SIZE; ++i)
      success &= (out[i] == gid + SPAN_SIZE * 2 - i);

    Record(success);

    marker(); // CHECK: [[MARKER]] [[# @LINE ]]
  }
};

struct SpanStripedWGTest {
  static constexpr Scope Scope = WG;
  KERNEL_OP {
    // TODO: group_helper with scratchpad
    constexpr int SPAN_SIZE = 2;

    for (int i = 0; i < SPAN_SIZE; ++i)
      global_mem[lid * SPAN_SIZE + i] = gid + SPAN_SIZE * 2 - i;

    // clang-format off
    // For int2, g_size = 4
    // ...
    // |  8  7 |  9  8 | 10  9 | 11 10 | <= G2
    // | 12 11 | 13 12 | 14 13 | 15 14 | <= G3
    //
    // blocked:
    //   pairs as written.
    // striped:
    //   ( 8 10), ( 7  9), ( 9 11), ( 8 10)
    //   (12 14), (11 13), (13 15), (12 14)

    // For int3, g_size = 8
    //    *                           *                            *
    // |  6  5  4 |  7  6  5 |  8  7  6 |  9  8  7 | 10  9  8 | 11 10  9 | 12 11 10 | 13 12 11 |
    // | 14 13 12 | 15 14 13 | 16 15 14 | 17 16 15 | 18 17 16 | 19 18 17 | 20 19 18 | 21 20 19 |
    //
    // striped:
    //  ( 6  6 10) ( 5  9  9) ( 4 8 12) ( 7 7 11 ) ( 6 10 10) ( 5  9  13) ( 8 8 12 ) ( 7 11 11)
    //
    // idx = group_local_id + vec_idx * G_SIZE
    // val = group_start + idx / SPAN_SIZE - idx % SPAN_SIZE
    // clang-format on

    group_barrier(g);

    // CHECK-LABEL: define weak_odr dso_local spir_kernel void @{{.*}}Kernel{{.*}}SpanStripedWGTest
    marker(); // CHECK: [[MARKER]] [[# @LINE ]]

    int out_arr[SPAN_SIZE];
    sycl::span<int, SPAN_SIZE> out(out_arr);
    group_load(g, global_mem, out, properties(data_placement<striped>));
    // Not optimized yet.
    // CHECK-NOT: SubgroupBlockRead

    bool success = true;
    for (int i = 0; i < SPAN_SIZE; ++i) {
      int striped_idx = lid + i * wg_size;
      success &=
          (out[i] == ndi.get_group(0) * wg_size + SPAN_SIZE * 2 +
                         striped_idx / SPAN_SIZE - striped_idx % SPAN_SIZE);
    }
    Record(success);

    marker(); // CHECK: [[MARKER]] [[# @LINE ]]
  }
};

#if 0
struct SpanBlockedSGTest {
  static constexpr Scope Scope = SG;
  KERNEL_OP {}
};

struct SpanStripedSGTest {
  static constexpr Scope Scope = SG;
  KERNEL_OP {}
};
#endif

int main() {
  capture_marker();

  test_ptr_dispatch(SG_SIZE);
  size_t wg_sizes[] = {SG_SIZE / 2, SG_SIZE, SG_SIZE * 3 / 2, SG_SIZE * 3};

  for (auto wg_size : wg_sizes) {
    std::cout << "WG_SIZE: " << wg_size << std::endl;

    test(ScalarWGTest{}, wg_size);
    test(ScalarSGTest{}, wg_size);

    test(VecBlockedWGTest{}, wg_size);
    test(VecStripedWGTest{}, wg_size);
    test(VecBlockedSGTest{}, wg_size);
    test(VecStripedSGTest{}, wg_size);

    test(SpanBlockedWGTest{}, wg_size);
    test(SpanStripedWGTest{}, wg_size);
    // test(SpanBlockedSGTest{}, wg_size);
    // test(SpanStripedSGTest{}, wg_size);
  }

  return 0;
}
