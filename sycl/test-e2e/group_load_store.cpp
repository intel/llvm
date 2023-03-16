// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Running like this manually for the time being:
// clang++ -fsycl %s &&  ./a.out && \
// clang++ -fsycl %s -S -emit-llvm -fsycl-device-only -o - | FileCheck %s

#include <sycl/sycl.hpp>
#include <iomanip>

using namespace sycl;

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::property;
using namespace access;

// Using-enum-declaration is C++20.
constexpr auto global_space = address_space::global_space;
constexpr auto local_space = address_space::local_space;
constexpr auto blocked = group_algorithm_data_placement::blocked;
constexpr auto striped = group_algorithm_data_placement::striped;

constexpr int WG_SIZE = 16;
constexpr int SG_SIZE = 16;

constexpr int N_WGS = 1;
constexpr int GLOBAL_SIZE = WG_SIZE * N_WGS;
// Not greater than 8 vec/array size + gap between WGs.
constexpr int ELEMS_PER_WI = 8 * 2;

__attribute__((noinline)) __attribute__((optnone)) void
marker(int l = __builtin_LINE()) {
  std::ignore = l;
}

enum Scope { WG, SG };

template <typename KernelName, Scope S, bool Disable = false, typename FuncTy>
void test(FuncTy Func, int Line = __builtin_LINE()) {
  if constexpr (Disable)
    return;
  std::cout << Line << std::endl;
  queue q;
  constexpr int N_RESULTS = 16;
  buffer<int, 1> results(N_RESULTS * GLOBAL_SIZE);
  {
    host_accessor res_acc{results};
    for (auto &res : res_acc)
      res = true;
  }

  buffer<int, 1> global_mem_buf(GLOBAL_SIZE * ELEMS_PER_WI);
  auto *usm_mem_alloc = malloc_device<int>(GLOBAL_SIZE * ELEMS_PER_WI, q);

  q.submit([&](handler &cgh) {
    accessor res_acc{results, cgh};
    accessor global_mem_acc{global_mem_buf, cgh};
    local_accessor<int, 1> local_mem_acc{WG_SIZE * ELEMS_PER_WI, cgh};
    cgh.parallel_for<KernelName>(
        nd_range{range{GLOBAL_SIZE}, range{WG_SIZE}},
        [=](nd_item<1> ndi) {
          // Low-level pointer arithmetic and skipping 0th index to make llvm IR
          // dumps nicer.
          auto *res_ptr = &res_acc[ndi.get_global_id(0) * N_RESULTS];
          auto Record = [&](auto val) {
            res_ptr++;
            *res_ptr = val;
          };

          // TODO: Use accessor::get_pointer once it starts returing raw
          // pointer.
          auto *global_mem =
              global_mem_acc.get_multi_ptr<decorated::no>().get_raw();
          auto *local_mem =
              local_mem_acc.get_multi_ptr<decorated::no>().get_raw();

          // Each WG receives its own "local" sub-region.
          int wg_offset = WG_SIZE * ELEMS_PER_WI * ndi.get_group(0);

          auto sg = ndi.get_sub_group();
          int sg_offset = 0;
          if constexpr (S == SG)
            sg_offset =
                sg.get_max_local_range() * ELEMS_PER_WI * sg.get_group_id();

          auto *usm_mem = usm_mem_alloc + wg_offset + sg_offset;
          global_mem += wg_offset + sg_offset;
          local_mem += sg_offset;

#define ARGS                                                                   \
  auto ndi, auto global_mem, auto usm_mem, auto local_mem, auto Record, auto lid
          Func(ndi, global_mem, usm_mem, local_mem, Record, ndi.get_local_id());
        });
  });
  {
    host_accessor res_acc{results};
    if constexpr (true) {
      for (int i = 1; i < N_RESULTS; ++i) {
        int errors = 0;
        bool all_same = [&]() {
          auto val = res_acc[i];
          for (int j = 0; j < GLOBAL_SIZE; ++j) {
            if (val != res_acc[j * N_RESULTS + i])
              return false;
          }
          return true;
        }();
        if (all_same) {
          std::cout << "All: " << res_acc[i] << std::endl;
          continue;
        }
        for (int j = 0; j < GLOBAL_SIZE; ++j) {
          std::cout << " " << std::setw(3) << res_acc[j * N_RESULTS + i];
        }
        std::cout << std::endl;
      }
    }
    for (int i = 1; i < N_RESULTS; ++i) {
      int errors = 0;
      for (int j = 0; j < GLOBAL_SIZE; ++j) {
        if (!res_acc[j * N_RESULTS + i]) {
          ++errors;
          if (errors < 2)
            std::cout << "Error in condition " << i << " for work item " << j
                      << std::endl;
        }
      }
    }
    assert(std::find(res_acc.begin(), res_acc.end(), false) == res_acc.end());
  }
  free(usm_mem_alloc, q);
}

int main() {
  std::cout << "ScalarWGKernel" << std::endl;
  test<class ScalarWGKernel, WG, true>([](ARGS) {
    int init = ndi.get_global_id(0);

    global_mem[lid] = init;
    usm_mem[lid] = init;
    local_mem[lid] = init;

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      int out;
      group_load(ndi.get_group(), Input, out);
      Record(out == init);
    };

    // CHECK: define weak_odr dso_local spir_kernel {{.*}}ScalarWGKernel
    Check(global_mem); // Dynamic address space dispatch.
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(usm_mem); // Dynamic address space dispatch.
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(local_mem);
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<global_space, decorated::yes>(global_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK-NOT: br
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<global_space, decorated::no>(global_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK-NOT: br
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<global_space, decorated::yes>(global_mem)
              .get_decorated());
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 2]]
    // CHECK-NOT: br
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<local_space, decorated::yes>(local_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK-NOT: SubgroupBlockRead

    Check(address_space_cast<local_space, decorated::no>(local_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK-NOT: SubgroupBlockRead

    Check(address_space_cast<local_space, decorated::yes>(local_mem)
              .get_decorated());
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 2]]
    // CHECK-NOT: SubgroupBlockRead

    marker(); // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE ]]
  });

  std::cout << "ScalarSGKernel" << std::endl;
  test<class ScalarSGKernel, SG, true>([](ARGS) {
    auto sg = ndi.get_sub_group();

    int init = ndi.get_global_id(0);

    global_mem[lid] = init;
    usm_mem[lid] = init;
    local_mem[lid] = init;

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      int out;
      group_load(sg, Input, out);
      Record(out == init);
    };

    // CHECK: define weak_odr dso_local spir_kernel {{.*}}ScalarSGKernel
    Check(global_mem); // Dynamic address space dispatch.
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(usm_mem); // Dynamic address space dispatch.
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(local_mem);
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<global_space, decorated::yes>(global_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK-NOT: br
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<global_space, decorated::no>(global_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK-NOT: br
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<global_space, decorated::yes>(global_mem)
              .get_decorated());
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 2]]
    // CHECK-NOT: br
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<local_space, decorated::yes>(local_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK-NOT: SubgroupBlockRead

    Check(address_space_cast<local_space, decorated::no>(local_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK-NOT: SubgroupBlockRead

    Check(address_space_cast<local_space, decorated::yes>(local_mem)
              .get_decorated());
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 2]]
    // CHECK-NOT: SubgroupBlockRead

    marker(); // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE ]]
  });

  std::cout << "VecBlockedWGKernel" << std::endl;
  test<class VecBlockedWGKernel, WG, true>([](ARGS) {
    constexpr int VEC_SIZE = 2;

    int init = ndi.get_global_id(0) + VEC_SIZE * 2;

    for (int i = 0; i < VEC_SIZE; ++i) {
      int idx = lid * VEC_SIZE + i;
      global_mem[idx] = init - i;
      usm_mem[idx] = init - i;
      local_mem[idx] = init - i;
    }

    auto g = ndi.get_group();

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      vec<int, VEC_SIZE> out;
      group_load(g, Input, out, properties(data_placement<blocked>));

      bool success = true;
      for (int i = 0; i < VEC_SIZE; ++i) {
        success &= (out[i] == init - i);
      }
      Record(success);
    };

    Check(global_mem);
    Check(usm_mem);
    Check(local_mem);
    Check(address_space_cast<global_space, decorated::yes>(global_mem));
    Check(address_space_cast<global_space, decorated::no>(global_mem));
    Check(address_space_cast<global_space, decorated::yes>(global_mem)
              .get_decorated());
  });

  std::cout << "VecStripedWGKernel" << std::endl;
  test<class VecStripedWGKernel, WG, true>([](ARGS) {
    constexpr int VEC_SIZE = 2;

    int init = ndi.get_global_id(0) + VEC_SIZE * 2;

    for (int i = 0; i < VEC_SIZE; ++i) {
      int idx = lid * VEC_SIZE + i;
      global_mem[idx] = init - i;
      usm_mem[idx] = init - i;
      local_mem[idx] = init - i;
    }

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

    auto g = ndi.get_group();

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      vec<int, VEC_SIZE> out;
      group_load(g, Input, out, properties(data_placement<striped>));

      bool success = true;
      for (int i = 0; i < VEC_SIZE; ++i) {
        int striped_idx = ndi.get_local_id(0) + i * WG_SIZE;
        success &=
            (out[i] == ndi.get_group(0) * WG_SIZE + VEC_SIZE * 2 +
                           striped_idx / VEC_SIZE - striped_idx % VEC_SIZE);
      }
      Record(success);
    };

    Check(global_mem);
    Check(usm_mem);
    Check(local_mem);
    Check(address_space_cast<global_space, decorated::yes>(global_mem));
    Check(address_space_cast<global_space, decorated::no>(global_mem));
    Check(address_space_cast<global_space, decorated::yes>(global_mem)
              .get_decorated());
  });

  std::cout << "VecBlockedSGKernel" << std::endl;
  test<class VecBlockedSGKernel, SG>([](ARGS) {
    constexpr int VEC_SIZE = 2;

    int init = ndi.get_global_id(0) + VEC_SIZE * 2;
    for (int i = 0; i < VEC_SIZE; ++i) {
      int idx = lid * VEC_SIZE + i;
      global_mem[idx] = init - i;
      usm_mem[idx] = init - i;
      local_mem[idx] = init - i;
    }

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      vec<int, VEC_SIZE> out;
      group_load(ndi.get_sub_group(), Input, out,
                 properties(data_placement<blocked>));

      for (int i = 0; i < VEC_SIZE; ++i) {
        Record(global_mem[i * WG_SIZE + lid]);
      }

      Record(42);
      for (int i = 0; i < VEC_SIZE; ++i) {
        Record(out[i]);
      }
      Record(43);
      for (int i = 0; i < VEC_SIZE; ++i) {
        Record(init - i);
      }

      bool success = true;
      for (int i = 0; i < VEC_SIZE; ++i) {
        success &= (out[i] == init - i);
      }
      Record(success);
    };

    Check(global_mem);
    return;
    Check(usm_mem);
    Check(local_mem);
    Check(address_space_cast<global_space, decorated::yes>(global_mem));
    Check(address_space_cast<global_space, decorated::no>(global_mem));
    Check(address_space_cast<global_space, decorated::yes>(global_mem)
              .get_decorated());
  });

  return 0;
  std::cout << "VecStripedSGKernel" << std::endl;
  test<class VecStripedSGKernel, SG>([](ARGS) {
    auto sg = ndi.get_sub_group();

    constexpr int VEC_SIZE = 2;
    int sg_size = sg.get_local_range().size();

    int init = ndi.get_global_id(0) + VEC_SIZE * 2;

    for (int i = 0; i < VEC_SIZE; ++i) {
      int idx = lid * VEC_SIZE + i;
      global_mem[idx] = init - i;
      usm_mem[idx] = init - i;
      local_mem[idx] = init - i;
    }

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

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      vec<int, VEC_SIZE> out;
      group_load(sg, Input, out, properties(data_placement<striped>));

      bool success = true;
      // Make IR dumps more readable by forcing unrolling.
      sycl::detail::dim_loop<VEC_SIZE>([&](size_t i) {
        int striped_idx = sg.get_local_id() + i * sg_size;
        success &=
            (out[i] == ndi.get_group(0) * WG_SIZE +
                           sg.get_group_id() * sg_size + VEC_SIZE * 2 +
                           striped_idx / VEC_SIZE - striped_idx % VEC_SIZE);
      });
      Record(success);
    };

    Check(global_mem);
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    Check(usm_mem);
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]

    // Check(local_mem); // FIXME: Why does it fail?
    // check: call {{.*}}marker

    Check(address_space_cast<global_space, decorated::yes>(global_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    Check(address_space_cast<global_space, decorated::no>(global_mem));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    Check(address_space_cast<global_space, decorated::yes>(global_mem)
              .get_decorated());
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 2]]
    marker(); // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE ]]
    return;
  });

  return 0;
  std::cout << "SpanBlockedWGKernel" << std::endl;
  test<class SpanBlockedWGKernel, WG>([](ARGS) {
    constexpr int SPAN_SIZE = 2;

    int init = ndi.get_global_id(0) + SPAN_SIZE * 2;

    for (int i = 0; i < SPAN_SIZE; ++i) {
      int idx = lid * SPAN_SIZE + i;
      global_mem[idx] = init - i;
      usm_mem[idx] = init - i;
      local_mem[idx] = init - i;
    }

    // TODO: group_helper with scratchpad
    auto g = ndi.get_group();

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      int out_arr[SPAN_SIZE];
      sycl::span<int, SPAN_SIZE> out(out_arr);
      group_load(g, Input, out, properties(data_placement<blocked>));

      bool success = true;
      for (int i = 0; i < SPAN_SIZE; ++i) {
        success &= (out[i] == init - i);
      }
      Record(success);
    };

    Check(global_mem);
    Check(usm_mem);
    Check(local_mem);
    Check(address_space_cast<global_space, decorated::yes>(global_mem));
    Check(address_space_cast<global_space, decorated::no>(global_mem));
    Check(address_space_cast<global_space, decorated::yes>(global_mem)
              .get_decorated());
  });

  std::cout << "SpanStripedWGKernel" << std::endl;
  test<class SpanStripedWGKernel, WG>([](ARGS) {
    constexpr int SPAN_SIZE = 2;

    int init = ndi.get_global_id(0) + SPAN_SIZE * 2;

    for (int i = 0; i < SPAN_SIZE; ++i) {
      int idx = lid * SPAN_SIZE + i;
      global_mem[idx] = init - i;
      usm_mem[idx] = init - i;
      local_mem[idx] = init - i;
    }

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

    // TODO: group_helper with scratchpad
    auto g = ndi.get_group();

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      int out_arr[SPAN_SIZE];
      sycl::span<int, SPAN_SIZE> out(out_arr);
      group_load(g, Input, out, properties(data_placement<striped>));

      bool success = true;
      for (int i = 0; i < SPAN_SIZE; ++i) {
        int striped_idx = ndi.get_local_id(0) + i * WG_SIZE;
        success &=
            (out[i] == ndi.get_group(0) * WG_SIZE + SPAN_SIZE * 2 +
                           striped_idx / SPAN_SIZE - striped_idx % SPAN_SIZE);
      }
      Record(success);
    };

    Check(global_mem);
    Check(usm_mem);
    Check(local_mem);
    Check(address_space_cast<global_space, decorated::yes>(global_mem));
    Check(address_space_cast<global_space, decorated::no>(global_mem));
    Check(address_space_cast<global_space, decorated::yes>(global_mem)
              .get_decorated());
  });

  return 0;
}
