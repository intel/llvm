// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Running like this manually for the time being:
// clang++ -fsycl %s &&  ./a.out && \
// clang++ -fsycl %s -S -emit-llvm -fsycl-device-only -o - | FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;

using sycl::ext::oneapi::experimental::group_load;
using sycl::ext::oneapi::experimental::group_store;

constexpr int WG_SIZE = 32;
constexpr int N_WGS = 2;
constexpr int GLOBAL_SIZE = WG_SIZE * N_WGS;
constexpr int N_RESULTS = 128;
// Not greater than 8 vec/array size + gap between WGs.
constexpr int ELEMS_PER_WI = 8 * 2;

__attribute__((noinline)) __attribute__((optnone)) void
marker(int l = __builtin_LINE()) {
  std::ignore = l;
}

template <typename KernelName, typename FuncTy> void test(FuncTy Func) {
  queue q;
  buffer<bool, 1> results(N_RESULTS * GLOBAL_SIZE);
  {
    host_accessor res_acc{results};
    for (auto &res : res_acc)
      res = true;
  }

  buffer<int, 1> global_mem_buf(GLOBAL_SIZE * ELEMS_PER_WI);
  auto *usm_mem = malloc_device<int>(GLOBAL_SIZE * ELEMS_PER_WI, q);

  q.submit([&](handler &cgh) {
    accessor res_acc{results, cgh};
    accessor global_mem_acc{global_mem_buf, cgh};
    local_accessor<int, 1> local_mem_acc{WG_SIZE * ELEMS_PER_WI, cgh};
    cgh.parallel_for<KernelName>(
        nd_range{range{GLOBAL_SIZE}, range{WG_SIZE}}, [=](nd_item<1> ndi) {
          // Low-level pointer arithmetic and skipping 0th index to make llvm IR
          // dumps nicer.
          auto *res_ptr = &res_acc[ndi.get_global_id(0) * N_RESULTS];
          auto Record = [&](bool val) {
            res_ptr++;
            *res_ptr = val;
          };

          // TODO: Use accessor::get_pointer once it starts returing raw
          // pointer.
          auto *global_mem =
              global_mem_acc.get_multi_ptr<access::decorated::no>().get_raw();
          auto *local_mem =
              local_mem_acc.get_multi_ptr<access::decorated::no>().get_raw();
          Func(ndi, global_mem, usm_mem, local_mem, Record);
        });
  });
  {
    host_accessor res_acc{results};
    if constexpr (false) {
      for (int i = 1; i < N_RESULTS; ++i) {
        int errors = 0;
        for (int j = 0; j < GLOBAL_SIZE; ++j) {
          std::cout << " " << res_acc[j * N_RESULTS + i];
        }
        std::cout << std::endl;
      }
    }
    for (int i = 1; i < N_RESULTS; ++i) {
      int errors = 0;
      for (int j = 0; j < GLOBAL_SIZE; ++j) {
        if (!res_acc[j * N_RESULTS + i]) {
          ++errors;
          if (errors < 5)
            std::cout << "Error in condition " << i << " for work item " << j
                      << std::endl;
        }
      }
    }
    assert(std::find(res_acc.begin(), res_acc.end(), false) == res_acc.end());
  }
  free(usm_mem, q);
}

int main() {
  std::cout << "ScalarWGKernel" << std::endl;
  test<class ScalarWGKernel>([](auto ndi, auto global_mem, auto usm_mem,
                                auto local_mem, auto Record) {
    // Make groups non-contiguous.
    int group_offset = ndi.get_group(0) * WG_SIZE * 2;
    int local_offset = ndi.get_local_id(0);

    int init = ndi.get_global_id(0);
    global_mem[group_offset + local_offset] = init;
    usm_mem[group_offset + local_offset] = init;
    local_mem[local_offset] = init;

    auto g = ndi.get_group();

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      int out;
      group_load(g, Input, out);
      Record(out == init);
    };

    Check(global_mem + group_offset);
    Check(usm_mem + group_offset);
    Check(local_mem);
    // Optimizer would optimize all of the following to a single load (instead
    // of three loads) but that still suffices for the purposes of this test. If
    // someone wants to read IR dumps and have all three operations done
    // independently, a call to marker() can be inserted in between Checks.
    Check(
        address_space_cast<access::address_space::global_space,
                           access::decorated::yes>(global_mem + group_offset));
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::no>(global_mem + group_offset));
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::yes>(global_mem + group_offset)
              .get_decorated());
  });

  std::cout << "ScalarSGKernel" << std::endl;
  test<class ScalarSGKernel>([](auto ndi, auto global_mem, auto usm_mem,
                                auto local_mem, auto Record) {
    auto sg = ndi.get_sub_group();

    // Make groups non-contiguous.
    int wg_offset = ndi.get_group(0) * WG_SIZE * 2;
    int sg_offset = sg.get_group_id() * sg.get_local_range() * 2;
    int group_offset = wg_offset + sg_offset;
    int local_offset = sg.get_local_id();

    int init = ndi.get_global_id(0);
    global_mem[group_offset + local_offset] = init;
    usm_mem[group_offset + local_offset] = init;
    local_mem[sg_offset + local_offset] = init;

    int out;

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      group_load(sg, Input, out);
      Record(out == init);
    };

    // CHECK: define weak_odr dso_local spir_kernel {{.*}}ScalarSGKernel
    Check(global_mem + group_offset); // Dynamic address space dispatch.
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(usm_mem + group_offset); // Dynamic address space dispatch.
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(local_mem + sg_offset);
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    // CHECK: call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(
        address_space_cast<access::address_space::global_space,
                           access::decorated::yes>(global_mem + group_offset));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 3]]
    // CHECK-NOT: br
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::no>(global_mem + group_offset));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 2]]
    // CHECK-NOT: br
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::yes>(global_mem + group_offset)
              .get_decorated());
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 3]]
    // CHECK-NOT: br
    // CHECK: call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    Check(address_space_cast<access::address_space::local_space,
                             access::decorated::yes>(local_mem + sg_offset));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 2]]
    // CHECK-NOT: SubgroupBlockRead

    Check(address_space_cast<access::address_space::local_space,
                             access::decorated::no>(local_mem + sg_offset));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 2]]
    // CHECK-NOT: SubgroupBlockRead

    Check(address_space_cast<access::address_space::local_space,
                             access::decorated::yes>(local_mem + sg_offset)
              .get_decorated());
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 3]]
    // CHECK-NOT: SubgroupBlockRead

    marker(); // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE ]]
  });

  std::cout << "VecWGKernel" << std::endl;
  test<class VecWGKernel>([](auto ndi, auto global_mem, auto usm_mem,
                             auto local_mem, auto Record) {
    constexpr int VEC_SIZE = 2;
    // Make groups non-contiguous.
    int group_offset = ndi.get_group(0) * WG_SIZE * 4;
    int local_offset = ndi.get_local_id(0) * VEC_SIZE;

    int init = ndi.get_global_id(0) + VEC_SIZE * 2;
    for (int i = 0; i < VEC_SIZE; ++i) {
      global_mem[group_offset + local_offset + i] = init - i;
      usm_mem[group_offset + local_offset + i] = init - i;
      local_mem[local_offset + i] = init - i;
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
      vec<int, VEC_SIZE> out, out_blocked, out_striped;
      using namespace sycl::ext::oneapi::experimental;
      using namespace sycl::ext::oneapi::experimental::property;
      auto blocked =
          properties(data_placement<group_algorithm_data_placement::blocked>);
      auto striped =
          properties(data_placement<group_algorithm_data_placement::striped>);
      group_load(g, Input, out);
      group_load(g, Input, out_blocked, blocked);
      group_load(g, Input, out_striped, striped);

      bool success = true;
      for (int i = 0; i < VEC_SIZE; ++i) {
        success &= (out[i] == out_blocked[i]);
        success &= (out_blocked[i] == init - i);
        int striped_idx = ndi.get_local_id(0) + i * WG_SIZE;
        success &= (out_striped[i] ==
                    ndi.get_group(0) * WG_SIZE + VEC_SIZE * 2 +
                        striped_idx / VEC_SIZE - striped_idx % VEC_SIZE);
      }
      Record(success);
    };

    Check(global_mem + group_offset);
    Check(usm_mem + group_offset);
    Check(local_mem);
    Check(
        address_space_cast<access::address_space::global_space,
                           access::decorated::yes>(global_mem + group_offset));
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::no>(global_mem + group_offset));
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::yes>(global_mem + group_offset)
              .get_decorated());
  });

  std::cout << "VecBlockedSGKernel" << std::endl;
  test<class VecBlockedSGKernel>([](auto ndi, auto global_mem, auto usm_mem,
                                    auto local_mem, auto Record) {
    auto sg = ndi.get_sub_group();

    constexpr int VEC_SIZE = 2;
    // Make groups non-contiguous.
    int wg_offset = ndi.get_group(0) * WG_SIZE * 4;
    int sg_size = sg.get_local_range().size();
    int sg_offset = sg.get_group_id() * sg_size * 4;
    int group_offset = wg_offset + sg_offset;
    int local_offset = sg.get_local_id() * VEC_SIZE;

    int init = ndi.get_global_id(0) + VEC_SIZE * 2;
    for (int i = 0; i < VEC_SIZE; ++i) {
      global_mem[group_offset + local_offset + i] = init - i;
      usm_mem[group_offset + local_offset + i] = init - i;
      local_mem[sg_offset + local_offset + i] = init - i;
    }

    auto Check = [&](auto Input, int l = __builtin_LINE()) {
      marker(l);
      vec<int, VEC_SIZE> out_blocked;
      using namespace sycl::ext::oneapi::experimental;
      using namespace sycl::ext::oneapi::experimental::property;
      auto blocked =
          properties(data_placement<group_algorithm_data_placement::blocked>);
      group_load(sg, Input, out_blocked, blocked);

      bool success = true;
      for (int i = 0; i < VEC_SIZE; ++i) {
        success &= (out_blocked[i] == init - i);
      }
      Record(success);
    };

    Check(global_mem + group_offset);
    Check(usm_mem + group_offset);
    Check(local_mem);
    Check(
        address_space_cast<access::address_space::global_space,
                           access::decorated::yes>(global_mem + group_offset));
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::no>(global_mem + group_offset));
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::yes>(global_mem + group_offset)
              .get_decorated());
  });

  std::cout << "VecStripedSGKernel" << std::endl;
  test<class VecStripedSGKernel>([](auto ndi, auto global_mem, auto usm_mem,
                                    auto local_mem, auto Record) {
    auto sg = ndi.get_sub_group();

    constexpr int VEC_SIZE = 2;
    // Make groups non-contiguous.
    int wg_offset = ndi.get_group(0) * WG_SIZE * 4;
    int sg_size = sg.get_local_range().size();
    int sg_offset = sg.get_group_id() * sg_size * 4;
    int group_offset = wg_offset + sg_offset;
    int local_offset = sg.get_local_id() * VEC_SIZE;

    int init = ndi.get_global_id(0) + VEC_SIZE * 2;
    for (int i = 0; i < VEC_SIZE; ++i) {
      global_mem[group_offset + local_offset + i] = init - i;
      usm_mem[group_offset + local_offset + i] = init - i;
      local_mem[sg_offset + local_offset + i] = init - i;
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
      vec<int, VEC_SIZE> out_striped;
      using namespace sycl::ext::oneapi::experimental;
      using namespace sycl::ext::oneapi::experimental::property;
      auto striped =
          properties(data_placement<group_algorithm_data_placement::striped>);
      group_load(sg, Input, out_striped, striped);

      bool success = true;
      // Make IR dumps more readable by forcing unrolling.
      sycl::detail::dim_loop<VEC_SIZE>([&](size_t i) {
        int striped_idx = sg.get_local_id() + i * sg_size;
        success &= (out_striped[i] ==
                    ndi.get_group(0) * WG_SIZE + sg.get_group_id() * sg_size +
                        VEC_SIZE * 2 + striped_idx / VEC_SIZE -
                        striped_idx % VEC_SIZE);
      });
      Record(success);
    };

    Check(global_mem + group_offset);
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]
    Check(usm_mem + group_offset);
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 1]]

    // Check(local_mem); // FIXME: Why does it fail?
    // check: call {{.*}}marker

    Check(
        address_space_cast<access::address_space::global_space,
                           access::decorated::yes>(global_mem + group_offset));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 3]]
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::no>(global_mem + group_offset));
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 2]]
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::yes>(global_mem + group_offset)
              .get_decorated());
    // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE - 3]]
    marker(); // CHECK: call spir_func void @_Z6markeri(i32 noundef [[# @LINE ]]
    return;
  });

  std::cout << "SpanWGKernel" << std::endl;
  test<class SpanWGKernel>([](auto ndi, auto global_mem, auto usm_mem,
                              auto local_mem, auto Record) {
    constexpr int SPAN_SIZE = 2;
    // Make groups non-contiguous.
    int group_offset = ndi.get_group(0) * WG_SIZE * 4;
    int local_offset = ndi.get_local_id(0) * SPAN_SIZE;

    int init = ndi.get_global_id(0) + SPAN_SIZE * 2;
    for (int i = 0; i < SPAN_SIZE; ++i) {
      global_mem[group_offset + local_offset + i] = init - i;
      usm_mem[group_offset + local_offset + i] = init - i;
      local_mem[local_offset + i] = init - i;
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
      int out_arr[SPAN_SIZE], out_blocked_arr[SPAN_SIZE],
          out_striped_arr[SPAN_SIZE];
      sycl::span<int, SPAN_SIZE> out(out_arr), out_blocked(out_blocked_arr),
          out_striped(out_striped_arr);
      using namespace sycl::ext::oneapi::experimental;
      using namespace sycl::ext::oneapi::experimental::property;
      auto blocked =
          properties(data_placement<group_algorithm_data_placement::blocked>);
      auto striped =
          properties(data_placement<group_algorithm_data_placement::striped>);
      group_load(g, Input, out);
      group_load(g, Input, out_blocked, blocked);
      group_load(g, Input, out_striped, striped);

      bool success = true;
      for (int i = 0; i < SPAN_SIZE; ++i) {
        success &= (out[i] == out_blocked[i]);
        success &= (out_blocked[i] == init - i);
        int striped_idx = ndi.get_local_id(0) + i * WG_SIZE;
        success &= (out_striped[i] ==
                    ndi.get_group(0) * WG_SIZE + SPAN_SIZE * 2 +
                        striped_idx / SPAN_SIZE - striped_idx % SPAN_SIZE);
      }
      Record(success);
    };

    Check(global_mem + group_offset);
    Check(usm_mem + group_offset);
    Check(local_mem);
    Check(
        address_space_cast<access::address_space::global_space,
                           access::decorated::yes>(global_mem + group_offset));
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::no>(global_mem + group_offset));
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::yes>(global_mem + group_offset)
              .get_decorated());
  });

  return 0;
}
