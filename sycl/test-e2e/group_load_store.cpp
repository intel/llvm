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
constexpr int BYTES_PER_WI = 64;

__attribute__((noinline)) void marker(int &x) { ++x; }

template <typename KernelName, typename FuncTy> void test(FuncTy Func) {
  queue q;
  buffer<bool, 2> results(range{N_RESULTS, GLOBAL_SIZE});
  {
    host_accessor res_acc{results};
    for (auto &res : res_acc)
      res = true;
  }

  buffer<char, 1> global_mem_buf(GLOBAL_SIZE * BYTES_PER_WI);
  char *usm_mem = malloc_device<char>(GLOBAL_SIZE * BYTES_PER_WI, q);

  q.submit([&](handler &cgh) {
    accessor res_acc{results, cgh};
    accessor global_mem_acc{global_mem_buf, cgh};
    local_accessor<char, 1> local_mem_acc{WG_SIZE * BYTES_PER_WI, cgh};
    cgh.parallel_for<KernelName>(
        nd_range{range{GLOBAL_SIZE}, range{WG_SIZE}}, [=](nd_item<1> ndi) {
          int result_index = 0;
          auto Record = [&](bool val) {
            res_acc[id<2>(result_index++, ndi.get_global_id(0))] = val;
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
      for (int i = 0; i < N_RESULTS; ++i) {
        int errors = 0;
        for (int j = 0; j < GLOBAL_SIZE; ++j) {
          std::cout << " " << res_acc[id<2>(i, j)];
        }
        std::cout << std::endl;
      }
    }
    for (int i = 0; i < N_RESULTS; ++i) {
      int errors = 0;
      for (int j = 0; j < GLOBAL_SIZE; ++j) {
        if (!res_acc[id<2>(i, j)]) {
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
  test<class ScalarWGKernel>([](auto ndi, auto global_ptr, auto usm_ptr,
                                auto local_ptr, auto Record) {
    auto *global_mem = reinterpret_cast<int *>(global_ptr);
    auto *usm_mem = reinterpret_cast<int *>(usm_ptr);
    auto *local_mem = reinterpret_cast<int *>(local_ptr);

    // Make groups non-contiguous.
    int group_offset = ndi.get_group(0) * WG_SIZE * 2;
    int local_offset = ndi.get_local_id(0);

    int init = ndi.get_global_id(0);
    global_mem[group_offset + local_offset] = init;
    usm_mem[group_offset + local_offset] = init;
    local_mem[local_offset] = init;

    int out;
    auto g = ndi.get_group();

    auto Check = [&](auto Input) {
      group_load(g, Input, out);
      Record(out == init);
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

  std::cout << "ScalarSGKernel" << std::endl;
  test<class ScalarSGKernel>([](auto ndi, auto global_ptr, auto usm_ptr,
                                auto local_ptr, auto Record) {
    // CHECK: define weak_odr dso_local spir_kernel {{.*}}ScalarSGKernel
    auto *global_mem = reinterpret_cast<int *>(global_ptr);
    auto *usm_mem = reinterpret_cast<int *>(usm_ptr);
    auto *local_mem = reinterpret_cast<int *>(local_ptr);

    int marker_var = 0;
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

    auto Check = [&](auto Input) {
      group_load(sg, Input, out);
      Record(out == init);
    };

    marker(marker_var);               // CHECK: call {{.*}}marker
    Check(global_mem + group_offset); // Dynamic address space dispatch.
    // CHECK: tail call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: tail call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    marker(marker_var);            // CHECK: call {{.*}}marker
    Check(usm_mem + group_offset); // Dynamic address space dispatch.
    // CHECK: tail call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: tail call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    marker(marker_var); // CHECK: call {{.*}}marker
    Check(local_mem + sg_offset);
    // CHECK: tail call spir_func {{.*}} @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
    // CHECK: tail call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    marker(marker_var); // CHECK: call {{.*}}marker
    Check(
        address_space_cast<access::address_space::global_space,
                           access::decorated::yes>(global_mem + group_offset));
    // CHECK-NOT: br
    // CHECK: tail call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    marker(marker_var); // CHECK: call {{.*}}marker
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::no>(global_mem + group_offset));
    // CHECK-NOT: br
    // CHECK: tail call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    marker(marker_var); // CHECK: call {{.*}}marker
    marker(marker_var); // CHECK: call {{.*}}marker
    Check(address_space_cast<access::address_space::global_space,
                             access::decorated::yes>(global_mem + group_offset)
              .get_decorated());
    // CHECK-NOT: br
    // CHECK: tail call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

    marker(marker_var); // CHECK: call {{.*}}marker
    Check(address_space_cast<access::address_space::local_space,
                             access::decorated::yes>(local_mem + sg_offset));
    // CHECK-NOT: SubgroupBlockRead

    marker(marker_var); // CHECK: call {{.*}}marker
    Check(address_space_cast<access::address_space::local_space,
                             access::decorated::no>(local_mem + sg_offset));
    // CHECK-NOT: SubgroupBlockRead

    marker(marker_var); // CHECK: call {{.*}}marker
    Check(address_space_cast<access::address_space::local_space,
                             access::decorated::yes>(local_mem + sg_offset)
              .get_decorated());
    // CHECK-NOT: SubgroupBlockRead

    marker(marker_var); // CHECK: call {{.*}}marker
    Record(marker_var != 0);
  });

  std::cout << "VecWGKernel" << std::endl;
  test<class VecWGKernel>([](auto ndi, auto global_ptr, auto usm_ptr,
                             auto local_ptr, auto Record) {
    auto *global_mem = reinterpret_cast<int *>(global_ptr);
    auto *usm_mem = reinterpret_cast<int *>(usm_ptr);
    auto *local_mem = reinterpret_cast<int *>(local_ptr);

    constexpr int VEC_SIZE = 2;
    // Make groups non-contiguous.
    int group_offset = ndi.get_group(0) * WG_SIZE * 4;
    int local_offset = ndi.get_local_id(0) * VEC_SIZE;

    int init = ndi.get_global_id(0);
    for (int i = 0; i < VEC_SIZE; ++i) {
      global_mem[group_offset + local_offset + i] = init + i;
      usm_mem[group_offset + local_offset + i] = init + i;
      local_mem[local_offset + i] = init + i;
    }

    auto g = ndi.get_group();

    auto Check = [&](auto Input) {
      int2 out, out_blocked; // Must be in sync with VEC_SIZE.
      group_load(g, Input, out);
      using namespace sycl::ext::oneapi::experimental;
      using namespace sycl::ext::oneapi::experimental::property;
      auto props =
          properties(data_placement<group_algorithm_data_placement::blocked>);
      group_load(g, Input, out_blocked, props);

      bool success = true;
      for (int i = 0; i < VEC_SIZE; ++i) {
        success &= (out[i] == init + i);
        // success &= (out_blocked[i] == out[i]);
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
