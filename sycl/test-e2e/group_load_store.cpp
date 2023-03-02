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
int main() {
  queue q;
  buffer<bool, 2> results(range{N_RESULTS, GLOBAL_SIZE});
  {
    host_accessor res_acc{results};
    for (auto &res : res_acc)
      res = true;
  }

  buffer<char, 1> global_mem_buf(GLOBAL_SIZE * BYTES_PER_WI);
  char *usm_mem_allocation = malloc_device<char>(GLOBAL_SIZE * BYTES_PER_WI, q);
  q.submit([&](handler &cgh) {
    accessor res_acc{results, cgh};
    accessor global_mem_acc{global_mem_buf, cgh};
    local_accessor<char, 1> local_mem_acc{WG_SIZE * BYTES_PER_WI, cgh};
    // CHECK: define weak_odr dso_local spir_kernel
    cgh.parallel_for(
        nd_range{range{GLOBAL_SIZE}, range{WG_SIZE}}, [=](nd_item<1> ndi) {
          int result_index = 0;
          auto Record = [&](bool val) {
            res_acc[id<2>(result_index++, ndi.get_global_id(0))] = val;
          };

          // TODO: Use accessor::get_pointer once it starts returing raw
          // pointer.
          auto *global_mem = reinterpret_cast<int *>(
              global_mem_acc.get_multi_ptr<access::decorated::no>().get_raw());
          auto *usm_mem = reinterpret_cast<int *>(usm_mem_allocation);
          auto *local_mem = reinterpret_cast<int *>(
              local_mem_acc.get_multi_ptr<access::decorated::no>().get_raw());

          {
            // Work-group scope tests.

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
            Check(address_space_cast<access::address_space::global_space,
                                     access::decorated::yes>(global_mem +
                                                             group_offset));
            Check(address_space_cast<access::address_space::global_space,
                                     access::decorated::no>(global_mem +
                                                            group_offset));
            Check(address_space_cast<access::address_space::global_space,
                                     access::decorated::yes>(global_mem +
                                                             group_offset)
                      .get_decorated());
          }
          {
            // Sub-group scope tests.
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
            Check(address_space_cast<access::address_space::global_space,
                                     access::decorated::yes>(global_mem +
                                                             group_offset));
            // CHECK-NOT: br
            // CHECK: tail call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

            marker(marker_var); // CHECK: call {{.*}}marker
            Check(address_space_cast<access::address_space::global_space,
                                     access::decorated::no>(global_mem +
                                                            group_offset));
            // CHECK-NOT: br
            // CHECK: tail call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

            marker(marker_var); // CHECK: call {{.*}}marker
            Check(address_space_cast<access::address_space::global_space,
                                     access::decorated::yes>(global_mem +
                                                             group_offset)
                      .get_decorated());
            Record(marker_var != 0);
            // CHECK-NOT: br
            // CHECK: tail call spir_func {{.*}} @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)* noundef

            marker(marker_var); // CHECK: call {{.*}}marker
            Check(address_space_cast<access::address_space::local_space,
                                     access::decorated::yes>(local_mem +
                                                             sg_offset));
            // CHECK-NOT: SubgroupBlockRead

            marker(marker_var); // CHECK: call {{.*}}marker
            Check(address_space_cast<access::address_space::local_space,
                                     access::decorated::no>(local_mem +
                                                            sg_offset));
            // CHECK-NOT: SubgroupBlockRead

            marker(marker_var); // CHECK: call {{.*}}marker
            Check(address_space_cast<access::address_space::local_space,
                                     access::decorated::yes>(local_mem +
                                                             sg_offset)
                      .get_decorated());
            // CHECK-NOT: SubgroupBlockRead
          }
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

  return 0;
}
