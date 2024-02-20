// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test appropriate !spirv.Decorations is applied to each fpga_mem.
// Make sure the mapping from property to SPIR-V decoration is correct

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental;   // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

const intel::fpga_mem<int[10]> empty{};
const intel::fpga_mem<int[10], decltype(oneapi::properties(
                                   intel::ram_stitching_min_ram))>
    min_ram{};
const intel::fpga_mem<int[10], decltype(oneapi::properties(
                                   intel::ram_stitching_max_fmax))>
    max_fmax{};
const intel::fpga_mem<int[10],
                      decltype(oneapi::properties(intel::clock_2x_true))>
    double_pumped{};
const intel::fpga_mem<int[10],
                      decltype(oneapi::properties(intel::clock_2x_false))>
    single_pumped{};
const intel::fpga_mem<int[10],
                      decltype(oneapi::properties(intel::resource_mlab))>
    mlab{};
const intel::fpga_mem<int[10], decltype(oneapi::properties(
                                   intel::bi_directional_ports_false))>
    simple_dual_port{};
const intel::fpga_mem<int[10], decltype(oneapi::properties(
                                   intel::bi_directional_ports_true))>
    true_dual_port{};
const intel::fpga_mem<int[10],
                      decltype(oneapi::properties(intel::resource_block_ram))>
    block_ram{};
const intel::fpga_mem<int[10],
                      decltype(oneapi::properties(intel::num_banks<4>))>
    banks{};
const intel::fpga_mem<int[10],
                      decltype(oneapi::properties(intel::stride_size<2>))>
    stride{};
const intel::fpga_mem<int[10],
                      decltype(oneapi::properties(intel::word_size<8>))>
    word{};
const intel::fpga_mem<int[10], decltype(oneapi::properties(
                                   intel::max_private_copies<3>))>
    copies{};
const intel::fpga_mem<int[10],
                      decltype(oneapi::properties(intel::num_replicates<5>))>
    replicates{};

// CHECK: {{.*}}empty = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[empty_md:[0-9]*]]
// CHECK: {{.*}}min_ram = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[min_ram_md:[0-9]*]]
// CHECK: {{.*}}max_fmax = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[max_fmax_md:[0-9]*]]
// CHECK: {{.*}}double_pumped = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[double_pumped_md:[0-9]*]]
// CHECK: {{.*}}single_pumped = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[single_pumped_md:[0-9]*]]
// CHECK: {{.*}}mlab = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[mlab_md:[0-9]*]]
// CHECK: {{.*}}simple_dual_port = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[simple_dual_port_md:[0-9]*]]
// CHECK: {{.*}}true_dual_port = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[true_dual_port_md:[0-9]*]]
// CHECK: {{.*}}block_ram = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[block_ram_md:[0-9]*]]
// CHECK: {{.*}}banks = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[banks_md:[0-9]*]]
// CHECK: {{.*}}stride = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[stride_md:[0-9]*]]
// CHECK: {{.*}}word = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[word_md:[0-9]*]]
// CHECK: {{.*}}copies = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[copies_md:[0-9]*]]
// CHECK: {{.*}}replicates = internal addrspace(1) constant {{.*}} zeroinitializer, align 4, !spirv.Decorations ![[replicates_md:[0-9]*]]

int main() {
  queue Q;
  int f = 5;

  Q.single_task([=]() {
    volatile int ReadVal = empty[f] + min_ram[f] + max_fmax[f] +
                           double_pumped[f] + single_pumped[f] + mlab[f] +
                           simple_dual_port[f] + true_dual_port[f] +
                           block_ram[f] + banks[f] + stride[f] + word[f] +
                           copies[f] + replicates[f];
  });
  return 0;
}

// CHECK: ![[empty_md]] = !{![[mem_default:[0-9]*]]}
// CHECK: ![[mem_default]] = !{i32 5826, !"DEFAULT"}
// CHECK: ![[min_ram_md]] = !{![[min_ram:[0-9]*]], ![[mem_default]]}
// CHECK: ![[min_ram]] = !{i32 5836, i32 0}
// CHECK: ![[max_fmax_md]] = !{![[max_fmax:[0-9]*]], ![[mem_default]]}
// CHECK: ![[max_fmax]] = !{i32 5836, i32 1}
// CHECK: ![[double_pumped_md]] = !{![[double_pumped:[0-9]*]], ![[mem_default]]}
// CHECK: ![[double_pumped]] = !{i32 5831}
// CHECK: ![[single_pumped_md]] = !{![[single_pumped:[0-9]*]], ![[mem_default]]}
// CHECK: ![[single_pumped]] = !{i32 5830}
// CHECK: ![[mlab_md]] = !{![[mlab:[0-9]*]]}
// CHECK: ![[mlab]] = !{i32 5826, !"MLAB"}
// CHECK: ![[simple_dual_port_md]] = !{![[simple_dual_port:[0-9]*]], ![[mem_default]]}
// CHECK: ![[simple_dual_port]] = !{i32 5833}
// CHECK: ![[true_dual_port_md]] = !{![[true_dual_port:[0-9]*]], ![[mem_default]]}
// CHECK: ![[true_dual_port]] = !{i32 5885}
// CHECK: ![[block_ram_md]] = !{![[block_ram:[0-9]*]]}
// CHECK: ![[block_ram]] = !{i32 5826, !"BLOCK_RAM"}
// CHECK: ![[banks_md]] = !{![[banks:[0-9]*]], ![[mem_default]]}
// CHECK: ![[banks]] = !{i32 5827, i32 4}
// CHECK: ![[stride_md]] = !{![[mem_default]], ![[stride:[0-9]*]]}
// CHECK: ![[stride]] = !{i32 5883, i32 2}
// CHECK: ![[word_md]] = !{![[mem_default]], ![[word:[0-9]*]]}
// CHECK: ![[word]] = !{i32 5884, i32 8}
// CHECK: ![[copies_md]] = !{![[copies:[0-9]*]], ![[mem_default]]}
// CHECK: ![[copies]] = !{i32 5829, i32 3}
// CHECK: ![[replicates_md]] = !{![[replicates:[0-9]*]], ![[mem_default]]}
// CHECK: ![[replicates]] = !{i32 5832, i32 5}
