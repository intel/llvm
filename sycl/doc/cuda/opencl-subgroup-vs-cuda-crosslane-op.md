# CUDA crosslane vs OpenCL sub-groups

## Sub-group function mapping
This document describes the mapping of the SYCL subgroup operations (based on the proposal SYCL subgroup proposal) to CUDA (queries responses and PTX instruction mapping)

### Sub-group device Queries

| Query                                                  | CUDA backend result                           |
| ---------------                                        | -------------------------                     |
| `info::device::max_num_sub_groups`                     | sm 3.0 to 7.0: 64; sm 7.5 32  (see [HW_spec]) |
| `info::device::sub_group_independent_forward_progress` | `true`                                        |
| `info::device::sub_group_sizes`                        | {32}                                          |

### Sub-group function mapping


| Sub-group function        | PTX mapping               | LLVM Intrinsic                      | Min version     | Note                                                                                     |
| ---------------           | ------------------------- | -------------                       | --------------- | ---------------                                                                          |
| `get_local_id()`          | `%laneid`                 | `@llvm.nvvm.read.ptx.sreg.laneid`   |                 |                                                                                          |
| `get_local_range()`       | `WARP_SZ`                 | `@llvm.nvvm.read.ptx.sreg.warpsize` |                 |                                                                                          |
| `get_max_local_range`     | `WARP_SZ`                 | `@llvm.nvvm.read.ptx.sreg.warpsize` |                 |                                                                                          |
| `get_group_id`            | `%warpid`                 | `@llvm.nvvm.read.ptx.sreg.warpid`   |                 |                                                                                          |
| `get_group_range`         | `%nwarpid`                | `@llvm.nvvm.read.ptx.sreg.nwarpid`  |                 |                                                                                          |
| `get_uniform_group_range` | `%nwarpid`                | `@llvm.nvvm.read.ptx.sreg.nwarpid`  |                 |                                                                                          |
| `barrier`                 | `bar.warp.sync`           | `@llvm.nvvm.bar.warp.sync`          |                 |                                                                                          |
| `any(bool)`               | `vote{.sync}.any.pred`    | `llvm.nvvm.vote.any{.sync}`         |                 |                                                                                          |
| `all(bool)`               | `vote{.sync}.all.pred`    | `llvm.nvvm.vote.all{.sync}`         |                 |                                                                                          |
| `broadcast`               | `shfl.sync.idx.b32`       | `llvm.shfl.sync.idx.{f32,i32}`      | `sm_30`         | Only implemented for float and int32 in LLVM but should extendable                       |
| `reduce`                  | None                      | None                                |                 | [cuda_reduce]                                                                            |
| `exclusive_scan`          | None                      | None                                |                 | [cuda_scan_example]/[ptx_scan_example]                                                   |
| `inclusive_scan`          | None                      | None                                |                 | [cuda_scan_example]/[ptx_scan_example]                                                   |
| `shuffle`                 | `shfl.sync.idx.b32`       | `llvm.shfl.sync.idx.{f32,i32}`      | `sm_30`         | Insn only for 32 bits. Requires emulation for non 32-bits.                               |
| `shuffle_down`            | `shfl.sync.down.b32`      | `llvm.shfl.sync.down.{f32,i32}`     | `sm_30`         | Insn only for 32 bits. Requires emulation for non 32-bits.                               |
| `shuffle_up`              | `shfl.sync.up.b32`        | `llvm.shfl.sync.up.{f32,i32}`       | `sm_30`         | Insn only for 32 bits. Requires emulation for non 32-bits.                               |
| `shuffle_xor`             | `shfl.sync.bfly.b32`      | `llvm.shfl.sync.bfly.{f32,i32}`     | `sm_30`         | Insn only for 32 bits. Requires emulation for non 32-bits.                               |
| `shuffle` (2 inputs)      | None                      | None                                |                 | Can be implemented using CUDA shuffle function (non in-place modification + predication) |
| `shuffle_down` (2 inputs) | None                      | None                                |                 | Can be implemented using CUDA shuffle function (non in-place modification + predication) |
| `shuffle_up` (2 inputs)   | None                      | None                                |                 | Can be implemented using CUDA shuffle function (non in-place modification + predication) |
| `load` (scalar)           | None                      | None                                |                 | Maps to normal load, guarantees coalesced access                                         |
| `load` (vector)           | None                      | None                                |                 | Maps to normal load, guarantees coalesced access                                         |
| `store` (scalar)          | None                      | None                                |                 | Maps to normal store, guarantees coalesced access                                        |
| `store` (vector)          | None                      | None                                |                 | Maps to normal store, guarantees coalesced access                                        |



[cuda_reduce]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-examples-reduction
[ptx_scan_example]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl
[cuda_scan_example]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-examples
[HW_spec]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
