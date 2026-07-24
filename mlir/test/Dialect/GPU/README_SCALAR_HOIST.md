# MLIR GPU Scalar Hoisting — Build & Run Guide

## 1. Overview

This guide documents how to build a full MLIR pipeline capable of running
`gpu.launch` kernels on Intel GPU via the Level Zero runtime, using the
intel/llvm `sycl` branch.

The pipeline enables:
- Writing GPU kernels in MLIR `gpu.launch` dialect (host + device in same IR)
- Lowering through SPIR-V to Intel GPU ISA
- JIT execution on Intel GPU via `mlir-runner` + Level Zero

## 2. Prerequisites

- Intel GPU with Level Zero driver installed
- `libhwloc-dev` (for hwloc)
- `opencl-headers` (for OpenCL headers)
- `libze-dev` (Level Zero headers, usually from Intel GPU driver package)

```bash
sudo apt-get install -y libhwloc-dev opencl-headers
# Level Zero headers should already be at /usr/include/level_zero/ from GPU driver
```

## 3. Source Code

- **Repository**: `https://github.com/intel/llvm.git`
- **Branch**: `sycl`
- **Commit**: `f4ae903c9819` (or latest sycl branch)

```bash
git clone https://github.com/intel/llvm.git
cd llvm
git checkout sycl
```

## 4. Build

```bash
cd /home2/jianyizh/llvm
mkdir build_imex

cmake -G Ninja -B build_imex -S llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;SPIRV" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_LEVELZERO_RUNNER=1

# Build the required tools and runtime libraries
ninja -C build_imex mlir-opt mlir-translate mlir-runner \
  mlir_levelzero_runtime mlir_runner_utils
```

Key CMake options:
| Option | Purpose |
|--------|---------|
| `LLVM_ENABLE_PROJECTS=mlir` | Build MLIR (no clang needed) |
| `LLVM_TARGETS_TO_BUILD=X86;SPIRV` | x86 host + SPIR-V GPU target |
| `MLIR_ENABLE_LEVELZERO_RUNNER=1` | Build Level Zero runtime for JIT GPU execution |

## 5. Verify GPU Access

```bash
# Check GPU is visible via Level Zero
source /opt/intel/oneapi/2026.0/oneapi-vars.sh --force
sycl-ls 2>/dev/null | grep level_zero
```

Expected output:
```
[level_zero:gpu][level_zero:0] Intel(R) Data Center GPU Max 1550 ...
```

## 6. Run Upstream GPU Test (Sanity Check)

```bash
BUILD=/home2/jianyizh/llvm/build_imex

$BUILD/bin/mlir-opt \
  $LLVM_SRC/mlir/test/Integration/GPU/LevelZero/gpu-addf32-to-spirv.mlir \
  -pass-pipeline='builtin.module(
    spirv-attach-target{ver=v1.0 caps=Addresses,Int64,Kernel},
    convert-gpu-to-spirv{use-64bit-index=true},
    gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)),
    func.func(llvm-request-c-wrappers),
    convert-scf-to-cf,
    convert-to-llvm,
    gpu-to-llvm{use-bare-pointers-for-kernels=true},
    gpu-module-to-binary{format=isa},
    expand-strided-metadata,
    lower-affine,
    reconcile-unrealized-casts
  )' \
| LD_LIBRARY_PATH=$BUILD/lib \
  $BUILD/bin/mlir-runner \
  --shared-libs=$BUILD/lib/libmlir_levelzero_runtime.so \
  --shared-libs=$BUILD/lib/libmlir_runner_utils.so \
  --entry-point-result=void

# Expected output:
# Unranked Memref base@ = ... data = [[[2.3, 4.5], [7.8, 10.2]], ...
```

## 7. BiasAdd Benchmark (Runtime Scalar Division)

### Benchmark Source

File: `mlir/test/Dialect/GPU/bias-add-runtime-shape.mlir`

```bash
$BUILD/bin/mlir-opt \
  $LLVM_SRC/mlir/test/Dialect/GPU/bias-add-runtime-shape.mlir \
  -pass-pipeline='builtin.module(
    spirv-attach-target{ver=v1.0 caps=Addresses,Int64,Kernel},
    convert-gpu-to-spirv{use-64bit-index=true},
    gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)),
    func.func(llvm-request-c-wrappers),
    convert-scf-to-cf,
    convert-to-llvm,
    gpu-to-llvm{use-bare-pointers-for-kernels=true},
    gpu-module-to-binary{format=isa},
    expand-strided-metadata,
    lower-affine,
    reconcile-unrealized-casts
  )' \
| LD_LIBRARY_PATH=$BUILD/lib \
  $BUILD/bin/mlir-runner \
  --shared-libs=$BUILD/lib/libmlir_levelzero_runtime.so \
  --shared-libs=$BUILD/lib/libmlir_runner_utils.so \
  --entry-point-result=void

# Expected output:
# Unranked Memref base@ = ... data = [1, 2, 3, 4, 6, 2, 2, 2, 3, 3, 3, 3]
```

### Kernel Structure

```
Host side (func.func @main):
  %tot = arith.constant 12   ← could be any runtime value
  %chw = arith.constant 12   ← scalar divisor #1
  %hw  = arith.constant 4    ← scalar divisor #2
  gpu.launch_func @bias_add_kernel args(..., %tot, %chw, %hw)

Kernel side (gpu.func @bias_add_kernel):
  %chw_i32 = arith.index_castui %chw : index to i32
  %hw_i32  = arith.index_castui %hw  : index to i32
  %rem = arith.remui %i, %chw_i32 : i32    ← target: division by scalar arg
  %ch  = arith.divui %rem, %hw_i32 : i32   ← target: division by scalar arg
```

### SPIR-V Lowering

The division ops lower to SPIR-V:
```mlir
%7 = spirv.UMod %2, %5 : i32       # i % chw
%8 = spirv.UDiv %7, %6 : i32       # (i%chw) / hw
```

Where `%5` and `%6` are scalar kernel arguments (chw, hw converted to i32).

## 8. Optimization Target

The scalar hoisting pass should:
1. Identify `arith.divui`/`arith.remui` ops in `gpu.func` bodies where the
   divisor is a scalar kernel argument (uniform across all work-items).
2. In the host function (before `gpu.launch_func`), insert magic/shift
   precomputation for each scalar divisor.
3. Add magic/shift as new kernel arguments.
4. Replace kernel-side division with magic multiply:
   `(mul_hi(magic, n) + n) >> shift`
5. Update `gpu.launch_func` call args to include magic/shift values.

## 9. MLIR Pass Pipeline Map

```
gpu.launch (host + device)
  │
  ├─ spirv-attach-target     ← adds SPIR-V target to gpu.module
  ├─ convert-gpu-to-spirv    ← lowers gpu ops → SPIR-V dialect
  │    └─ gpu.module:
  │         ├─ spirv-lower-abi-attrs
  │         └─ spirv-update-vce
  ├─ func.func:
  │    └─ llvm-request-c-wrappers
  ├─ convert-scf-to-cf       ← scf → cf
  ├─ convert-to-llvm         ← remaining dialects → LLVM
  ├─ gpu-to-llvm             ← gpu.launch_func → LLVM calls
  ├─ gpu-module-to-binary    ← SPIR-V + target → embedded binary
  ├─ expand-strided-metadata
  ├─ lower-affine
  └─ reconcile-unrealized-casts
         │
         ▼
  LLVM dialect (with gpu.binary embedded)
         │
    mlir-runner + Level Zero → GPU execution
```