/// ########################################################################

/// Check the phases for SYCL compilation using new offload model
// RUN: %clangxx -ccc-print-phases -fsycl --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHK-PHASES %s
// CHK-PHASES:       0: input, "[[INPUT:.*]]", c++, (host-sycl)
// CHK-PHASES-NEXT:  1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-NEXT:  2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NEXT:  3: compiler, {2}, ir, (host-sycl)
// CHK-PHASES-NEXT:  4: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-NEXT:  5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-PHASES-NEXT:  6: compiler, {5}, ir, (device-sycl)
// CHK-PHASES-NEXT:  7: backend, {6}, assembler, (device-sycl)
// CHK-PHASES-NEXT:  8: assembler, {7}, object, (device-sycl)
// CHK-PHASES-NEXT:  9: offload, "device-sycl (spir64-unknown-unknown)" {8}, object
// CHK-PHASES-NEXT: 10: clang-offload-packager, {9}, image, (device-sycl)
// CHK-PHASES-NEXT: 11: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (x86_64-unknown-linux-gnu)" {10}, ir
// CHK-PHASES-NEXT: 12: backend, {11}, assembler, (host-sycl)
// CHK-PHASES-NEXT: 13: assembler, {12}, object, (host-sycl)
// CHK-PHASES-NEXT: 14: clang-linker-wrapper, {13}, image, (host-sycl)

/// ########################################################################

/// Check the toolflow for SYCL compilation using new offload model
// RUN: %clangxx -### -fsycl --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHK-FLOW %s
// CHK-FLOW: "[[PATH:.*]]/clang-18" "-cc1" "-triple" "spir64-unknown-unknown" "-aux-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" {{.*}} "-fsycl-int-header=[[HEADER:.*]].h" "-fsycl-int-footer=[[FOOTER:.*]].h" {{.*}} "--offload-new-driver" {{.*}} "-o" "[[CC1DEVOUT:.*]]" "-x" "c++" "[[INPUT:.*]]"
// CHK-FLOW-NEXT: "[[PATH]]/clang-offload-packager" "-o" "[[PACKOUT:.*]]" "--image=file=[[CC1DEVOUT]],triple=spir64-unknown-unknown,arch=,kind=sycl"
// CHK-FLOW-NEXT: "[[PATH]]/append-file" "[[INPUT]]" "--append=[[FOOTER]].h" "--orig-filename=[[INPUT]]" "--output=[[APPENDOUT:.*]]" "--use-include"
// CHK-FLOW-NEXT: "[[PATH]]/clang-18" "-cc1" "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[HEADER]].h" "-dependency-filter" "[[HEADER]].h" {{.*}} "-fsycl-is-host"{{.*}} "-full-main-file-name" "[[INPUT]]" {{.*}} "--offload-new-driver" {{.*}} "-fembed-offload-object=[[PACKOUT]]" {{.*}} "-o" "[[CC1FINALOUT:.*]]" "-x" "c++" "[[APPENDOUT]]"
// CHK-FLOW-NEXT: "[[PATH]]/clang-linker-wrapper" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" {{.*}} "[[CC1FINALOUT]]"

/// Check for no crashes for SYCL compilation using new offload model
/// TODO(NOM4): Enable when driver support is available
// RUNXXX: %clangxx -fsycl --offload-new-driver %s

/// Check for no crashes for standalone clang-linker-wrapper run for sycl
/// TODO(NOM5): Enable when driver support is available
// RUNXXX: clang-linker-wrapper -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.o,libsycl-complex.o,libsycl-complex-fp64.o,libsycl-cmath.o,libsycl-cmath-fp64.o,libsycl-imf.o,libsycl-imf-fp64.o,libsycl-imf-bf16.o,libsycl-fallback-cassert.o,libsycl-fallback-cstring.o,libsycl-fallback-complex.o,libsycl-fallback-complex-fp64.o,libsycl-fallback-cmath.o,libsycl-fallback-cmath-fp64.o,libsycl-fallback-imf.o,libsycl-fallback-imf-fp64.o,libsycl-fallback-imf-bf16.o,libsycl-itt-user-wrappers.o,libsycl-itt-compiler-wrappers.o,libsycl-itt-stubs.o -sycl-post-link-options="-split=auto -emit-param-info -symbols -emit-exported-symbols -split-esimd -lower-esimd -O2 -spec-const=native -device-globals" -llvm-spirv-options="-spirv-max-version=1.4 -spirv-debug-info-version=ocl-100 -spirv-allow-extra-diexpressions -spirv-allow-unknown-intrinsics=llvm.genx. -spirv-ext=-all,+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr,+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io,+SPV_INTEL_device_side_avc_motion_estimation,+SPV_INTEL_fpga_loop_controls,+SPV_INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg,+SPV_INTEL_blocking_pipes,+SPV_INTEL_function_pointers,+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers,+SPV_INTEL_float_controls2,+SPV_INTEL_vector_compute,+SPV_INTEL_fast_composite,+SPV_INTEL_arbitrary_precision_fixed_point,+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode,+SPV_INTEL_long_constant_composite,+SPV_INTEL_arithmetic_fence,+SPV_INTEL_global_variable_decorations,+SPV_INTEL_fpga_buffer_location,+SPV_INTEL_fpga_argument_interfaces,+SPV_INTEL_fpga_invocation_pipelining_attributes,+SPV_INTEL_fpga_latency_control,+SPV_INTEL_token_type,+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_matrix,+SPV_INTEL_hw_thread_queries,+SPV_KHR_uniform_group_instructions,+SPV_INTEL_masked_gather_scatter,+SPV_INTEL_tensor_float32_conversion" --host-triple=x86_64-unknown-linux-gnu --wrapper-verbose --triple=spir64 --linker-path=/usr/bin/ld -- -z relro --hash-style=gnu --eh-frame-hdr -m elf_x86_64 -dynamic-linker /lib64/ld-linux-x86-64.so.2 -o %s.out %S/Inputs/test-sycl-new-offload.o

#include "Inputs/sycl.hpp"

int main(void) {
  sycl::queue queue;
  sycl::event event = queue.submit([&](sycl::handler &h) {
    h.single_task<class test_kernel1>(
        [=]() {
          // Empty
        });
  });
  return 0;
}
