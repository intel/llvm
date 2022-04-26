set(obj_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
# FIXME: Other location.
set(bc_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if (WIN32)
  set(lib-suffix obj)
  set(spv_binary_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  set(install_dest_spv bin)
else()
  set(lib-suffix o)
  set(spv_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  set(install_dest_spv lib${LLVM_LIBDIR_SUFFIX})
endif()
set(install_dest_lib lib${LLVM_LIBDIR_SUFFIX})
set(clang $<TARGET_FILE:clang>)
set(llvm-link $<TARGET_FILE:llvm-link>)
set(llvm-spirv $<TARGET_FILE:llvm-spirv>)


string(CONCAT sycl_targets_opt
  "-fsycl-targets="
  "spir64_x86_64-unknown-unknown,"
  "spir64_gen-unknown-unknown,"
  "spir64_fpga-unknown-unknown,"
  "spir64-unknown-unknown")

set(compile_opts
  # suppress an error about SYCL_EXTERNAL being used for
  # a function with a raw pointer parameter.
  -Wno-sycl-strict
  # Disable warnings for the host compilation, where
  # we declare all functions as 'static'.
  -Wno-undefined-internal
  -sycl-std=2020
  )

if (WIN32)
  list(APPEND compile_opts -D_ALLOW_RUNTIME_LIBRARY_MISMATCH)
  list(APPEND compile_opts -D_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
endif()

add_custom_target(libsycldevice-obj)
add_custom_target(libsycldevice-spv)

add_custom_target(libsycldevice DEPENDS
  libsycldevice-obj
  libsycldevice-spv)

function(add_devicelib_obj obj_filename)
  cmake_parse_arguments(OBJ  "" "" "SRC;DEP" ${ARGN})
  set(devicelib-obj-file ${obj_binary_dir}/${obj_filename}.${lib-suffix})
  add_custom_command(OUTPUT ${devicelib-obj-file}
                     COMMAND ${clang} -fsycl -c
                             ${compile_opts} ${sycl_targets_opt}
                             ${CMAKE_CURRENT_SOURCE_DIR}/${OBJ_SRC}
                             -o ${devicelib-obj-file}
                     MAIN_DEPENDENCY ${OBJ_SRC}
                     DEPENDS ${OBJ_DEP}
                     VERBATIM)
  set(devicelib-obj-target ${obj_filename}-obj)
  add_custom_target(${devicelib-obj-target} DEPENDS ${devicelib-obj-file})
  add_dependencies(libsycldevice-obj ${devicelib-obj-target})
  install(FILES ${devicelib-obj-file}
          DESTINATION ${install_dest_lib}
          COMPONENT libsycldevice)
endfunction()

function(add_devicelib_bc bc_filename)
  cmake_parse_arguments(BC  "" "" "SRC;DEP" ${ARGN})
  set(devicelib-bc-file ${bc_binary_dir}/${bc_filename}.bc)
  add_custom_command(OUTPUT ${devicelib-bc-file}
                     COMMAND ${clang} -fsycl-device-only -fsycl-use-bitcode
                             ${compile_opts}
                             ${CMAKE_CURRENT_SOURCE_DIR}/${BC_SRC}
                             -o ${devicelib-bc-file}
                     MAIN_DEPENDENCY ${BC_SRC}
                     DEPENDS ${BC_DEP}
                     VERBATIM)
  set(devicelib-bc-target ${bc_filename}-bc)
  add_custom_target(${devicelib-bc-target} DEPENDS ${devicelib-bc-file})
endfunction()


function(add_devicelib_spv spv_filename)
  add_devicelib_bc(${spv_filename} ${ARGN})

  cmake_parse_arguments(SPV  "" "" "SRC;DEP" ${ARGN})
  set(devicelib-bc-file ${bc_binary_dir}/${spv_filename}.bc)
  set(devicelib-linked-bc-file ${bc_binary_dir}/${spv_filename}.linked.bc)
  set(devicelib-spv-file ${spv_binary_dir}/${spv_filename}.spv)
  add_custom_command(OUTPUT ${devicelib-spv-file}
                     COMMAND ${llvm-link} ${devicelib-bc-file} ${bc_binary_dir}/libsycl-itt-user-wrappers.bc -o ${devicelib-linked-bc-file}
                     COMMAND ${llvm-spirv} ${devicelib-linked-bc-file}
                             -spirv-max-version=1.4
                             -spirv-debug-info-version=ocl-100
                             -spirv-allow-extra-diexpressions
                             -spirv-allow-unknown-intrinsics=llvm.genx.
                             -spirv-ext=-all,+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr,+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io,+SPV_INTEL_device_side_avc_motion_estimation,+SPV_INTEL_fpga_loop_controls,+SPV_INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg,+SPV_INTEL_blocking_pipes,+SPV_INTEL_function_pointers,+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers,+SPV_INTEL_float_controls2,+SPV_INTEL_vector_compute,+SPV_INTEL_fast_composite,+SPV_INTEL_arbitrary_precision_fixed_point,+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode,+SPV_INTEL_long_constant_composite,+SPV_INTEL_arithmetic_fence,+SPV_INTEL_token_type,+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_matrix,+SPV_INTEL_hw_thread_queries,+SPV_KHR_uniform_group_instructions
                             -o ${devicelib-spv-file}
                     DEPENDS ${spv_filename}-bc libsycl-itt-user-wrappers-bc
                     VERBATIM)
  set(devicelib-spv-target ${spv_filename}-spv)
  add_custom_target(${devicelib-spv-target} DEPENDS ${devicelib-spv-file})
  add_dependencies(libsycldevice-spv ${devicelib-spv-target})
  install(FILES ${devicelib-spv-file}
          DESTINATION ${install_dest_spv}
          COMPONENT libsycldevice)
endfunction()

function(add_fallback_devicelib fallback_filename)
  cmake_parse_arguments(FB "" "" "SRC;DEP" ${ARGN})
  add_devicelib_spv(${fallback_filename} SRC ${FB_SRC} DEP ${FB_DEP})
  add_devicelib_obj(${fallback_filename} SRC ${FB_SRC} DEP ${FB_DEP})
endfunction()

set(crt_obj_deps wrapper.h device.h spirv_vars.h sycl-compiler)
set(complex_obj_deps device_complex.h device.h sycl-compiler)
set(cmath_obj_deps device_math.h device.h sycl-compiler)
set(itt_obj_deps device_itt.h spirv_vars.h device.h sycl-compiler)

add_devicelib_obj(libsycl-itt-stubs SRC itt_stubs.cpp DEP ${itt_obj_deps})
add_devicelib_obj(libsycl-itt-compiler-wrappers SRC itt_compiler_wrappers.cpp DEP ${itt_obj_deps})
add_devicelib_obj(libsycl-itt-user-wrappers SRC itt_user_wrappers.cpp DEP ${itt_obj_deps})
add_devicelib_bc(libsycl-itt-user-wrappers SRC itt_user_wrappers.cpp DEP ${itt_obj_deps})

add_devicelib_obj(libsycl-crt SRC crt_wrapper.cpp DEP ${crt_obj_deps})
add_devicelib_obj(libsycl-complex SRC complex_wrapper.cpp DEP ${complex_obj_deps})
add_devicelib_obj(libsycl-complex-fp64 SRC complex_wrapper_fp64.cpp DEP ${complex_obj_deps} )
add_devicelib_obj(libsycl-cmath SRC cmath_wrapper.cpp DEP ${cmath_obj_deps})
add_devicelib_obj(libsycl-cmath-fp64 SRC cmath_wrapper_fp64.cpp DEP ${cmath_obj_deps} )

add_fallback_devicelib(libsycl-fallback-cassert SRC fallback-cassert.cpp DEP ${crt_obj_deps})
add_fallback_devicelib(libsycl-fallback-cstring SRC fallback-cstring.cpp DEP ${crt_obj_deps})
add_fallback_devicelib(libsycl-fallback-complex SRC fallback-complex.cpp DEP ${complex_obj_deps})
add_fallback_devicelib(libsycl-fallback-complex-fp64 SRC fallback-complex-fp64.cpp DEP ${complex_obj_deps} )
add_fallback_devicelib(libsycl-fallback-cmath SRC fallback-cmath.cpp DEP ${cmath_obj_deps})
add_fallback_devicelib(libsycl-fallback-cmath-fp64 SRC fallback-cmath-fp64.cpp DEP ${cmath_obj_deps})
