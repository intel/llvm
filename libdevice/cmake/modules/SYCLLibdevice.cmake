set(obj_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if (WIN32)
  set(lib-suffix obj)
  set(spv_binary_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  set(host_offload_target "host-x86_64-pc-windows-msvc")
else()
  set(lib-suffix o)
  set(spv_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  set(host_offload_target "host-x86_64-unknown-linux-gnu")
endif()
set(clang $<TARGET_FILE:clang>)
set(bundler $<TARGET_FILE:clang-offload-bundler>)
set(llvm-link $<TARGET_FILE:llvm-link>)
set(llvm-spirv $<TARGET_FILE:llvm-spirv>)

string(CONCAT bundler_targets_opt
  "-targets="
  "sycl-spir64_x86_64-unknown-unknown,"
  "sycl-spir64_gen-unknown-unknown,"
  "sycl-spir64_fpga-unknown-unknown,"
  "sycl-spir64-unknown-unknown,"
  ${host_offload_target})
string(CONCAT bundler_inputs_opt
  "-inputs="
  ${obj_binary_dir} "/fallback-cassert_spir64_x86_64." ${lib-suffix} ","
  ${obj_binary_dir} "/fallback-cassert_spir64_gen." ${lib-suffix} ","
  ${obj_binary_dir} "/fallback-cassert_spir64_fpga." ${lib-suffix} ","
  ${obj_binary_dir} "/fallback-cassert_spir64." ${lib-suffix} ","
  ${obj_binary_dir} "/assert_no_read_host." ${lib-suffix})


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
  # Force definition of CL_SYCL_LANGUAGE_VERSION, as long as
  # SYCL specific code is guarded by it.
  -sycl-std=2017
  )

if (WIN32)
  list(APPEND compile_opts -D_ALLOW_RUNTIME_LIBRARY_MISMATCH)
endif()

set(devicelib-obj-file ${obj_binary_dir}/libsycl-crt.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-file}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/crt_wrapper.cpp
                           -o ${devicelib-obj-file}
                   MAIN_DEPENDENCY crt_wrapper.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)

set(devicelib-obj-complex ${obj_binary_dir}/libsycl-complex.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-complex}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/complex_wrapper.cpp
                           -o ${devicelib-obj-complex}
                   MAIN_DEPENDENCY complex_wrapper.cpp
                   DEPENDS device_complex.h device.h sycl-compiler
                   VERBATIM)

set(devicelib-obj-complex-fp64 ${obj_binary_dir}/libsycl-complex-fp64.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-complex-fp64}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/complex_wrapper_fp64.cpp
                           -o ${devicelib-obj-complex-fp64}
                   MAIN_DEPENDENCY complex_wrapper_fp64.cpp
                   DEPENDS device_complex.h device.h sycl-compiler
                   VERBATIM)

set(devicelib-obj-cmath ${obj_binary_dir}/libsycl-cmath.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-cmath}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/cmath_wrapper.cpp
                           -o ${devicelib-obj-cmath}
                   MAIN_DEPENDENCY cmath_wrapper.cpp
                   DEPENDS device_math.h device.h sycl-compiler
                   VERBATIM)

set(devicelib-obj-cmath-fp64 ${obj_binary_dir}/libsycl-cmath-fp64.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-cmath-fp64}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/cmath_wrapper_fp64.cpp
                           -o ${devicelib-obj-cmath-fp64}
                   MAIN_DEPENDENCY cmath_wrapper_fp64.cpp
                   DEPENDS device_math.h device.h sycl-compiler
                   VERBATIM)

################################################################################
#[[
add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cassert_no_read.bc
                   COMMAND ${clang} -fsycl-device-only -emit-llvm
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o
                           ${spv_binary_dir}/libsycl-fallback-cassert_no_read.bc
                   MAIN_DEPENDENCY fallback-cassert.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${spv_binary_dir}/assert_read_spir64_spv.bc
                   COMMAND ${clang} -c -x cl -emit-llvm
                           --target=spir64-unknown-unknown -cl-std=CL2.0
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                           -o ${spv_binary_dir}/assert_read_spir64_spv.bc
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                   DEPENDS sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cassert.bc
                   COMMAND ${llvm-link} -o
                           ${spv_binary_dir}/libsycl-fallback-cassert.bc
                           ${spv_binary_dir}/libsycl-fallback-cassert_no_read.bc
                           ${spv_binary_dir}/assert_read_spir64_spv.bc
                   DEPENDS ${spv_binary_dir}/libsycl-fallback-cassert_no_read.bc
                           ${spv_binary_dir}/assert_read_spir64_spv.bc
                           llvm-link
                   VERBATIM)
add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cassert.spv
                   COMMAND ${llvm-spirv} -o
                           ${spv_binary_dir}/libsycl-fallback-cassert.spv
                           ${spv_binary_dir}/libsycl-fallback-cassert.bc
                   DEPENDS ${spv_binary_dir}/libsycl-fallback-cassert.bc
                           llvm-spirv
                   VERBATIM)
]]


add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cassert.bc
                   COMMAND ${clang} -fsycl-device-only -emit-llvm
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o
                           ${spv_binary_dir}/libsycl-fallback-cassert.bc
                   MAIN_DEPENDENCY fallback-cassert.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cassert.spv
                   COMMAND ${llvm-spirv} -o
                           ${spv_binary_dir}/libsycl-fallback-cassert.spv
                           ${spv_binary_dir}/libsycl-fallback-cassert.bc
                   DEPENDS ${spv_binary_dir}/libsycl-fallback-cassert.bc
                           llvm-spirv
                   VERBATIM)
#[[
add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cassert.spv
                   COMMAND ${clang} -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-cassert.spv
                   MAIN_DEPENDENCY fallback-cassert.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)
]]
################################################################################

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cstring.spv
                   COMMAND ${clang} -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cstring.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-cstring.spv
                   MAIN_DEPENDENCY fallback-cstring.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)

################################################################################
#[[
add_custom_command(OUTPUT ${obj_binary_dir}/assert_read_spir64_x86_64.bc
                   COMMAND ${clang} -c -x cl -emit-llvm
                           --target=spir64_x86_64-unknown-unknown -cl-std=CL2.0
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                           -o ${obj_binary_dir}/assert_read_spir64_x86_64.bc
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                   DEPENDS sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/assert_read_spir64_gen.bc
                   COMMAND ${clang} -c -x cl -emit-llvm
                           --target=spir64_gen-unknown-unknown -cl-std=CL2.0
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                           -o ${obj_binary_dir}/assert_read_spir64_gen.bc
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                   DEPENDS sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/assert_read_spir64_fpga.bc
                   COMMAND ${clang} -c -x cl -emit-llvm
                           --target=spir64_fpga-unknown-unknown -cl-std=CL2.0
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                           -o ${obj_binary_dir}/assert_read_spir64_fpga.bc
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                   DEPENDS sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/assert_read_spir64.bc
                   COMMAND ${clang} -c -x cl -emit-llvm
                           --target=spir64-unknown-unknown -cl-std=CL2.0
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                           -o ${obj_binary_dir}/assert_read_spir64.bc
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cl
                   DEPENDS sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/assert_no_read_spir64.bc
                   COMMAND ${clang} -fsycl-device-only -emit-llvm
                           -fsycl-targets=spir64-unknown-unknown ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${obj_binary_dir}/assert_no_read_spir64.bc
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/assert_no_read_spir64_x86_64.bc
                   COMMAND ${clang} -fsycl-device-only -emit-llvm
                           -fsycl-targets=spir64_x86_64-unknown-unknown ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${obj_binary_dir}/assert_no_read_spir64_x86_64.bc
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/assert_no_read_spir64_gen.bc
                   COMMAND ${clang} -fsycl-device-only -emit-llvm
                           -fsycl-targets=spir64_gen-unknown-unknown ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${obj_binary_dir}/assert_no_read_spir64_gen.bc
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/assert_no_read_spir64_fpga.bc
                   COMMAND ${clang} -fsycl-device-only -emit-llvm
                           -fsycl-targets=spir64_fpga-unknown-unknown ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${obj_binary_dir}/assert_no_read_spir64_fpga.bc
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/assert_no_read_host.o
                   COMMAND ${clang} -c
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${obj_binary_dir}/assert_no_read_host.o
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/fallback-cassert_spir64.${lib-suffix}
                   COMMAND ${llvm-link} -o ${obj_binary_dir}/fallback-cassert_spir64.${lib-suffix}
                           ${obj_binary_dir}/assert_read_spir64.bc
                           ${obj_binary_dir}/assert_no_read_spir64.bc
                   DEPENDS ${obj_binary_dir}/assert_read_spir64.bc
                           ${obj_binary_dir}/assert_no_read_spir64.bc
                           llvm-link
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/fallback-cassert_spir64_x86_64.${lib-suffix}
                   COMMAND ${llvm-link} -o ${obj_binary_dir}/fallback-cassert_spir64_x86_64.${lib-suffix}
                           ${obj_binary_dir}/assert_read_spir64_x86_64.bc
                           ${obj_binary_dir}/assert_no_read_spir64_x86_64.bc
                   DEPENDS ${obj_binary_dir}/assert_read_spir64_x86_64.bc
                           ${obj_binary_dir}/assert_no_read_spir64_x86_64.bc
                           llvm-link
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/fallback-cassert_spir64_gen.${lib-suffix}
                   COMMAND ${llvm-link} -o ${obj_binary_dir}/fallback-cassert_spir64_gen.${lib-suffix}
                           ${obj_binary_dir}/assert_read_spir64_gen.bc
                           ${obj_binary_dir}/assert_no_read_spir64_gen.bc
                   DEPENDS ${obj_binary_dir}/assert_read_spir64_gen.bc
                           ${obj_binary_dir}/assert_no_read_spir64_gen.bc
                           llvm-link
                   VERBATIM)
add_custom_command(OUTPUT ${obj_binary_dir}/fallback-cassert_spir64_fpga.${lib-suffix}
                   COMMAND ${llvm-link} -o ${obj_binary_dir}/fallback-cassert_spir64_fpga.${lib-suffix}
                           ${obj_binary_dir}/assert_read_spir64_fpga.bc
                           ${obj_binary_dir}/assert_no_read_spir64_fpga.bc
                   DEPENDS ${obj_binary_dir}/assert_read_spir64_fpga.bc
                           ${obj_binary_dir}/assert_no_read_spir64_fpga.bc
                           llvm-link
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-cassert.${lib-suffix}
                   COMMAND ${bundler} -type=o ${bundler_targets_opt}
                           -outputs=${obj_binary_dir}/libsycl-fallback-cassert.${lib-suffix}
                           ${bundler_inputs_opt}
                   DEPENDS ${obj_binary_dir}/fallback-cassert_spir64_x86_64.${lib-suffix}
                           ${obj_binary_dir}/fallback-cassert_spir64_gen.${lib-suffix}
                           ${obj_binary_dir}/fallback-cassert_spir64_fpga.${lib-suffix}
                           ${obj_binary_dir}/fallback-cassert_spir64.${lib-suffix}
                           ${obj_binary_dir}/assert_no_read_host.${lib-suffix}
                   VERBATIM)
]]


add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-cassert.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-cassert.${lib-suffix}
                   MAIN_DEPENDENCY fallback-cassert.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)

################################################################################

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-cstring.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cstring.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-cstring.${lib-suffix}
                   MAIN_DEPENDENCY fallback-cstring.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-complex.spv
                   COMMAND ${clang} -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-complex.spv
                   MAIN_DEPENDENCY fallback-complex.cpp
                   DEPENDS device_math.h device_complex.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-complex.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-complex.${lib-suffix}
                   MAIN_DEPENDENCY fallback-complex.cpp
                   DEPENDS device_math.h device_complex.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
                   COMMAND ${clang} -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex-fp64.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
                   MAIN_DEPENDENCY fallback-complex-fp64.cpp
                   DEPENDS device_math.h device_complex.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-complex-fp64.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex-fp64.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-complex-fp64.${lib-suffix}
                   MAIN_DEPENDENCY fallback-complex-fp64.cpp
                   DEPENDS device_math.h device_complex.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cmath.spv
                   COMMAND ${clang} -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-cmath.spv
                   MAIN_DEPENDENCY fallback-cmath.cpp
                   DEPENDS device_math.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-cmath.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-cmath.${lib-suffix}
                   MAIN_DEPENDENCY fallback-cmath.cpp
                   DEPENDS device_math.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
                   COMMAND ${clang} -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath-fp64.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
                   MAIN_DEPENDENCY fallback-cmath-fp64.cpp
                   DEPENDS device_math.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-cmath-fp64.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath-fp64.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-cmath-fp64.${lib-suffix}
                   MAIN_DEPENDENCY fallback-cmath-fp64.cpp
                   DEPENDS device_math.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-itt-stubs.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/itt_stubs.cpp
                           -o ${obj_binary_dir}/libsycl-itt-stubs.${lib-suffix}
                   MAIN_DEPENDENCY itt_stubs.cpp
                   DEPENDS device_itt.h spirv_vars.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-itt-compiler-wrappers.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/itt_compiler_wrappers.cpp
                           -o ${obj_binary_dir}/libsycl-itt-compiler-wrappers.${lib-suffix}
                   MAIN_DEPENDENCY itt_compiler_wrappers.cpp
                   DEPENDS device_itt.h spirv_vars.h device.h sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-itt-user-wrappers.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/itt_user_wrappers.cpp
                           -o ${obj_binary_dir}/libsycl-itt-user-wrappers.${lib-suffix}
                   MAIN_DEPENDENCY itt_user_wrappers.cpp
                   DEPENDS device_itt.h spirv_vars.h device.h sycl-compiler
                   VERBATIM)

set(devicelib-obj-itt-files
  ${obj_binary_dir}/libsycl-itt-stubs.${lib-suffix}
  ${obj_binary_dir}/libsycl-itt-compiler-wrappers.${lib-suffix}
  ${obj_binary_dir}/libsycl-itt-user-wrappers.${lib-suffix}
  )

add_custom_target(libsycldevice-obj DEPENDS
  ${devicelib-obj-file}
  ${devicelib-obj-complex}
  ${devicelib-obj-complex-fp64}
  ${devicelib-obj-cmath}
  ${devicelib-obj-cmath-fp64}
  ${devicelib-obj-itt-files}
)
add_custom_target(libsycldevice-spv DEPENDS
  ${spv_binary_dir}/libsycl-fallback-cassert.spv
  ${spv_binary_dir}/libsycl-fallback-cstring.spv
  ${spv_binary_dir}/libsycl-fallback-complex.spv
  ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
  ${spv_binary_dir}/libsycl-fallback-cmath.spv
  ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
  )
add_custom_target(libsycldevice-fallback-obj DEPENDS
  ${obj_binary_dir}/libsycl-fallback-cassert.${lib-suffix}
  ${obj_binary_dir}/libsycl-fallback-cstring.${lib-suffix}
  ${obj_binary_dir}/libsycl-fallback-complex.${lib-suffix}
  ${obj_binary_dir}/libsycl-fallback-complex-fp64.${lib-suffix}
  ${obj_binary_dir}/libsycl-fallback-cmath.${lib-suffix}
  ${obj_binary_dir}/libsycl-fallback-cmath-fp64.${lib-suffix}
)
add_custom_target(libsycldevice DEPENDS
  libsycldevice-obj
  libsycldevice-fallback-obj
  libsycldevice-spv)

# Place device libraries near the libsycl.so library in an install
# directory as well
if (WIN32)
  set(install_dest_spv bin)
else()
  set(install_dest_spv lib${LLVM_LIBDIR_SUFFIX})
endif()

set(install_dest_lib lib${LLVM_LIBDIR_SUFFIX})

install(FILES ${devicelib-obj-file}
              ${obj_binary_dir}/libsycl-fallback-cassert.${lib-suffix}
              ${obj_binary_dir}/libsycl-fallback-cstring.${lib-suffix}
              ${devicelib-obj-complex}
              ${obj_binary_dir}/libsycl-fallback-complex.${lib-suffix}
              ${devicelib-obj-complex-fp64}
              ${obj_binary_dir}/libsycl-fallback-complex-fp64.${lib-suffix}
              ${devicelib-obj-cmath}
              ${obj_binary_dir}/libsycl-fallback-cmath.${lib-suffix}
              ${devicelib-obj-cmath-fp64}
              ${obj_binary_dir}/libsycl-fallback-cmath-fp64.${lib-suffix}
              ${devicelib-obj-itt-files}
        DESTINATION ${install_dest_lib}
        COMPONENT libsycldevice)

install(FILES ${spv_binary_dir}/libsycl-fallback-cassert.spv
              ${spv_binary_dir}/libsycl-fallback-cstring.spv
              ${spv_binary_dir}/libsycl-fallback-complex.spv
              ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
              ${spv_binary_dir}/libsycl-fallback-cmath.spv
              ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
        DESTINATION ${install_dest_spv}
        COMPONENT libsycldevice)
