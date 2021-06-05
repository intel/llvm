set(obj_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if (WIN32)
  set(lib-suffix obj)
  set(spv_binary_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
else()
  set(lib-suffix o)
  set(spv_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()
set(clang $<TARGET_FILE:clang>)

string(CONCAT sycl_targets_opt
  "-fsycl-targets="
  "spir64_x86_64-unknown-unknown-sycldevice,"
  "spir64_gen-unknown-unknown-sycldevice,"
  "spir64_fpga-unknown-unknown-sycldevice,"
  "spir64-unknown-unknown-sycldevice")

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

set(devicelib-obj-file ${obj_binary_dir}/libsycl-crt.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-file}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/crt_wrapper.cpp
                           -o ${devicelib-obj-file}
                   MAIN_DEPENDENCY crt_wrapper.cpp
                   DEPENDS wrapper.h device.h spirv_vars.h clang clang-offload-bundler
                   VERBATIM)

set(devicelib-obj-complex ${obj_binary_dir}/libsycl-complex.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-complex}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/complex_wrapper.cpp
                           -o ${devicelib-obj-complex}
                   MAIN_DEPENDENCY complex_wrapper.cpp
                   DEPENDS device_complex.h device.h clang clang-offload-bundler
                   VERBATIM)

set(devicelib-obj-complex-fp64 ${obj_binary_dir}/libsycl-complex-fp64.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-complex-fp64}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/complex_wrapper_fp64.cpp
                           -o ${devicelib-obj-complex-fp64}
                   MAIN_DEPENDENCY complex_wrapper_fp64.cpp
                   DEPENDS device_complex.h device.h clang clang-offload-bundler
                   VERBATIM)

set(devicelib-obj-cmath ${obj_binary_dir}/libsycl-cmath.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-cmath}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/cmath_wrapper.cpp
                           -o ${devicelib-obj-cmath}
                   MAIN_DEPENDENCY cmath_wrapper.cpp
                   DEPENDS device_math.h device.h clang clang-offload-bundler
                   VERBATIM)

set(devicelib-obj-cmath-fp64 ${obj_binary_dir}/libsycl-cmath-fp64.${lib-suffix})
add_custom_command(OUTPUT ${devicelib-obj-cmath-fp64}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/cmath_wrapper_fp64.cpp
                           -o ${devicelib-obj-cmath-fp64}
                   MAIN_DEPENDENCY cmath_wrapper_fp64.cpp
                   DEPENDS device_math.h device.h clang clang-offload-bundler
                   VERBATIM)

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cassert.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-cassert.spv
                   MAIN_DEPENDENCY fallback-cassert.cpp
                   DEPENDS wrapper.h device.h clang spirv_vars.h llvm-spirv
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-cassert.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-cassert.${lib-suffix}
                   MAIN_DEPENDENCY fallback-cassert.cpp
                   DEPENDS wrapper.h device.h clang spirv_vars.h clang-offload-bundler
                   VERBATIM)

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-complex.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-complex.spv
                   MAIN_DEPENDENCY fallback-complex.cpp
                   DEPENDS device_math.h device_complex.h device.h clang llvm-spirv
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-complex.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-complex.${lib-suffix}
                   MAIN_DEPENDENCY fallback-complex.cpp
                   DEPENDS device_math.h device_complex.h device.h clang clang-offload-bundler
                   VERBATIM)

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex-fp64.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
                   MAIN_DEPENDENCY fallback-complex-fp64.cpp
                   DEPENDS device_math.h device_complex.h device.h clang llvm-spirv
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-complex-fp64.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex-fp64.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-complex-fp64.${lib-suffix}
                   MAIN_DEPENDENCY fallback-complex-fp64.cpp
                   DEPENDS device_math.h device_complex.h device.h clang clang-offload-bundler
                   VERBATIM)

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cmath.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-cmath.spv
                   MAIN_DEPENDENCY fallback-cmath.cpp
                   DEPENDS device_math.h device.h clang llvm-spirv
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-cmath.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-cmath.${lib-suffix}
                   MAIN_DEPENDENCY fallback-cmath.cpp
                   DEPENDS device_math.h device.h clang clang-offload-bundler
                   VERBATIM)

add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath-fp64.cpp
                           -o ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
                   MAIN_DEPENDENCY fallback-cmath-fp64.cpp
                   DEPENDS device_math.h device.h clang llvm-spirv
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-cmath-fp64.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath-fp64.cpp
                           -o ${obj_binary_dir}/libsycl-fallback-cmath-fp64.${lib-suffix}
                   MAIN_DEPENDENCY fallback-cmath-fp64.cpp
                   DEPENDS device_math.h device.h clang clang-offload-bundler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-itt-stubs.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/itt_stubs.cpp
                           -o ${obj_binary_dir}/libsycl-itt-stubs.${lib-suffix}
                   MAIN_DEPENDENCY itt_stubs.cpp
                   DEPENDS device_itt.h spirv_vars.h device.h clang clang-offload-bundler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-itt-compiler-wrappers.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/itt_compiler_wrappers.cpp
                           -o ${obj_binary_dir}/libsycl-itt-compiler-wrappers.${lib-suffix}
                   MAIN_DEPENDENCY itt_compiler_wrappers.cpp
                   DEPENDS device_itt.h spirv_vars.h device.h clang clang-offload-bundler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-itt-user-wrappers.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/itt_user_wrappers.cpp
                           -o ${obj_binary_dir}/libsycl-itt-user-wrappers.${lib-suffix}
                   MAIN_DEPENDENCY itt_user_wrappers.cpp
                   DEPENDS device_itt.h spirv_vars.h device.h clang clang-offload-bundler
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
  ${spv_binary_dir}/libsycl-fallback-complex.spv
  ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
  ${spv_binary_dir}/libsycl-fallback-cmath.spv
  ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
  )
add_custom_target(libsycldevice-fallback-obj DEPENDS
  ${obj_binary_dir}/libsycl-fallback-cassert.${lib-suffix}
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
              ${spv_binary_dir}/libsycl-fallback-complex.spv
              ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
              ${spv_binary_dir}/libsycl-fallback-cmath.spv
              ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
        DESTINATION ${install_dest_spv}
        COMPONENT libsycldevice)
