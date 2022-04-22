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
  list(APPEND compile_opts -D_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
endif()

function(addSYCLLibDeviceObj input output)
set(extra_deps ${ARGN})
add_custom_command(OUTPUT ${obj_binary_dir}/${output}.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${CMAKE_CURRENT_SOURCE_DIR}/${input}.cpp
                           -o ${obj_binary_dir}/${output}.${lib-suffix}
                   MAIN_DEPENDENCY ${input}.cpp
                   DEPENDS device.h sycl-compiler ${extra_deps}
                   VERBATIM)
endfunction(addSYCLLibDeviceObj)

function(addSYCLLibDeviceSPV input output)
set(extra_deps ${ARGN})
add_custom_command(OUTPUT ${spv_binary_dir}/${output}.spv
                   COMMAND ${clang} -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/${input}.cpp
                           -o ${spv_binary_dir}/${output}.spv
                   MAIN_DEPENDENCY ${input}.cpp
                   DEPENDS device.h sycl-compiler ${extra_deps}
                   VERBATIM)
endfunction(addSYCLLibDeviceSPV input output)

function(addSYCLLibDeviceObjAndSPV input output)
  addSYCLLibDeviceObj(${input} ${output} ${ARGN})
  addSYCLLibDeviceSPV(${input} ${output} ${ARGN})
endfunction(addSYCLLibDeviceObjAndSPV input output)

addSYCLLibDeviceObj(crt_wrapper           libsycl-crt wrapper.h         spirv_vars.h)
addSYCLLibDeviceObj(complex_wrapper       libsycl-complex               device_complex.h)
addSYCLLibDeviceObj(complex_wrapper_fp64  libsycl-complex-fp64          device_complex.h)
addSYCLLibDeviceObj(cmath_wrapper         libsycl-cmath                 device_math.h)
addSYCLLibDeviceObj(cmath_wrapper_fp64    libsycl-cmath-fp64            device_math.h)

addSYCLLibDeviceObj(itt_stubs             libsycl-itt-stubs             device_itt.h spirv_vars.h)
addSYCLLibDeviceObj(itt_compiler_wrappers libsycl-itt-compiler-wrappers device_itt.h spirv_vars.h)
addSYCLLibDeviceObj(itt_user_wrappers     libsycl-itt-user-wrappers     device_itt.h spirv_vars.h)

addSYCLLibDeviceObjAndSPV(fallback-cassert      libsycl-fallback-cassert      wrapper.h spirv_vars.h)
addSYCLLibDeviceObjAndSPV(fallback-cstring      libsycl-fallback-cstring      wrapper.h spirv_vars.h)
addSYCLLibDeviceObjAndSPV(fallback-complex      libsycl-fallback-complex      device_math.h device_complex.h)
addSYCLLibDeviceObjAndSPV(fallback-complex-fp64 libsycl-fallback-complex-fp64 device_math.h device_complex.h)
addSYCLLibDeviceObjAndSPV(fallback-cmath        libsycl-fallback-cmath        device_math.h)
addSYCLLibDeviceObjAndSPV(fallback-cmath-fp64   libsycl-fallback-cmath-fp64   device_math.h)

set(devicelib-obj-itt-files
  ${obj_binary_dir}/libsycl-itt-stubs.${lib-suffix}
  ${obj_binary_dir}/libsycl-itt-compiler-wrappers.${lib-suffix}
  ${obj_binary_dir}/libsycl-itt-user-wrappers.${lib-suffix}
)

add_custom_target(libsycldevice-obj DEPENDS
  ${obj_binary_dir}/libsycl-crt.${lib-suffix}
  ${obj_binary_dir}/libsycl-complex.${lib-suffix}
  ${obj_binary_dir}/libsycl-complex-fp64.${lib-suffix}
  ${obj_binary_dir}/libsycl-cmath.${lib-suffix}
  ${obj_binary_dir}/libsycl-cmath-fp64.${lib-suffix}
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
  libsycldevice-spv
)

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
