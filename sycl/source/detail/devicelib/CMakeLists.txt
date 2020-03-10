# Place device libraries near the libsycl.so library in a build
# directory
if (WIN32)
    set(binary_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
else()
    set(binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

set(clang $<TARGET_FILE:clang>)

set(compile_opts
    # suppress an error about SYCL_EXTERNAL
    -Wno-sycl-strict
    # for CL/__spirv/spirv_vars.hpp
    -I${sycl_inc_dir})

if (WIN32)
    set(devicelib-obj-file ${binary_dir}/libsycl-msvc.o)
    add_custom_command(OUTPUT ${devicelib-obj-file}
                       COMMAND ${clang} -fsycl -c
                               ${compile_opts}
                               ${CMAKE_CURRENT_SOURCE_DIR}/msvc_wrapper.cpp
                               -o ${devicelib-obj-file}
                       MAIN_DEPENDENCY msvc_wrapper.cpp
                       DEPENDS wrapper.h clang
                       VERBATIM)
else()
    set(devicelib-obj-file ${binary_dir}/libsycl-glibc.o)
    add_custom_command(OUTPUT ${devicelib-obj-file}
                       COMMAND ${clang} -fsycl -c
                               ${compile_opts}
                               ${CMAKE_CURRENT_SOURCE_DIR}/glibc_wrapper.cpp
                               -o ${devicelib-obj-file}
                       MAIN_DEPENDENCY glibc_wrapper.cpp
                       DEPENDS wrapper.h clang
                       VERBATIM)
endif()


set(devicelib-obj-complex ${binary_dir}/libsycl-complex.o)
add_custom_command(OUTPUT ${devicelib-obj-complex}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/complex_wrapper.cpp
                           -o ${devicelib-obj-complex}
                   MAIN_DEPENDENCY complex_wrapper.cpp
                   DEPENDS device_complex.h clang
                   VERBATIM)

set(devicelib-obj-complex-fp64 ${binary_dir}/libsycl-complex-fp64.o)
add_custom_command(OUTPUT ${devicelib-obj-complex-fp64}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/complex_wrapper_fp64.cpp
                           -o ${devicelib-obj-complex-fp64}
                   MAIN_DEPENDENCY complex_wrapper_fp64.cpp
                   DEPENDS device_complex.h clang
                   VERBATIM)

set(devicelib-obj-cmath ${binary_dir}/libsycl-cmath.o)
add_custom_command(OUTPUT ${devicelib-obj-cmath}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/cmath_wrapper.cpp
                           -o ${devicelib-obj-cmath}
                   MAIN_DEPENDENCY cmath_wrapper.cpp
                   DEPENDS device_complex.h clang
                   VERBATIM)

set(devicelib-obj-cmath-fp64 ${binary_dir}/libsycl-cmath-fp64.o)
add_custom_command(OUTPUT ${devicelib-obj-cmath-fp64}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/cmath_wrapper_fp64.cpp
                           -o ${devicelib-obj-cmath-fp64}
                   MAIN_DEPENDENCY cmath_wrapper_fp64.cpp
                   DEPENDS device_math.h clang
                   VERBATIM)

add_custom_command(OUTPUT ${binary_dir}/libsycl-fallback-cassert.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${binary_dir}/libsycl-fallback-cassert.spv
                   MAIN_DEPENDENCY fallback-cassert.cpp
                   DEPENDS wrapper.h clang llvm-spirv
                   VERBATIM)

add_custom_command(OUTPUT ${binary_dir}/libsycl-fallback-complex.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex.cpp
                           -o ${binary_dir}/libsycl-fallback-complex.spv
                   MAIN_DEPENDENCY fallback-complex.cpp
                   DEPENDS device_math.h device_complex.h clang llvm-spirv
                   VERBATIM)

add_custom_command(OUTPUT ${binary_dir}/libsycl-fallback-complex-fp64.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-complex-fp64.cpp
                           -o ${binary_dir}/libsycl-fallback-complex-fp64.spv
                   MAIN_DEPENDENCY fallback-complex-fp64.cpp
                   DEPENDS device_math.h device_complex.h clang llvm-spirv
                   VERBATIM)

add_custom_command(OUTPUT ${binary_dir}/libsycl-fallback-cmath.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath.cpp
                           -o ${binary_dir}/libsycl-fallback-cmath.spv
                   MAIN_DEPENDENCY fallback-cmath.cpp
                   DEPENDS device_math.h clang llvm-spirv
                   VERBATIM)

add_custom_command(OUTPUT ${binary_dir}/libsycl-fallback-cmath-fp64.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cmath-fp64.cpp
                           -o ${binary_dir}/libsycl-fallback-cmath-fp64.spv
                   MAIN_DEPENDENCY fallback-cmath-fp64.cpp
                   DEPENDS device_math.h clang llvm-spirv
                   VERBATIM)

add_custom_target(devicelib-obj DEPENDS ${devicelib-obj-file}
                                        ${devicelib-obj-complex}
					${devicelib-obj-complex-fp64}
                                        ${devicelib-obj-cmath}
					${devicelib-obj-cmath-fp64})
add_custom_target(devicelib-spv DEPENDS ${binary_dir}/libsycl-fallback-cassert.spv
                                        ${binary_dir}/libsycl-fallback-complex.spv
					${binary_dir}/libsycl-fallback-complex-fp64.spv
                                        ${binary_dir}/libsycl-fallback-cmath.spv
					${binary_dir}/libsycl-fallback-cmath-fp64.spv)
add_dependencies(sycl devicelib-obj devicelib-spv)
if (MSVC)
    add_dependencies(sycld devicelib-obj devicelib-spv)
endif()

# Place device libraries near the libsycl.so library in an install
# directory as well
if (WIN32)
    set(install_dest bin)
else()
    set(install_dest lib${LLVM_LIBDIR_SUFFIX})
endif()

install(FILES ${devicelib-obj-file}
              ${binary_dir}/libsycl-fallback-cassert.spv
              ${devicelib-obj-complex}
              ${binary_dir}/libsycl-fallback-complex.spv
              ${devicelib-obj-complex-fp64}
              ${binary_dir}/libsycl-fallback-complex-fp64.spv
              ${devicelib-obj-cmath}
              ${binary_dir}/libsycl-fallback-cmath.spv
              ${devicelib-obj-cmath-fp64}
              ${binary_dir}/libsycl-fallback-cmath-fp64.spv
        DESTINATION ${install_dest}
        COMPONENT sycl)
