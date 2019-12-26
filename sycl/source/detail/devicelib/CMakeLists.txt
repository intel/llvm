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

add_custom_command(OUTPUT ${binary_dir}/libsycl-fallback-cassert.spv
                   COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                           ${compile_opts}
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o ${binary_dir}/libsycl-fallback-cassert.spv
                   MAIN_DEPENDENCY fallback-cassert.cpp
                   DEPENDS wrapper.h clang llvm-spirv
                   VERBATIM)

add_custom_target(devicelib-obj DEPENDS ${devicelib-obj-file})
add_custom_target(devicelib-spv DEPENDS ${binary_dir}/libsycl-fallback-cassert.spv)
add_dependencies(sycl devicelib-obj devicelib-spv)
if (MSVC)
    add_dependencies(sycld devicelib-obj devicelib-spv)
endif()

# Place device libraries near the libsycl.so library in an install
# directory as well
if (WIN32)
    set(install_dest bin)
else()
    set(install_dest lib)
endif()

install(FILES ${devicelib-obj-file}
              ${binary_dir}/libsycl-fallback-cassert.spv
        DESTINATION ${install_dest}
        COMPONENT sycl)
