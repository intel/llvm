# Place device libraries near the libsycl.so library in a build
# directory
if (MSVC)
    set(binary_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
else()
    set(binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

set(clang $<TARGET_FILE:clang>)

if (MSVC)
    set(devicelib-obj-file ${binary_dir}/libsycl-msvc.o)
    add_custom_command(OUTPUT ${devicelib-obj-file}
                       COMMAND ${clang} -fsycl -c
                               # suppress an error about SYCL_EXTERNAL
                               -Wno-error=sycl-strict -Wno-sycl-strict
                               ${CMAKE_CURRENT_SOURCE_DIR}/msvc_wrapper.cpp
                               -o ${devicelib-obj-file}
                       MAIN_DEPENDENCY msvc_wrapper.cpp
                       DEPENDS wrapper.h clang
                       VERBATIM)
else()
    set(devicelib-obj-file ${binary_dir}/libsycl-glibc.o)
    add_custom_command(OUTPUT ${devicelib-obj-file}
                       COMMAND ${clang} -fsycl -c
                               # suppress an error about SYCL_EXTERNAL
                               -Wno-error=sycl-strict -Wno-sycl-strict
                               ${CMAKE_CURRENT_SOURCE_DIR}/glibc_wrapper.cpp
                               -o ${devicelib-obj-file}
                       MAIN_DEPENDENCY glibc_wrapper.cpp
                       DEPENDS wrapper.h clang
                       VERBATIM)
endif()

add_custom_command(OUTPUT ${binary_dir}/libsycl-fallback-cassert.spv
                   COMMAND ${clang} -fsycl-device-only -S -Xclang -emit-llvm-bc
                            # suppress an error about SYCL_EXTERNAL
                            -Wno-error=sycl-strict -Wno-sycl-strict
                           ${CMAKE_CURRENT_SOURCE_DIR}/fallback-cassert.cpp
                           -o libsycl-fallback-cassert.bc
                   COMMAND llvm-spirv libsycl-fallback-cassert.bc
                           -o ${binary_dir}/libsycl-fallback-cassert.spv
                   MAIN_DEPENDENCY fallback-cassert.cpp
                   DEPENDS wrapper.h clang
                   VERBATIM)

add_custom_target(devicelib-obj DEPENDS ${devicelib-obj-file})
add_custom_target(devicelib-spv DEPENDS ${binary_dir}/libsycl-fallback-cassert.spv)
add_dependencies(sycl devicelib-obj devicelib-spv)
if (MSVC)
    add_dependencies(sycld devicelib-obj devicelib-spv)
endif()

# Place device libraries near the libsycl.so library in an install
# directory as well
if (MSVC)
    set(install_dest bin)
else()
    set(install_dest lib)
endif()

install(FILES ${devicelib-obj-file}
              ${binary_dir}/libsycl-fallback-cassert.spv
        DESTINATION ${install_dest}
        COMPONENT sycl)
