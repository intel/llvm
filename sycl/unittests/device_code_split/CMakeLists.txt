
function(add_device_code_split_unittest)
list(GET ARGV 0 currentTest)
list(GET ARGV 1 splitFlag)
list(GET ARGV 2 hostSources)

add_sycl_unittest(${currentTest} OBJECT
    ${hostSources}
)

set(CXX_COMPILER clang++)
if(MSVC)
    set(CXX_COMPILER clang-cl.exe)
endif()
set(DEVICE_COMPILER_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/${CXX_COMPILER})

list(APPEND DEVICE_COMPILER_FLAGS -fsycl -fsycl-device-code-split=${splitFlag} -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -D__SYCL_BUILD_SYCL_DLL)
list(APPEND inputFiles ${CMAKE_CURRENT_SOURCE_DIR}/src/device-code-first-file.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/device-code-second-file.cpp)

if(MSVC)
	if (CMAKE_BUILD_TYPE MATCHES "Debug")
		list(APPEND DEVICE_COMPILER_FLAGS /LDd)
		list(APPEND inputFiles ${LLVM_LIBRARY_OUTPUT_INTDIR}/sycld.lib)
	else()
		list(APPEND DEVICE_COMPILER_FLAGS /LD)
		list(APPEND inputFiles ${LLVM_LIBRARY_OUTPUT_INTDIR}/sycl.lib)
	endif()
    set(outputCommand ${CMAKE_CURRENT_BINARY_DIR}/DeviceCodeLib${splitFlag})
	set(LibName ${outputCommand}.lib)
else()
	list(APPEND DEVICE_COMPILER_FLAGS -shared -fpic)
	if (CMAKE_BUILD_TYPE MATCHES "Debug")
		list(APPEND DEVICE_COMPILER_FLAGS -g)
	endif()
    set_target_properties(${currentTest} PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_CURRENT_BINARY_DIR} -L${CMAKE_CURRENT_BINARY_DIR}")
    set(LibName libDeviceCodeLib${splitFlag}.so)
    set(outputCommand ${CMAKE_CURRENT_BINARY_DIR}/${LibName})
endif()

add_custom_command(
    OUTPUT ${LibName}
    COMMAND ${DEVICE_COMPILER_EXECUTABLE}
            ${DEVICE_COMPILER_FLAGS}
           -I${SYCL_SOURCE_DIR}/unittests/
           -I${SYCL_SOURCE_DIR}/source/
           -o ${outputCommand}
            ${inputFiles}
    COMMENT "Building a dynamic library: ${outputCommand}"
)
add_custom_target(${currentTest}_lib DEPENDS
    ${LibName}
)

add_dependencies(${currentTest} sycl clang ${currentTest}_lib)
target_link_libraries(${currentTest} PRIVATE ${LibName})
endfunction()

add_device_code_split_unittest( DeviceCodeSplitPerSource
    per_source
    src/split-per-source.cpp
)

add_device_code_split_unittest( DeviceCodeSplitPerKernel
    per_kernel
    src/split-per-kernel.cpp
)
