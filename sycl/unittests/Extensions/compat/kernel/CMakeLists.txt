add_sycl_unittest_with_extra_flags(Extensionscompatkerneltests SHARED
  KernelFunctionLin.cpp
  KernelModuleLin.cpp
)

add_library(Extensionscompat_kernel_function_test SHARED KernelModuleLin.cpp)
set_property(TARGET Extensionscompat_kernel_function_test
             PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET Extensionscompat_kernel_function_test
             PROPERTY LIBRARY_OUTPUT_DIRECTORY ${CURRENT_DIR})
target_compile_options(Extensionscompat_kernel_function_test
                       PRIVATE ${SYCL_COMPAT_TEST_FLAGS})
target_link_options(Extensionscompat_kernel_function_test 
                    PRIVATE ${SYCL_COMPAT_TEST_FLAGS})

add_dependencies(Extensionscompatkerneltests
                 Extensionscompat_kernel_function_test)
target_link_libraries(Extensionscompatkerneltests 
                      PRIVATE dl)
