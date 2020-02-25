set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_executable(test_pi_plugins EXCLUDE_FROM_ALL
  "test.cpp"
)
target_link_libraries(test_pi_plugins PRIVATE
  piapi
)
add_test(NAME pi_plugins COMMAND test_pi_plugins)

# The test assumes pi_opencl is always available
if(WIN32)
  set_tests_properties(pi_plugins PROPERTIES
    ENVIRONMENT "PATH=$<TARGET_FILE_DIR:pi_opencl>;$ENV{PATH}"
  )
else()
  set_target_properties(test_pi_plugins PROPERTIES
    INSTALL_RPATH "$<TARGET_FILE_DIR:pi_opencl>"
    BUILD_WITH_INSTALL_RPATH ON
  )
endif()

add_subdirectory(opencl)
add_dependencies(test_pi_plugins pi_opencl)

if(PI_BUILD_CUDA)
  add_subdirectory(cuda)
  add_dependencies(test_pi_plugins pi_cuda)
endif()

if(PI_BUILD_LEVEL_ZERO)
  add_subdirectory(level_zero)
  add_dependencies(test_pi_plugins pi_level_zero)
endif ()
