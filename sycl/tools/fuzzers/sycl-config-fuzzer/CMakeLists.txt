add_executable(sycl-config-fuzzer
  EXCLUDE_FROM_ALL
  main.cpp
  # ${SYCL_SOURCE_DIR}/source/detail/config.cpp
  # ${SYCL_SOURCE_DIR}/source/exception.cpp
  # ${SYCL_SOURCE_DIR}/source/detail/os_util.cpp
)

if (TARGET sycl-config-fuzzer)
  target_compile_options(
    sycl-config-fuzzer
    PRIVATE
      -fsanitize=fuzzer
      #-fsanitize=address
      -fprofile-instr-generate -fcoverage-mapping
      -g
  )

  target_link_libraries(sycl-config-fuzzer
    PRIVATE
    $<TARGET_OBJECTS:sycl_object>
    $<GENEX_EVAL:$<TARGET_PROPERTY:sycl,LINK_LIBRARIES>>
  )

  target_link_options(
    sycl-config-fuzzer
    PRIVATE
      -fsanitize=fuzzer
      -fprofile-instr-generate -fcoverage-mapping
      #-fsanitize=address
  )

  set_target_properties(sycl-config-fuzzer PROPERTIES FOLDER "SYCL/Fuzzers")
  target_include_directories(sycl-config-fuzzer
    PRIVATE SYSTEM
      ${SYCL_SOURCE_DIR}/source/
  )

  set(CORPUS_BINARY_DIR ${CMAKE_BINARY_DIR}/fuzzer-artifacts/sycl-config-corpus)

  add_custom_command(TARGET sycl-config-fuzzer PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/fuzzer-artifacts/sycl-config-artifacts
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/corpus ${CORPUS_BINARY_DIR}
  )

  add_custom_target(fuzz-sycl-config
    COMMENT "Running the SYCL config fuzzer for a minute..."
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/fuzzer-artifacts/sycl-config-artifacts
    COMMAND $<TARGET_FILE:sycl-config-fuzzer> ${CORPUS_BINARY_DIR} -artifact_prefix=config- -max_total_time=600
    USES_TERMINAL
  )
  set_target_properties(fuzz-sycl-config PROPERTIES FOLDER "SYCL/Fuzzers")
endif()
