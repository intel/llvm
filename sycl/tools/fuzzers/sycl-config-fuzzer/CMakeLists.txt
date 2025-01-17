#set(LLVM_LINK_COMPONENTS
  #  Support # For dealing with temporary files
  #)

message(WARNING ${LLVM_LIB_FUZZING_ENGINE})

add_executable(sycl-config-fuzzer
  EXCLUDE_FROM_ALL
  main.cpp
)

if (TARGET sycl-config-fuzzer)
  target_link_libraries(sycl-config-fuzzer
    PRIVATE
    ${LLVM_LIB_FUZZING_ENGINE}
    # We link the whole SYCL RT at object level
    $<TARGET_OBJECTS:sycl_object>
    $<TARGET_PROPERTY:sycl,LINK_LIBRARIES>
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
    COMMAND $<TARGET_FILE:sycl-config-fuzzer> ${CORPUS_BINARY_DIR} -artifact_prefix=config- -max_total_time=60
    USES_TERMINAL
  )
  set_target_properties(fuzz-sycl-config PROPERTIES FOLDER "SYCL/Fuzzers")
endif()
