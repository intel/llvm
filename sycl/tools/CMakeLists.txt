function(link_llvm_libs target)
  foreach(LIB ${ARGN})
    target_include_directories(${target} SYSTEM PRIVATE ${LLVM_MAIN_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE ${LIB})
  endforeach()
endfunction()

add_subdirectory(sycl-ls)
add_subdirectory(syclbin-dump)

if (SYCL_ENABLE_XPTI_TRACING)
  if (UNIX)
    add_subdirectory(sycl-prof)
    add_subdirectory(sycl-trace)
    add_subdirectory(sycl-sanitize)
  endif()
endif()

if(WIN32)
  set(abi_symbols_dump "${PROJECT_SOURCE_DIR}/test/abi/sycl_symbols_windows.dump")
else()
  set(abi_symbols_dump "${PROJECT_SOURCE_DIR}/test/abi/sycl_symbols_linux.dump")
endif()
add_custom_command(
  OUTPUT "${abi_symbols_dump}"
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/abi_check.py"
          --mode dump_symbols
          --llvm-bin-path "$<TARGET_FILE_DIR:llvm-readobj>"
          --output "${abi_symbols_dump}"
          # Note that TARGET_LINKER_FILE is required to avoid the SONAME,
          # but on Windows it gives us the .lib file, so we use TARGET_FILE instead
          "$<IF:$<PLATFORM_ID:Windows>,$<TARGET_FILE:sycl>,$<TARGET_LINKER_FILE:sycl>>"
  DEPENDS sycl llvm-readobj
  COMMENT "Generating ABI symbols dump for SYCL"
)
add_custom_target(generate_sycl_abi_symbols DEPENDS "${abi_symbols_dump}")
