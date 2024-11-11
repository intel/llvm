function(link_llvm_libs target)
  foreach(LIB ${ARGN})
    target_include_directories(${target} SYSTEM PRIVATE ${LLVM_MAIN_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE ${LIB})
  endforeach()
endfunction()

add_subdirectory(sycl-ls)
