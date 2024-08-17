include(GNUInstallDirs)

macro(add_syclcompress_lib target_name)
  add_library(${target_name} STATIC ${ARGN})
  target_include_directories(${target_name} PRIVATE ${SYCL_COMPRESS_DIR}/include ${ZSTD_INCLUDE_DIR})
  target_link_libraries(${target_name} PRIVATE ${ZSTD_LIBRARY})

  # if (MSVC)
  #   target_compile_options(${target_name} PRIVATE /EHsc)
  # endif()

  # Set the location of the library installation
  install(TARGETS ${target_name}
    RUNTIME DESTINATION bin COMPONENT sycl-compress
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT sycl-compress
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT sycl-compress
  )
endmacro()

set(SOURCES
  sycl-compress.cpp
)

add_syclcompress_lib(sycl-compress ${SOURCES})