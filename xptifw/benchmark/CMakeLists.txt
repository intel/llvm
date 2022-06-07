add_executable(XPTIFWBenchmark
  object_table.cpp
  string_table.cpp
  main.cpp
)

target_include_directories(XPTIFWBenchmark PRIVATE
    $<BUILD_INTERFACE:${XPTIFW_DIR}/include>
    $<BUILD_INTERFACE:${XPTI_DIR}/include>
)

target_link_libraries(XPTIFWBenchmark PRIVATE benchmark)

if (XPTI_ENABLE_STATISTICS)
  target_compile_definitions(XPTIFWBenchmark PRIVATE XPTI_STATISTICS)
endif()
