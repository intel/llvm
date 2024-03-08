file(GLOB SOURCES *.cpp *.hpp)
include_directories(${XPTIFW_DIR}/include)
include_directories(${XPTI_DIR}/include)
include_directories(${SAMPLES_DIR}/include)

add_executable(XPTIFWBasicTests ${SOURCES})
target_link_libraries(XPTIFWBasicTests PRIVATE xptifw ${CMAKE_DL_LIBS})

if (UNIX)
  add_compile_options(-gline-tables-only -O3)
endif()

# Override default LLVM bin location
set_target_properties(XPTIFWBasicTests
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(check-xptifw-basictest COMMAND
  $<TARGET_PROPERTY:XPTIFWBasicTests,RUNTIME_OUTPUT_DIRECTORY>/XPTIFWBasicTests)
add_dependencies(check-xptifw-basictest XPTIFWBasicTests)
add_dependencies(check-xptifw check-xptifw-basictest)
