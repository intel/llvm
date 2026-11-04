# MemorySanitizer for the level_zero plugin: loadZeShim() dlopens
# liboffload_msan_ze.so, which the plugin's dlopen("libze_loader.so") then
# binds to, and forwards the hooked calls to msan_ze_intercept.cpp.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../sanitizers/msan
                 ${CMAKE_CURRENT_BINARY_DIR}/sanitizers_msan)

add_library(offload_msan_level_zero OBJECT
  msan_ze.cpp
  msan_ze_intercept.cpp
)
llvm_update_compile_flags(offload_msan_level_zero)
target_compile_options(offload_msan_level_zero PRIVATE ${offload_compile_flags})
target_compile_definitions(offload_msan_level_zero PRIVATE
  TARGET_NAME=MSAN
  DEBUG_PREFIX="MSAN"
)
# Full L0 headers: the bundled dynamic_l0 ones lack the virtual memory API.
target_include_directories(offload_msan_level_zero PRIVATE
  ${LIBOMPTARGET_DEP_LEVEL_ZERO_INCLUDE_DIR})
target_link_libraries(offload_msan_level_zero PUBLIC offload_sanitizer_msan)
set_target_properties(offload_msan_level_zero PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)
# Both listed: OBJECT libraries' objects do not propagate transitively.
target_link_libraries(omptarget.rtl.level_zero PRIVATE
                      offload_msan_level_zero offload_sanitizer_msan)
# L0DynWrapper.cpp calls loadZeShim() under this define.
target_compile_definitions(omptarget.rtl.level_zero PRIVATE OFFLOAD_MSAN_ZE=1)

# Link-time stand-in giving the shim DT_NEEDED libze_loader.so.1 whether or
# not a real loader is installed at build time.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ze_loader_needed.cpp "")
add_library(offload_msan_ze_needed SHARED
            ${CMAKE_CURRENT_BINARY_DIR}/ze_loader_needed.cpp)
set_target_properties(offload_msan_ze_needed PROPERTIES
                      NO_SONAME ON
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_link_options(offload_msan_ze_needed PRIVATE
                    "-Wl,-soname,libze_loader.so.1")

add_library(offload_msan_ze SHARED ze_shim/msan_ze_shim.cpp)
llvm_update_compile_flags(offload_msan_ze)
target_compile_options(offload_msan_ze PRIVATE ${offload_compile_flags})
target_include_directories(offload_msan_ze PRIVATE
                           ${LIBOMPTARGET_DEP_LEVEL_ZERO_INCLUDE_DIR})
target_link_libraries(offload_msan_ze PRIVATE ${CMAKE_DL_LIBS}
  -Wl,--no-as-needed offload_msan_ze_needed -Wl,--as-needed)
set_target_properties(offload_msan_ze PROPERTIES NO_SONAME ON)
target_link_options(offload_msan_ze PRIVATE "-Wl,-soname,libze_loader.so")
add_dependencies(omptarget.rtl.level_zero offload_msan_ze)
install(TARGETS offload_msan_ze
  COMPONENT offload
  LIBRARY DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
