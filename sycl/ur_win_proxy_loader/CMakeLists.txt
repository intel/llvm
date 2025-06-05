project(ur_win_proxy_loader)
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/versioninfo.rc
               PROPERTY COMPILE_DEFINITIONS
               "RC_VERSION_FIELD_1=${SYCL_MAJOR_VERSION}"
               "RC_VERSION_FIELD_2=${SYCL_MAJOR_VERSION}"
               "RC_VERSION_FIELD_3=${SYCL_PATCH_VERSION}"
               "RC_VERSION_FIELD_4=${SYCL_DEV_ABI_VERSION}"
               "RC_FILE_VERSION=\"${SYCL_VERSION_STRING}\""
               "RC_INTERNAL_NAME=\"ur_win_proxy_loader\""
               "RC_PRODUCT_NAME=\"ur_win_proxy_loader\""
               "RC_PRODUCT_VERSION=\"${SYCL_VERSION_STRING}\""
               "RC_COPYRIGHT=\"Copyright (C) 2023 Intel Corporation\"")
configure_file(../../llvm/resources/windows_version_resource.rc ${CMAKE_CURRENT_BINARY_DIR}/versioninfo.rc @ONLY)
add_library(ur_win_proxy_loader SHARED  ur_win_proxy_loader.cpp  ${CMAKE_CURRENT_BINARY_DIR}/versioninfo.rc)
if (CMAKE_SYSTEM_NAME STREQUAL Windows AND
    (CMAKE_BUILD_TYPE STREQUAL Debug OR CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebugDLL))
    set_target_properties(ur_win_proxy_loader PROPERTIES OUTPUT_NAME "ur_win_proxy_loaderd")
    target_compile_definitions(ur_win_proxy_loader PRIVATE UR_WIN_PROXY_LOADER_DEBUG_POSTFIX)
endif()
add_dependencies(ur_win_proxy_loader UnifiedRuntimeLoader)
install(TARGETS ur_win_proxy_loader
  RUNTIME DESTINATION "bin" COMPONENT ur_win_proxy_loader
)

if (MSVC)
  target_link_libraries(ur_win_proxy_loader PRIVATE shlwapi)
  # 0x2000: LOAD_LIBRARY_SAFE_CURRENT_DIRS flag. Using this flag means that loading dependency DLLs
  # from the current directory is only allowed if it is under a directory in the Safe load list.
  target_link_options(ur_win_proxy_loader PRIVATE LINKER:/DEPENDENTLOADFLAG:0x2000)
endif()
