diff --git a/libclc/CMakeLists.txt b/libclc/CMakeLists.txt
index c3dd52c4ad6b..02fd6197df82 100644
--- a/libclc/CMakeLists.txt
+++ b/libclc/CMakeLists.txt
@@ -22,18 +22,8 @@ option(
   LIBCLC_USE_SPIRV_BACKEND "Build SPIR-V targets with the SPIR-V backend." OFF
 )
 
-# List of all supported targets.
-set( LIBCLC_TARGETS_ALL
-  amdgcn-amd-amdhsa-llvm
-  clspv--
-  clspv64--
-  native_cpu
-  nvptx64--
-  nvptx64--nvidiacl
-  nvptx64-nvidia-cuda
-  spirv-mesa3d-
-  spirv64-mesa3d-
-)
+# List of all supported architectures.
+set( LIBCLC_ARCHS_ALL amdgpu amdgcn clspv clspv64 nvptx64 spirv spirv64 )
 
 # Handle both LLVM_RUNTIMES_TARGET (per-target builds) and LIBCLC_TARGETS_TO_BUILD (multi-target builds)
 if(LLVM_RUNTIMES_TARGET)
@@ -52,9 +42,13 @@ endif()
 if(NOT LIBCLC_TARGET)
   message(FATAL_ERROR "libclc target is empty\n")
 endif()
-if(NOT "${LIBCLC_TARGET}" IN_LIST LIBCLC_TARGETS_ALL)
-  message(FATAL_ERROR "Unknown libclc target: ${LIBCLC_TARGET}\n"
-    "Valid targets are: ${LIBCLC_TARGETS_ALL}\n")
+
+string( REPLACE "-" ";" _target_components ${LIBCLC_TARGET} )
+list(GET _target_components 0 _target_arch)
+if(NOT "${_target_arch}" IN_LIST LIBCLC_ARCHS_ALL)
+  message(FATAL_ERROR "Unknown libclc target architecture: ${_target_arch}\n"
+    "Target was: ${LIBCLC_TARGET}\n"
+    "Valid architectures are: ${LIBCLC_ARCHS_ALL}\n")
 endif()
 
 if( LIBCLC_STANDALONE_BUILD OR CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
@@ -82,6 +76,11 @@ else()
     message(FATAL_ERROR "Clang is not enabled, but is required to build libclc in-tree")
   endif()
 
+  # The package version is required to find the resource directory.
+  if( LLVM_PACKAGE_VERSION AND NOT PACKAGE_VERSION )
+    set( PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}" )
+  endif()
+
   get_host_tool_path( llvm-link LLVM_LINK llvm-link_exe llvm-link_target )
   get_host_tool_path( opt OPT opt_exe opt_target )
 
@@ -119,35 +118,12 @@ else()
   list(GET TRIPLE 2 OS)
 endif()
 
-<<<<<<< HEAD
-if( LIBCLC_TARGETS_TO_BUILD STREQUAL "all" )
-  set( LIBCLC_TARGETS_TO_BUILD ${LIBCLC_TARGETS_ALL} )
-else()
-  foreach(TARGET_TO_BUILD ${LIBCLC_TARGETS_TO_BUILD})
-    if (NOT ${TARGET_TO_BUILD} IN_LIST LIBCLC_TARGETS_ALL)
-      message( FATAL_ERROR
-        "Unknown target in LIBCLC_TARGETS_TO_BUILD: \"${TARGET_TO_BUILD}\"\n"
-        "Valid targets are: ${LIBCLC_TARGETS_ALL}\n")
-    endif()
-  endforeach()
-endif()
-
-option( LIBCLC_NATIVECPU_HOST_TARGET "Build libclc for Native CPU." Off)
-
-if( LIBCLC_NATIVECPU_HOST_TARGET )
-  list(APPEND LIBCLC_TARGETS_TO_BUILD native_cpu)
-endif()
-
-list( SORT LIBCLC_TARGETS_TO_BUILD )
-
-=======
 if(ARCH STREQUAL spirv OR ARCH STREQUAL spirv64)
   if(NOT LIBCLC_USE_SPIRV_BACKEND AND NOT llvm-spirv_exe)
     message(FATAL_ERROR "SPIR-V backend or llvm-spirv is required for libclc ${LIBCLC_TARGET}")
   endif()
 endif()
 
->>>>>>> 4fa7f992e5330303545c7de42211bcca7881ad4c
 foreach( tool IN ITEMS opt llvm-link )
   if( NOT EXISTS "${${tool}_exe}" AND "${${tool}_target}" STREQUAL "" )
     message( FATAL_ERROR "libclc toolchain incomplete - missing tool ${tool}!" )
@@ -155,14 +131,6 @@ foreach( tool IN ITEMS opt llvm-link )
 endforeach()
 
 add_subdirectory(clc/lib/generic)
-<<<<<<< HEAD
-add_subdirectory(clc/lib/amdgpu)
-add_subdirectory(clc/lib/ptx-nvidiacl)
-add_subdirectory(clc/lib/spirv)
-add_subdirectory(clc/lib/clspv)
-
-=======
->>>>>>> 4fa7f992e5330303545c7de42211bcca7881ad4c
 add_subdirectory(opencl/lib/generic)
 
 if(ARCH STREQUAL amdgcn)
@@ -386,54 +354,6 @@ if(BUILD_LIBSPIRV)
     PARENT_TARGET libspirv-builtins
   )
 
-<<<<<<< HEAD
-  if( BUILD_LIBSPIRV_${t} )
-    # Build base libspirv library (without remangling)
-    add_libclc_library(libspirv-${t}
-      ARCH ${ARCH}
-      TRIPLE ${clang_triple}
-      TARGET_TRIPLE ${t}
-      SOURCES ${libspirv_sources}
-      COMPILE_OPTIONS ${compile_flags} "SHELL:-Xclang -fdeclare-spirv-builtins"
-      INCLUDE_DIRS
-        ${CMAKE_CURRENT_SOURCE_DIR}/clc/include
-        ${CMAKE_CURRENT_SOURCE_DIR}/libspirv/include
-      COMPILE_DEFINITIONS ${_common_defs}
-      INTERNALIZE_LIBRARIES ${clc_lib}
-      OPT_FLAGS ${opt_flags}
-      OUTPUT_FILENAME libspirv
-      PARENT_TARGET libspirv-builtins
-    )
-
-    # Build remangled libspirv variants for different long widths and char signedness
-    foreach(long_width 32 64)
-      foreach(char_signedness signed unsigned)
-        add_libclc_library(libspirv-${t}-l${long_width}-${char_signedness}
-          ARCH ${ARCH}
-          TRIPLE ${clang_triple}
-          TARGET_TRIPLE ${t}
-          SOURCES ${libspirv_sources}
-          COMPILE_OPTIONS ${compile_flags}
-            "SHELL:-Xclang -fdeclare-spirv-builtins"
-            "SHELL:-Xclang -fsycl-remangle-libspirv"
-            "SHELL:-mllvm -remangle-long-width=${long_width}"
-            "SHELL:-mllvm -remangle-char-signedness=${char_signedness}"
-          INCLUDE_DIRS
-            ${CMAKE_CURRENT_SOURCE_DIR}/clc/include
-            ${CMAKE_CURRENT_SOURCE_DIR}/libspirv/include
-          COMPILE_DEFINITIONS ${_common_defs}
-          INTERNALIZE_LIBRARIES ${clc_lib}
-          OPT_FLAGS ${opt_flags}
-          OUTPUT_FILENAME libspirv.l${long_width}.${char_signedness}_char
-          PARENT_TARGET libspirv-builtins
-        )
-      endforeach()
-    endforeach()
-  endif()
-endforeach()
-
-add_subdirectory(test)
-=======
   # Build remangled variants for SYCL with different long widths and char signedness
   foreach(long_width 32 64)
     foreach(char_signedness signed unsigned)
@@ -460,7 +380,8 @@ add_subdirectory(test)
   endforeach()
 endif()
 
+set(LIBCLC_UNRESOLVED_SYMBOL_TEST_TARGETS libclc-${LIBCLC_TARGET})
+
 if(LLVM_INCLUDE_TESTS)
   add_subdirectory(test)
 endif()
->>>>>>> 4fa7f992e5330303545c7de42211bcca7881ad4c
diff --git a/libclc/cmake/modules/AddLibclc.cmake b/libclc/cmake/modules/AddLibclc.cmake
index 628eb2e650fe..cdf99f735a4e 100644
--- a/libclc/cmake/modules/AddLibclc.cmake
+++ b/libclc/cmake/modules/AddLibclc.cmake
@@ -193,7 +193,6 @@ function(add_libclc_library target_name)
     DESTINATION ${LIBCLC_INSTALL_DIR}/${ARG_TRIPLE}
     COMPONENT ${ARG_PARENT_TARGET}
   )
-<<<<<<< HEAD
 
   # SPIR-V targets can exit early here
   if( ARG_ARCH STREQUAL spirv OR ARG_ARCH STREQUAL spirv64 )
@@ -211,6 +210,4 @@ function(add_libclc_library target_name)
       WORKING_DIRECTORY ${LIBCLC_SOURCE_DIR} )
   endif()
 
-=======
->>>>>>> 4fa7f992e5330303545c7de42211bcca7881ad4c
 endfunction()
diff --git a/libclc/cmake/modules/CMakeDetermineCLCCompiler.cmake b/libclc/cmake/modules/CMakeDetermineCLCCompiler.cmake
index 2138ad85d005..60ea64d2ce26 100644
--- a/libclc/cmake/modules/CMakeDetermineCLCCompiler.cmake
+++ b/libclc/cmake/modules/CMakeDetermineCLCCompiler.cmake
@@ -1,9 +1,4 @@
 if(NOT CMAKE_CLC_COMPILER)
-  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
-    message(FATAL_ERROR
-      "The CLC language requires the C compiler (CMAKE_C_COMPILER) to be "
-      "Clang, but CMAKE_C_COMPILER_ID is '${CMAKE_C_COMPILER_ID}'.")
-  endif()
   set(CMAKE_CLC_COMPILER "${CMAKE_C_COMPILER}" CACHE FILEPATH "CLC compiler")
 endif()
 
diff --git a/libclc/opencl/lib/generic/atomic/atomic_fetch_add.cl b/libclc/opencl/lib/generic/atomic/atomic_fetch_add.cl
index 12b6f37b3fdf..7a475a2ebb83 100644
--- a/libclc/opencl/lib/generic/atomic/atomic_fetch_add.cl
+++ b/libclc/opencl/lib/generic/atomic/atomic_fetch_add.cl
@@ -52,22 +52,21 @@ _CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_add(volatile atomic_uintptr_t *p,
 
 #ifdef __opencl_c_atomic_scope_device
 
-_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_add(
+_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_add_explicit(
     volatile __local atomic_uintptr_t *p, ptrdiff_t v, memory_order order) {
   return __scoped_atomic_fetch_add((volatile __local uintptr_t *)p, v, order,
                                    __MEMORY_SCOPE_DEVICE);
 }
 
-_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_add(
+_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_add_explicit(
     volatile __global atomic_uintptr_t *p, ptrdiff_t v, memory_order order) {
   return __scoped_atomic_fetch_add((volatile __global uintptr_t *)p, v, order,
                                    __MEMORY_SCOPE_DEVICE);
 }
 
 #if _CLC_GENERIC_AS_SUPPORTED
-_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_add(volatile atomic_uintptr_t *p,
-                                                  ptrdiff_t v,
-                                                  memory_order order) {
+_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_add_explicit(
+    volatile atomic_uintptr_t *p, ptrdiff_t v, memory_order order) {
   return __scoped_atomic_fetch_add((volatile uintptr_t *)p, v, order,
                                    __MEMORY_SCOPE_DEVICE);
 }
@@ -75,25 +74,24 @@ _CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_add(volatile atomic_uintptr_t *p,
 #endif // __opencl_c_atomic_scope_device
 
 _CLC_DEF _CLC_OVERLOAD uintptr_t
-atomic_fetch_add(volatile __local atomic_uintptr_t *p, ptrdiff_t v,
-                 memory_order order, memory_scope scope) {
+atomic_fetch_add_explicit(volatile __local atomic_uintptr_t *p, ptrdiff_t v,
+                          memory_order order, memory_scope scope) {
   return __scoped_atomic_fetch_add((volatile __local uintptr_t *)p, v, order,
                                    __opencl_get_clang_memory_scope(scope));
 }
 
 _CLC_DEF _CLC_OVERLOAD uintptr_t
-atomic_fetch_add(volatile __global atomic_uintptr_t *p, ptrdiff_t v,
-                 memory_order order, memory_scope scope) {
+atomic_fetch_add_explicit(volatile __global atomic_uintptr_t *p, ptrdiff_t v,
+                          memory_order order, memory_scope scope) {
   return __scoped_atomic_fetch_add((volatile __global uintptr_t *)p, v, order,
                                    __opencl_get_clang_memory_scope(scope));
 }
 
 #if _CLC_GENERIC_AS_SUPPORTED
 
-_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_add(volatile atomic_uintptr_t *p,
-                                                  ptrdiff_t v,
-                                                  memory_order order,
-                                                  memory_scope scope) {
+_CLC_DEF _CLC_OVERLOAD uintptr_t
+atomic_fetch_add_explicit(volatile atomic_uintptr_t *p, ptrdiff_t v,
+                          memory_order order, memory_scope scope) {
   return __scoped_atomic_fetch_add((volatile uintptr_t *)p, v, order,
                                    __opencl_get_clang_memory_scope(scope));
 }
diff --git a/libclc/opencl/lib/generic/atomic/atomic_fetch_sub.cl b/libclc/opencl/lib/generic/atomic/atomic_fetch_sub.cl
index 6dfcdde207ef..8ef4ae2dd3d7 100644
--- a/libclc/opencl/lib/generic/atomic/atomic_fetch_sub.cl
+++ b/libclc/opencl/lib/generic/atomic/atomic_fetch_sub.cl
@@ -52,22 +52,21 @@ _CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub(volatile atomic_uintptr_t *p,
 
 #ifdef __opencl_c_atomic_scope_device
 
-_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub(
+_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub_explicit(
     volatile __local atomic_uintptr_t *p, ptrdiff_t v, memory_order order) {
   return __scoped_atomic_fetch_sub((volatile __local uintptr_t *)p, v, order,
                                    __MEMORY_SCOPE_DEVICE);
 }
 
-_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub(
+_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub_explicit(
     volatile __global atomic_uintptr_t *p, ptrdiff_t v, memory_order order) {
   return __scoped_atomic_fetch_sub((volatile __global uintptr_t *)p, v, order,
                                    __MEMORY_SCOPE_DEVICE);
 }
 
 #if _CLC_GENERIC_AS_SUPPORTED
-_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub(volatile atomic_uintptr_t *p,
-                                                  ptrdiff_t v,
-                                                  memory_order order) {
+_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub_explicit(
+    volatile atomic_uintptr_t *p, ptrdiff_t v, memory_order order) {
   return __scoped_atomic_fetch_sub((volatile uintptr_t *)p, v, order,
                                    __MEMORY_SCOPE_DEVICE);
 }
@@ -75,25 +74,24 @@ _CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub(volatile atomic_uintptr_t *p,
 #endif // __opencl_c_atomic_scope_device
 
 _CLC_DEF _CLC_OVERLOAD uintptr_t
-atomic_fetch_sub(volatile __local atomic_uintptr_t *p, ptrdiff_t v,
-                 memory_order order, memory_scope scope) {
+atomic_fetch_sub_explicit(volatile __local atomic_uintptr_t *p, ptrdiff_t v,
+                          memory_order order, memory_scope scope) {
   return __scoped_atomic_fetch_sub((volatile __local uintptr_t *)p, v, order,
                                    __opencl_get_clang_memory_scope(scope));
 }
 
 _CLC_DEF _CLC_OVERLOAD uintptr_t
-atomic_fetch_sub(volatile __global atomic_uintptr_t *p, ptrdiff_t v,
-                 memory_order order, memory_scope scope) {
+atomic_fetch_sub_explicit(volatile __global atomic_uintptr_t *p, ptrdiff_t v,
+                          memory_order order, memory_scope scope) {
   return __scoped_atomic_fetch_sub((volatile __global uintptr_t *)p, v, order,
                                    __opencl_get_clang_memory_scope(scope));
 }
 
 #if _CLC_GENERIC_AS_SUPPORTED
 
-_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub(volatile atomic_uintptr_t *p,
-                                                  ptrdiff_t v,
-                                                  memory_order order,
-                                                  memory_scope scope) {
+_CLC_DEF _CLC_OVERLOAD uintptr_t
+atomic_fetch_sub_explicit(volatile atomic_uintptr_t *p, ptrdiff_t v,
+                          memory_order order, memory_scope scope) {
   return __scoped_atomic_fetch_sub((volatile uintptr_t *)p, v, order,
                                    __opencl_get_clang_memory_scope(scope));
 }
diff --git a/libclc/test/CMakeLists.txt b/libclc/test/CMakeLists.txt
index 9054fd347473..a162150209d2 100644
--- a/libclc/test/CMakeLists.txt
+++ b/libclc/test/CMakeLists.txt
@@ -52,33 +52,31 @@ foreach( t ${LIBCLC_TARGETS_TO_BUILD} )
       )
 endforeach( t )
 
-<<<<<<< HEAD
-=======
 # Testing unresolved symbols.
 # Skip nvptx, clspv, spirv targets
 if(ARCH MATCHES amdgcn)
-  # Get the output file from the target property
-  set(target_file "$<TARGET_PROPERTY:libclc-${LIBCLC_TARGET},TARGET_FILE>")
+  foreach(tgt IN LISTS LIBCLC_UNRESOLVED_SYMBOL_TEST_TARGETS)
+    set(target_file "$<TARGET_PROPERTY:${tgt},TARGET_FILE>")
 
-  set(LIBCLC_TARGET_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/${LIBCLC_TARGET})
-  file(MAKE_DIRECTORY ${LIBCLC_TARGET_TEST_DIR})
-  file(GENERATE OUTPUT ${LIBCLC_TARGET_TEST_DIR}/check-external-funcs.test
-    CONTENT "; RUN: llvm-nm -u \"${target_file}\" | FileCheck %s --allow-empty\n\n; CHECK-NOT: {{.+}}\n"
-  )
+    set(LIBCLC_TARGET_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/${tgt})
+    file(MAKE_DIRECTORY ${LIBCLC_TARGET_TEST_DIR})
+    file(GENERATE OUTPUT ${LIBCLC_TARGET_TEST_DIR}/check-external-funcs.test
+      CONTENT "; RUN: llvm-nm -u \"${target_file}\" | FileCheck %s --allow-empty\n\n; CHECK-NOT: {{.+}}\n"
+    )
 
-  configure_lit_site_cfg(
-      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
-      ${LIBCLC_TARGET_TEST_DIR}/lit.site.cfg.py
-      MAIN_CONFIG
-      ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
-  )
+    configure_lit_site_cfg(
+        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
+        ${LIBCLC_TARGET_TEST_DIR}/lit.site.cfg.py
+        MAIN_CONFIG
+        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
+    )
 
-  add_lit_testsuite(check-libclc-external-funcs-${LIBCLC_TARGET} "Running ${LIBCLC_TARGET} tests"
-    ${LIBCLC_TARGET_TEST_DIR}
-    DEPENDS ${LIBCLC_TEST_DEPS}
-  )
-  set_target_properties(check-libclc-external-funcs-${LIBCLC_TARGET} PROPERTIES FOLDER "libclc tests")
+    add_lit_testsuite(check-libclc-external-funcs-${tgt} "Running ${tgt} unresolved symbols tests"
+      ${LIBCLC_TARGET_TEST_DIR}
+      DEPENDS ${LIBCLC_TEST_DEPS}
+    )
+    set_target_properties(check-libclc-external-funcs-${tgt} PROPERTIES FOLDER "libclc tests")
+  endforeach()
 endif()
 
->>>>>>> 4fa7f992e5330303545c7de42211bcca7881ad4c
 umbrella_lit_testsuite_end(check-libclc)
