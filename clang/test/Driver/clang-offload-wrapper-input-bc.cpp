// Generate fake AOCX files
// RUN: echo 'pseudo-aocx-0' > %t0.aocx
// RUN: echo 'pseudo-aocx-1' > %t1.aocx
// RUN: echo 'pseudo-aocx-2' > %t2.aocx

// Generate property files
// RUN: echo '[SYCL/devicelib req mask]'                                     >  %t0.prop
// RUN: echo 'DeviceLibReqMask=1|0'                                          >> %t0.prop
// RUN: echo '[SYCL/device requirements]'                                    >> %t0.prop
// RUN: echo 'aspects=2|AAAAAAAAAAA'                                         >> %t0.prop
// RUN: echo 'reqd_work_group_size=2|AAAAAAAAAAA'                            >> %t0.prop
// RUN: echo '[SYCL/kernel param opt]'                                       >> %t0.prop
// RUN: echo '_ZTSZ4add5N4sycl3_V15queueEPiiiE10add5_dummy=2|DAAAAAAAAAAA'   >> %t0.prop
// RUN: echo '_ZZ4add5N4sycl3_V15queueEPiiiENKUlvE_clEv=2|BAAAAAAAAAAA'      >> %t0.prop
// RUN: echo '[SYCL/misc properties]'                                        >> %t0.prop
// RUN: echo 'optLevel=1|2'                                                  >> %t0.prop

// RUN: echo '[SYCL/devicelib req mask]'                                     >  %t1.prop
// RUN: echo 'DeviceLibReqMask=1|0'                                          >> %t1.prop
// RUN: echo '[SYCL/device requirements]'                                    >> %t1.prop
// RUN: echo 'aspects=2|AAAAAAAAAAA'                                         >> %t1.prop
// RUN: echo 'reqd_work_group_size=2|AAAAAAAAAAA'                            >> %t1.prop
// RUN: echo '[SYCL/kernel param opt]'                                       >> %t1.prop
// RUN: echo '_ZZ4add4N4sycl3_V15queueEPiiiENKUlvE_clEv=2|BAAAAAAAAAAA'      >> %t1.prop
// RUN: echo '_ZTSZ4add4N4sycl3_V15queueEPiiiE10add4_dummy=2|DAAAAAAAAAAA'   >> %t1.prop
// RUN: echo '[SYCL/misc properties]'                                        >> %t1.prop
// RUN: echo 'optLevel=1|2'                                                  >> %t1.prop

// RUN: echo '[SYCL/devicelib req mask]'                                     >  %t2.prop
// RUN: echo 'DeviceLibReqMask=1|0'                                          >> %t2.prop
// RUN: echo '[SYCL/device requirements]'                                    >> %t2.prop
// RUN: echo 'aspects=2|AAAAAAAAAAA'                                         >> %t2.prop
// RUN: echo 'reqd_work_group_size=2|AAAAAAAAAAA'                            >> %t2.prop
// RUN: echo '[SYCL/kernel param opt]'                                       >> %t2.prop
// RUN: echo '_ZTSZ4add3N4sycl3_V15queueEPiiiE10add3_dummy=2|DAAAAAAAAAAA'   >> %t2.prop
// RUN: echo '_ZZ4add3N4sycl3_V15queueEPiiiENKUlvE_clEv=2|BAAAAAAAAAAA'      >> %t2.prop
// RUN: echo '[SYCL/misc properties]'                                        >> %t2.prop
// RUN: echo 'optLevel=1|2'                                                  >> %t2.prop

// Generate sym files
// RUN: echo '_ZTSZ4add5N4sycl3_V15queueEPiiiE10add5_dummy'                  >  %t0.sym
// RUN: echo '_ZTSZ4add4N4sycl3_V15queueEPiiiE10add4_dummy'                  >  %t1.sym
// RUN: echo '_ZTSZ4add3N4sycl3_V15queueEPiiiE10add3_dummy'                  >  %t2.sym

// Generate table file
// RUN: echo '[Code|Properties|Symbols]'                                     >  %t.table
// RUN: echo '%t0.aocx|%t0.prop|%t0.sym'                                     >> %t.table
// RUN: echo '%t1.aocx|%t1.prop|%t1.sym'                                     >> %t.table
// RUN: echo '%t2.aocx|%t2.prop|%t2.sym'                                     >> %t.table

// Generate BC file with Code, Properties and Symbols
// RUN: clang-offload-wrapper "-o=%t1.bc" "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx-intel-unknown" "-kind=sycl" "-batch"  %t.table

// Update fake AOCX files
// RUN: echo 'fake-aocx-0-updated'        > %t0.aocx
// RUN: echo 'fake-aocx-1-enhanced'       > %t1.aocx
// RUN: echo 'fake-aocx-2-different-form' > %t2.aocx

// Create BC file with only Code in table
// RUN: file-table-tform  --extract=Code  %t.table  -o %t_code_only.table

// Generate BC file with only Code coming from the table but everything else coming through --wrapped-bc-input
// Thus %t1.bc and %t2.bc should be the same except for their Code.
// RUN: clang-offload-wrapper "-o=%t2.bc" "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx-intel-unknown" "-kind=sycl" "-batch"  %t_code_only.table --wrapped-bc-input=%t1.bc

// RUN: llvm-dis %t1.bc
// RUN: llvm-dis %t2.bc

// Filter out expected differences
// ModuleID:                      filename is different
// sycl_offloading.?.data:        fake AOCX were updated
// sycl_offloading.device_images: pointer type to offloading data includes the length which is different
// RUN: egrep -v '(ModuleID|^@.sycl_offloading.[0-9].data|^@.sycl_offloading.device_images)' %t1.ll > %tfiltered1.txt
// RUN: egrep -v '(ModuleID|^@.sycl_offloading.[0-9].data|^@.sycl_offloading.device_images)' %t2.ll > %tfiltered2.txt

// RUN: cmp %tfiltered1.txt %tfiltered2.txt

// RUN: FileCheck --check-prefix=CHECK_LL1 < %t1.ll %s
// RUN: FileCheck --check-prefix=CHECK_LL2 < %t2.ll %s

// Check that expected code is found
// CHECK_LL1: pseudo-aocx-0
// CHECK_LL1: pseudo-aocx-1
// CHECK_LL1: pseudo-aocx-2

// CHECK_LL2: fake-aocx-0-updated
// CHECK_LL2: fake-aocx-1-enhanced
// CHECK_LL2: fake-aocx-2-different-form
