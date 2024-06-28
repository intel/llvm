//.kernel _ZTS7imatrixIfLm16ELm16ELm8EE
//.platform DG2
//.thread_config numGRF=128, numAcc=4, numSWSB=16
//.options_string "-emitCrossThreadOffR0Reloc "
//.full_options "-emitLocation -wideMulMadOpsEn -enableCoalesceScalarMoves -enablePreemption -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -linker 63 -abortonspill -enableBundleCR 3 -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-ctrl 22 -presched-rp 100 -nodpsendreorder -SBIDDepLoc -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -enableHalfLSC -hasNoInt64Add -waLscUgmFence -LSCFenceWA "
//.instCount 23
//.RA type	LOCAL_ROUND_ROBIN_RA
//.git-hash 7b9e65874d020c7dfbade66b590437437e32379b

//.declare BuiltInR0 (0)  rf=r size=32 type=ud align=16 words (r0.0) IsBuiltin
//.declare R0_Copy0 (1)  rf=r size=32 type=ud align=16 words (r2.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r1.0)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r0.3)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r0.4)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r0.5)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %tsc (22)  rf=r size=20 type=ud align=2 words
//.declare %arg (23)  rf=r size=0 type=ud align=16 words (r26.0)
//.declare %retval (24)  rf=r size=0 type=ud align=16 words (r26.0) Output
//.declare %sp (25)  rf=r size=8 type=uq align=4 words (r127.3)
//.declare %fp (26)  rf=r size=8 type=uq align=4 words (r127.2)
//.declare %sr0 (27)  rf=r size=16 type=ud align=2 words
//.declare %cr0 (28)  rf=r size=12 type=ud align=2 words
//.declare %ce0 (29)  rf=r size=4 type=ud align=2 words
//.declare %dbg0 (30)  rf=r size=8 type=ud align=2 words
//.declare implBufPtr (32)  rf=r size=8 type=uq align=4 words (r126.0)
//.declare localIdBufPtr (33)  rf=r size=8 type=uq align=4 words (r126.3)
//.declare %msg0 (34)  rf=r size=12 type=ud align=2 words
//.declare R0_0 (42)  rf=r size=32 type=d alias=R0_Copy0+0 align=16 words (r2.0)
//.declare r0_0 (44)  rf=r size=32 type=d alias=R0_Copy0+0 align=16 words (r2.0)
//.declare payloadHeader (45)  rf=r size=32 type=d align=16 words (r7.0)
//.declare enqueuedLocalSize (46)  rf=r size=12 type=d align=2 words (r8.0)
//.declare localIdX (47)  rf=r size=32 type=w align=16 words (r1.0)
//.declare localIdX_0 (48)  rf=r size=32 type=w align=16 words (r2.0)
//.declare localIdY (49)  rf=r size=32 type=w align=16 words (r3.0)
//.declare localIdY_0 (50)  rf=r size=32 type=w align=16 words (r4.0)
//.declare localIdZ (51)  rf=r size=32 type=w align=16 words (r5.0)
//.declare localIdZ_0 (52)  rf=r size=32 type=w align=16 words (r6.0)
//.declare M2 (53)  rf=r size=32 type=ud align=16 words (r127.0)
//.declare r0 (54)  rf=r size=32 type=ud align=16 words (r0.0)
//.declare rtmp (55)  rf=r size=32 type=ud align=16 words (r127.0)
//.declare inlineRegFromTDL (56)  rf=r size=32 type=ud align=16 words (r1.0)
//.declare inlineRegExpectedLocation (57)  rf=r size=32 type=ud align=16 words (r7.0)
//.declare TV6 (58)  rf=r size=128 type=ud align=16 words (r1.0)
//.declare TV7 (59)  rf=r size=64 type=ud align=16 words (r5.0)
//.declare TV8 (60)  rf=r size=32 type=ud align=16 words (r8.0)

// .inputs
// +-------------------+----------+--------+----------+------------------+
// | id                | type     |  bytes | at       | from             |
// +-------------------+----------+--------+----------+------------------+
// | localIdX          | :w x 16  |   0x20 | r1       | pti[tid]+0x0     |
// | localIdX_0        | :w x 16  |   0x20 | r2       | pti[tid]+0x20    |
// | localIdY          | :w x 16  |   0x20 | r3       | pti[tid]+0x40    |
// | localIdY_0        | :w x 16  |   0x20 | r4       | pti[tid]+0x60    |
// | localIdZ          | :w x 16  |   0x20 | r5       | pti[tid]+0x80    |
// | localIdZ_0        | :w x 16  |   0x20 | r6       | pti[tid]+0xA0    |
// | payloadHeader     | :d x 8   |   0x20 | r7       | inline+0x0       |
// | enqueuedLocalSize | :d x 3   |    0xC | r8       | cti+0x0          |
// +-------------------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (8|M0)               r127.0<1>:ud  0x0:ud                                                //  ALU pipe: int; 
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x20:ud              {I@2}          //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     mad (1|M0)               r127.2<1>:ud  r127.2<0;0>:ud    r127.0<0;0>:uw    0xC0:uw              {I@1} //  ALU pipe: int; 
(W)     mov (8|M0)               r7.0<1>:ud    r1.0<1;1,0>:ud                                        //  ALU pipe: int; 
(W)     send.dc0 (8|M0)          r1       r127    null:0  0x0            0x024844FD           {A@1,$0} // wr:1h+0, rd:4; oword aligned block read x8 // 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x80:uw              {$0.src}       //  ALU pipe: int; 
(W)     send.dc0 (8|M0)          r5       r127    null:0  0x0            0x022843FD           {A@1,$1} // wr:1h+0, rd:2; oword aligned block read x4 // 
        nop                                                                                          // 
        nop                                                                                          // 
// B001: Preds:{B000},  Succs:{B002}
// cross_thread_prolog:
(W)     mov (8|M0)               r127.0<1>:ud  0x0:ud                              {$1.src}          //  ALU pipe: int; 
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     send.dc0 (8|M0)          r8       r127    null:0  0x0            0x021842FD           {A@1,$2} // wr:1h+0, rd:1; oword aligned block read x2 // 
// B002: Preds:{B001},  Succs:{}
// _main:
(W)     mov (8|M0)               r2.0<1>:ud    r0.0<1;1,0>:ud                   {$0.dst}             //  ALU pipe: int; $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {A@1}         // $1
        sync.nop                             null                             {Compacted,A@1}        // $2
        sync.nop                             null                             {Compacted,$2.src}     // $2
(W)     mov (8|M0)               r127.0<1>:f   r2.0<1;1,0>:f                    {Compacted,A@1}      //  ALU pipe: float; $2
(W)     send.gtwy (1|M0)         null     r127    null:0  0x0            0x02000010           {EOT,A@1} // wr:1+0, rd:0; end of thread // $2
L328:
        nop                                                                                          // $2


//.BankConflicts: 0
//.ByteRMWs: 0
//


//.numALUInst: 17
//.accSubDef: 0
//.accSubUse: 0
//.accSubCandidateDef: 0
//.accSubCandidateUse: 0
//
//
//.singlePipeAtOneDistNum: 5
//.allAtOneDistNum: 4
//.syncInstCount: 2
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 1
//.AfterReadTokenDepCount: 3
