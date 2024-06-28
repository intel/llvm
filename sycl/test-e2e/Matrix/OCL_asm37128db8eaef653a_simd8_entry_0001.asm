//.kernel _ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm8ELm8ELm16EE
//.platform DG2
//.thread_config numGRF=128, numAcc=4, numSWSB=16
//.options_string "-emitCrossThreadOffR0Reloc "
//.full_options "-emitLocation -enableCoalesceScalarMoves -enablePreemption -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -linker 63 -abortOnSpill 4 -enableBundleCR 3 -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-rp 100 -nodpsendreorder -SBIDDepLoc -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -hasNoInt64Add -waLscUgmFence -LSCFenceWA "
//.instCount 494
//.RA type	LOCAL_ROUND_ROBIN_RA
//.git-hash 0fb8c2e00e8a30145f7000997a3221be79640aa3

//.declare BuiltInR0 (0)  rf=r size=32 type=ud align=16 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=32 type=ud align=16 words (r105.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare %null (9)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (12)  rf=r size=4 type=ud align=2 words (r2.7)
//.declare %local_id_y (13)  rf=r size=4 type=ud align=2 words (r3.0)
//.declare %local_size_x (14)  rf=r size=4 type=ud align=2 words (r2.3)
//.declare %local_size_y (15)  rf=r size=4 type=ud align=2 words (r2.4)
//.declare %group_id_x (16)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (17)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (18)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (19)  rf=r size=4 type=ud align=2 words (r2.5)
//.declare %group_count_y (20)  rf=r size=4 type=ud align=2 words (r2.6)
//.declare %tsc (21)  rf=r size=20 type=ud align=2 words
//.declare %arg (22)  rf=r size=0 type=ud align=16 words (r26.0)
//.declare %retval (23)  rf=r size=0 type=ud align=16 words (r26.0) Output
//.declare %sp (24)  rf=r size=8 type=uq align=4 words (r127.3)
//.declare %fp (25)  rf=r size=8 type=uq align=4 words (r127.2)
//.declare %sr0 (26)  rf=r size=16 type=ud align=2 words
//.declare %cr0 (27)  rf=r size=12 type=ud align=2 words
//.declare %ce0 (28)  rf=r size=4 type=ud align=2 words
//.declare %dbg0 (29)  rf=r size=8 type=ud align=2 words
//.declare implBufPtr (31)  rf=r size=8 type=uq align=4 words (r126.0)
//.declare localIdBufPtr (32)  rf=r size=8 type=uq align=4 words (r126.3)
//.declare %msg0 (33)  rf=r size=12 type=ud align=2 words
//.declare V0033 (41)  rf=r size=32 type=d alias=+0 align=16 words (r105.0)
//.declare V0034 (42)  rf=r size=8 type=q align=4 words (r5.3)
//.declare V0036 (44)  rf=r size=32 type=d alias=+0 align=16 words (r105.0)
//.declare V0037 (45)  rf=r size=32 type=d align=16 words (r4.0)
//.declare V0038 (46)  rf=r size=12 type=d align=2 words (r9.5)
//.declare V0039 (47)  rf=r size=12 type=d align=2 words (r10.0)
//.declare V0040 (48)  rf=r size=32 type=w align=16 words (r1.0)
//.declare V0041 (49)  rf=r size=32 type=w align=16 words (r2.0)
//.declare V0042 (50)  rf=r size=32 type=w align=16 words (r3.0)
//.declare V0043 (51)  rf=r size=8 type=uq align=4 words (r9.0)
//.declare V0044 (52)  rf=r size=8 type=q align=4 words (r6.0)
//.declare V0045 (53)  rf=r size=8 type=q align=4 words (r6.1)
//.declare V0046 (54)  rf=r size=8 type=q align=4 words (r6.2)
//.declare V0047 (55)  rf=r size=8 type=q align=4 words (r6.3)
//.declare V0048 (56)  rf=r size=8 type=q align=4 words (r7.0)
//.declare V0049 (57)  rf=r size=8 type=q align=4 words (r7.1)
//.declare V0050 (58)  rf=r size=8 type=q align=4 words (r7.2)
//.declare V0051 (59)  rf=r size=8 type=q align=4 words (r7.3)
//.declare V0052 (60)  rf=r size=8 type=q align=4 words (r8.0)
//.declare V0053 (61)  rf=r size=8 type=q align=4 words (r8.1)
//.declare V0054 (62)  rf=r size=8 type=q align=4 words (r8.2)
//.declare V0055 (63)  rf=r size=8 type=q align=4 words (r8.3)
//.declare V0056 (64)  rf=r size=4 type=d align=2 words (r9.2)
//.declare V0057 (65)  rf=r size=4 type=d align=2 words (r9.3)
//.declare V0058 (66)  rf=r size=4 type=d align=2 words (r9.4)
//.declare V0059 (67)  rf=r size=256 type=d align=16 words (r14.0)
//.declare V0060 (68)  rf=r size=256 type=d align=16 words (r38.0)
//.declare V0061 (69)  rf=r size=256 type=d align=16 words (r73.0)
//.declare V0062 (70)  rf=r size=256 type=d align=16 words (r9.0)
//.declare V0063 (71)  rf=r size=256 type=d align=16 words (r37.0)
//.declare V0064 (72)  rf=r size=256 type=d align=16 words (r53.0)
//.declare V0065 (73)  rf=r size=8 type=d align=2 words (r106.0)
//.declare V0066 (74)  rf=r size=8 type=d alias=V0034+0 align=4 words (r5.6)
//.declare V0067 (75)  rf=r size=8 type=d align=2 words (r3.0)
//.declare V0068 (76)  rf=r size=8 type=d alias=V0045+0 align=4 words (r6.2)
//.declare V0069 (77)  rf=r size=8 type=d align=2 words (r5.0)
//.declare V0070 (78)  rf=r size=8 type=d alias=V0046+0 align=4 words (r6.4)
//.declare V0071 (79)  rf=r size=8 type=d align=2 words (r6.0)
//.declare V0072 (80)  rf=r size=8 type=d alias=V0047+0 align=4 words (r6.6)
//.declare V0073 (81)  rf=r size=8 type=d align=2 words (r7.0)
//.declare V0074 (82)  rf=r size=8 type=d alias=V0049+0 align=4 words (r7.2)
//.declare V0075 (83)  rf=r size=8 type=d align=2 words (r8.0)
//.declare V0076 (84)  rf=r size=8 type=d alias=V0050+0 align=4 words (r7.4)
//.declare V0077 (85)  rf=r size=8 type=d align=2 words (r106.2)
//.declare V0078 (86)  rf=r size=8 type=d alias=V0051+0 align=4 words (r7.6)
//.declare V0079 (87)  rf=r size=8 type=d align=2 words (r9.0)
//.declare V0080 (88)  rf=r size=8 type=d alias=V0053+0 align=4 words (r8.2)
//.declare V0081 (89)  rf=r size=8 type=d align=2 words (r10.3)
//.declare V0082 (90)  rf=r size=8 type=d alias=V0054+0 align=4 words (r8.4)
//.declare V0083 (91)  rf=r size=8 type=d align=2 words (r106.4)
//.declare V0084 (92)  rf=r size=8 type=d alias=V0055+0 align=4 words (r8.6)
//.declare V0086 (94)  rf=r size=4 type=d align=2 words (r15.0)
//.declare V0088 (96)  rf=r size=8 type=ud alias=V0069+0 align=2 words (r5.0)
//.declare V0089 (97)  rf=r size=8 type=ud alias=V0067+0 align=2 words (r3.0)
//.declare V0090 (98)  rf=r size=4 type=d align=2 words (r14.0)
//.declare V0092 (100)  rf=r size=64 type=ud align=16 words (r11.0)
//.declare V0093 (101)  rf=r size=4 type=ud alias=V0090+0 align=2 words (r14.0)
//.declare V0094 (102)  rf=r size=4 type=d align=16 words (r13.0)
//.declare V0095 (103)  rf=r size=4 type=d align=2 words (r16.0)
//.declare V0096 (104)  rf=r size=4 type=d align=2 words (r17.0)
//.declare V0097 (105)  rf=r size=4 type=d align=2 words (r18.0)
//.declare V0098 (106)  rf=r size=4 type=d align=2 words (r106.6)
//.declare V0099 (107)  rf=r size=4 type=d align=2 words (r106.7)
//.declare V0100 (108)  rf=r size=4 type=d align=2 words (r107.0)
//.declare V0101 (109)  rf=r size=4 type=ud alias=V0099+0 align=2 words (r106.7)
//.declare V0102 (110)  rf=r size=8 type=ud alias=V0075+0 align=2 words (r8.0)
//.declare V0103 (111)  rf=r size=8 type=ud alias=V0073+0 align=2 words (r7.0)
//.declare V0104 (112)  rf=r size=4 type=d align=2 words (r22.0)
//.declare V0106 (114)  rf=r size=64 type=ud align=16 words (r19.0)
//.declare V0107 (115)  rf=r size=4 type=ud alias=V0104+0 align=2 words (r22.0)
//.declare V0108 (116)  rf=r size=4 type=d align=16 words (r21.0)
//.declare V0109 (117)  rf=r size=4 type=d align=2 words (r23.0)
//.declare V0110 (118)  rf=r size=4 type=d align=2 words (r107.1)
//.declare V0111 (119)  rf=r size=4 type=d align=2 words (r107.2)
//.declare V0112 (120)  rf=r size=4 type=d align=2 words (r107.3)
//.declare V0113 (121)  rf=r size=4 type=d align=2 words (r107.4)
//.declare V0114 (122)  rf=r size=4 type=d align=2 words (r107.5)
//.declare V0115 (123)  rf=r size=4 type=ud alias=V0113+0 align=2 words (r107.4)
//.declare V0116 (124)  rf=r size=8 type=ud alias=V0081+0 align=2 words (r10.3)
//.declare V0117 (125)  rf=r size=8 type=ud alias=V0079+0 align=2 words (r9.0)
//.declare V0118 (126)  rf=r size=4 type=d align=2 words (r27.0)
//.declare V0120 (128)  rf=r size=64 type=ud align=16 words (r24.0)
//.declare V0121 (129)  rf=r size=4 type=ud alias=V0118+0 align=2 words (r27.0)
//.declare V0122 (130)  rf=r size=4 type=d align=16 words (r26.0)
//.declare V0123 (131)  rf=r size=4 type=d align=2 words (r28.0)
//.declare V0124 (132)  rf=r size=4 type=d align=2 words (r107.6)
//.declare V0125 (133)  rf=r size=4 type=d align=2 words (r107.7)
//.declare V0126 (134)  rf=r size=4 type=d align=2 words (r108.0)
//.declare V0127 (135)  rf=r size=4 type=d align=16 words (r29.0)
//.declare V0129 (137)  rf=r size=32 type=uw alias=V0041+0 align=16 words (r2.0)
//.declare V0130 (138)  rf=r size=32 type=d align=16 words (r30.0)
//.declare V0131 (139)  rf=r size=4 type=d align=16 words (r31.0)
//.declare V0133 (141)  rf=r size=32 type=uw alias=V0040+0 align=16 words (r1.0)
//.declare V0134 (142)  rf=r size=32 type=d align=16 words (r32.0)
//.declare V0137 (145)  rf=r size=64 type=q align=16 words (r33.0)
//.declare V0138 (146)  rf=r size=64 type=d alias=V0137+0 align=16 words (r33.0)
//.declare V0139 (147)  rf=r size=8 type=q align=4 words (r35.0)
//.declare V0140 (148)  rf=r size=8 type=ud alias=V0139+0 align=4 words (r35.0)
//.declare V0141 (149)  rf=r size=64 type=ud alias=V0137+0 align=16 words (r33.0)
//.declare V0142 (150)  rf=r size=8 type=d align=2 words (r36.0)
//.declare V0143 (151)  rf=r size=8 type=d alias=V0139+0 align=4 words (r35.0)
//.declare V0144 (152)  rf=r size=4 type=d align=16 words (r109.0)
//.declare V0145 (153)  rf=r size=4 type=d align=2 words (r108.1)
//.declare V0146 (154)  rf=r size=4 type=ud alias=V0144+0 align=2 words (r109.0)
//.declare V0147 (155)  rf=r size=4 type=ud alias=V0145+0 align=2 words (r108.1)
//.declare V0148 (156)  rf=r size=32 type=ud alias=V0130+0 align=16 words (r30.0)
//.declare V0149 (157)  rf=r size=8 type=ud alias=V0142+0 align=2 words (r36.0)
//.declare V0150 (158)  rf=r size=32 type=ud align=16 words (r37.0)
//.declare V0151 (159)  rf=r size=32 type=d align=16 words (r39.0)
//.declare V0152 (160)  rf=r size=32 type=d align=16 words (r110.0)
//.declare V0153 (161)  rf=r size=32 type=d align=16 words (r111.0)
//.declare V0154 (162)  rf=r size=32 type=ud alias=V0152+0 align=16 words (r110.0)
//.declare V0155 (163)  rf=r size=32 type=ud alias=V0153+0 align=16 words (r111.0)
//.declare V0156 (164)  rf=r size=32 type=ud alias=V0134+0 align=16 words (r32.0)
//.declare V0157 (165)  rf=r size=32 type=ud alias=V0151+0 align=16 words (r39.0)
//.declare V0159 (167)  rf=r size=4 type=d align=16 words (r40.0)
//.declare V0160 (168)  rf=r size=4 type=d align=2 words (r42.0)
//.declare V0161 (169)  rf=r size=4 type=ud alias=V0159+0 align=2 words (r40.0)
//.declare V0162 (170)  rf=r size=4 type=ud alias=V0160+0 align=2 words (r42.0)
//.declare V0163 (171)  rf=r size=4 type=ud alias=V0086+0 align=2 words (r15.0)
//.declare V0164 (172)  rf=r size=8 type=ud alias=V0071+0 align=2 words (r6.0)
//.declare V0165 (173)  rf=r size=4 type=ud align=16 words (r41.0)
//.declare V0167 (175)  rf=r size=4 type=d align=2 words (r108.2)
//.declare V0168 (176)  rf=r size=4 type=d align=2 words (r108.3)
//.declare V0169 (177)  rf=r size=32 type=d align=16 words (r43.0)
//.declare P01 (178)  rf=f8  size=2 type=uw align=1 words (f0.1)
//.declare V0170 (179)  rf=r size=32 type=ud alias=V0169+0 align=16 words (r43.0)
//.declare V0171 (180)  rf=r size=4 type=d align=2 words (r2.0)
//.declare V0172 (181)  rf=r size=8 type=ud alias=V0065+0 align=2 words (r106.0)
//.declare V0173 (182)  rf=r size=4 type=d align=2 words (r3.0)
//.declare V0175 (184)  rf=r size=4 type=d align=2 words (r4.0)
//.declare V0176 (185)  rf=r size=4 type=d align=2 words (r5.0)
//.declare V0177 (186)  rf=r size=4 type=f align=2 words (r6.0)
//.declare V0178 (187)  rf=r size=4 type=ud alias=V0176+0 align=2 words (r5.0)
//.declare V0179 (188)  rf=r size=4 type=d align=2 words (r8.0)
//.declare V0181 (190)  rf=r size=4 type=f align=2 words (r9.0)
//.declare V0182 (191)  rf=r size=4 type=ud alias=V0179+0 align=2 words (r8.0)
//.declare V0183 (192)  rf=r size=4 type=d align=2 words (r10.0)
//.declare V0184 (193)  rf=r size=4 type=f align=2 words (r11.0)
//.declare V0185 (194)  rf=r size=4 type=ud alias=V0183+0 align=2 words (r10.0)
//.declare V0186 (195)  rf=r size=4 type=f align=2 words (r13.0)
//.declare V0187 (196)  rf=r size=4 type=f align=2 words (r12.0)
//.declare V0188 (197)  rf=r size=4 type=f align=2 words (r15.0)
//.declare V0189 (198)  rf=r size=4 type=f align=2 words (r14.0)
//.declare V0190 (199)  rf=r size=32 type=d align=16 words (r16.0)
//.declare V0191 (200)  rf=r size=32 type=f align=16 words (r17.0)
//.declare V0192 (201)  rf=r size=32 type=ud alias=V0190+0 align=16 words (r16.0)
//.declare V0193 (202)  rf=r size=4 type=f align=2 words (r18.0)
//.declare V0194 (203)  rf=r size=4 type=f align=2 words (r20.0)
//.declare V0195 (204)  rf=r size=4 type=f align=2 words (r19.0)
//.declare V0196 (205)  rf=r size=32 type=f align=16 words (r21.0)
//.declare V0197 (206)  rf=r size=32 type=f align=16 words (r22.0)
//.declare V0198 (207)  rf=r size=32 type=f align=16 words (r23.0)
//.declare V0199 (208)  rf=r size=4 type=f align=2 words (r24.0)
//.declare V0200 (209)  rf=r size=32 type=f align=16 words (r25.0)
//.declare V0201 (210)  rf=r size=32 type=d align=16 words (r26.0)
//.declare V0202 (211)  rf=r size=32 type=d align=16 words (r27.0)
//.declare V0203 (212)  rf=r size=32 type=d align=16 words (r28.0)
//.declare V0204 (213)  rf=r size=32 type=f align=16 words (r29.0)
//.declare V0205 (214)  rf=r size=32 type=ud alias=V0201+0 align=16 words (r26.0)
//.declare V0206 (215)  rf=r size=32 type=f align=16 words (r30.0)
//.declare V0207 (216)  rf=r size=32 type=f align=16 words (r31.0)
//.declare V0208 (217)  rf=r size=32 type=f align=16 words (r32.0)
//.declare V0209 (218)  rf=r size=4 type=f align=2 words (r33.0)
//.declare V0210 (219)  rf=r size=32 type=f align=16 words (r34.0)
//.declare V0211 (220)  rf=r size=4 type=f align=2 words (r35.0)
//.declare V0212 (221)  rf=r size=32 type=f align=16 words (r36.0)
//.declare V0213 (222)  rf=r size=32 type=f align=16 words (r37.0)
//.declare V0214 (223)  rf=r size=32 type=f align=16 words (r38.0)
//.declare V0215 (224)  rf=r size=32 type=f align=16 words (r40.0)
//.declare V0216 (225)  rf=r size=4 type=f align=2 words (r39.0)
//.declare V0217 (226)  rf=r size=32 type=f align=16 words (r41.0)
//.declare V0219 (228)  rf=r size=32 type=d align=16 words (r42.0)
//.declare V0220 (229)  rf=r size=32 type=f align=16 words (r43.0)
//.declare V0221 (230)  rf=r size=32 type=ud alias=V0219+0 align=16 words (r42.0)
//.declare V0222 (231)  rf=r size=32 type=f align=16 words (r44.0)
//.declare V0223 (232)  rf=r size=32 type=f align=16 words (r45.0)
//.declare V0224 (233)  rf=r size=32 type=f align=16 words (r46.0)
//.declare V0225 (234)  rf=r size=32 type=f align=16 words (r47.0)
//.declare V0226 (235)  rf=r size=32 type=f align=16 words (r48.0)
//.declare V0227 (236)  rf=r size=32 type=f align=16 words (r49.0)
//.declare V0228 (237)  rf=r size=32 type=f align=16 words (r50.0)
//.declare V0229 (238)  rf=r size=32 type=f align=16 words (r51.0)
//.declare V0230 (239)  rf=r size=32 type=f align=16 words (r52.0)
//.declare V0231 (240)  rf=r size=4 type=f align=2 words (r53.0)
//.declare V0232 (241)  rf=r size=32 type=f align=16 words (r54.0)
//.declare V0233 (242)  rf=r size=32 type=f align=16 words (r55.0)
//.declare V0234 (243)  rf=r size=32 type=f align=16 words (r56.0)
//.declare V0235 (244)  rf=r size=32 type=f align=16 words (r57.0)
//.declare V0236 (245)  rf=r size=32 type=f align=16 words (r58.0)
//.declare V0237 (246)  rf=r size=32 type=d align=16 words (r59.0)
//.declare V0238 (247)  rf=r size=32 type=d align=16 words (r60.0)
//.declare V0239 (248)  rf=r size=32 type=d align=16 words (r61.0)
//.declare V0240 (249)  rf=r size=32 type=f align=16 words (r62.0)
//.declare V0241 (250)  rf=r size=32 type=f align=16 words (r63.0)
//.declare V0242 (251)  rf=r size=32 type=f align=16 words (r64.0)
//.declare V0243 (252)  rf=r size=32 type=f align=16 words (r65.0)
//.declare V0244 (253)  rf=r size=32 type=f align=16 words (r66.0)
//.declare V0245 (254)  rf=r size=32 type=f align=16 words (r67.0)
//.declare V0246 (255)  rf=r size=32 type=d align=16 words (r68.0)
//.declare V0247 (256)  rf=r size=32 type=d align=16 words (r69.0)
//.declare V0248 (257)  rf=r size=32 type=d align=16 words (r70.0)
//.declare V0249 (258)  rf=r size=32 type=d align=16 words (r71.0)
//.declare V0250 (259)  rf=r size=32 type=d align=16 words (r72.0)
//.declare V0251 (260)  rf=r size=32 type=ud alias=V0247+0 align=16 words (r69.0)
//.declare V0252 (261)  rf=r size=32 type=d align=16 words (r73.0)
//.declare V0253 (262)  rf=r size=32 type=f align=16 words (r74.0)
//.declare V0254 (263)  rf=r size=32 type=f align=16 words (r75.0)
//.declare V0255 (264)  rf=r size=32 type=f align=16 words (r77.0)
//.declare V0256 (265)  rf=r size=4 type=f align=2 words (r76.0)
//.declare V0257 (266)  rf=r size=32 type=d align=16 words (r78.0)
//.declare V0258 (267)  rf=r size=32 type=ud alias=V0257+0 align=16 words (r78.0)
//.declare V0259 (268)  rf=r size=32 type=d align=16 words (r79.0)
//.declare V0260 (269)  rf=r size=32 type=d align=16 words (r80.0)
//.declare V0261 (270)  rf=r size=32 type=ud alias=V0259+0 align=16 words (r79.0)
//.declare V0262 (271)  rf=r size=32 type=ud alias=V0260+0 align=16 words (r80.0)
//.declare V0263 (272)  rf=r size=32 type=ud alias=V0248+0 align=16 words (r70.0)
//.declare V0264 (273)  rf=r size=32 type=ud alias=V0252+0 align=16 words (r73.0)
//.declare V0266 (275)  rf=r size=32 type=d align=16 words (r81.0)
//.declare V0267 (276)  rf=r size=32 type=d align=16 words (r82.0)
//.declare V0268 (277)  rf=r size=32 type=ud alias=V0266+0 align=16 words (r81.0)
//.declare V0269 (278)  rf=r size=32 type=ud alias=V0267+0 align=16 words (r82.0)
//.declare V0270 (279)  rf=r size=32 type=ud alias=V0250+0 align=16 words (r72.0)
//.declare V0271 (280)  rf=r size=32 type=ud alias=V0249+0 align=16 words (r71.0)
//.declare V0273 (282)  rf=r size=32 type=d align=16 words (r83.0)
//.declare V0274 (283)  rf=r size=32 type=d align=16 words (r84.0)
//.declare V0275 (284)  rf=r size=32 type=d align=16 words (r85.0)
//.declare V0276 (285)  rf=r size=32 type=ud alias=V0238+0 align=16 words (r60.0)
//.declare V0277 (286)  rf=r size=32 type=d align=16 words (r86.0)
//.declare V0278 (287)  rf=r size=32 type=d align=16 words (r87.0)
//.declare V0279 (288)  rf=r size=32 type=ud alias=V0277+0 align=16 words (r86.0)
//.declare V0280 (289)  rf=r size=32 type=ud alias=V0278+0 align=16 words (r87.0)
//.declare V0281 (290)  rf=r size=32 type=ud alias=V0274+0 align=16 words (r84.0)
//.declare V0282 (291)  rf=r size=32 type=ud alias=V0275+0 align=16 words (r85.0)
//.declare V0283 (292)  rf=r size=32 type=ud alias=V0273+0 align=16 words (r83.0)
//.declare P02 (294)  rf=f8  size=2 type=uw align=1 words (f0.1)
//.declare P03 (295)  rf=f8  size=2 type=uw align=1 words (f0.0)
//.declare P04 (296)  rf=f8  size=2 type=uw align=1 words (f1.0)
//.declare V0285 (297)  rf=r size=32 type=d align=16 words (r89.0)
//.declare V0286 (298)  rf=r size=32 type=d align=16 words (r90.0)
//.declare V0287 (299)  rf=r size=32 type=d align=16 words (r112.0)
//.declare P05 (300)  rf=f8  size=2 type=uw align=1 words (f0.0)
//.declare V0288 (301)  rf=r size=4 type=f align=2 words (r2.0)
//.declare V0289 (302)  rf=r size=4 type=d align=2 words (r3.0)
//.declare V0290 (303)  rf=r size=4 type=ud alias=V0289+0 align=2 words (r3.0)
//.declare V0291 (304)  rf=r size=4 type=d align=2 words (r4.0)
//.declare V0292 (305)  rf=r size=32 type=f align=16 words (r5.0)
//.declare V0293 (306)  rf=r size=4 type=f align=2 words (r6.0)
//.declare V0294 (307)  rf=r size=4 type=f align=2 words (r8.0)
//.declare V0295 (308)  rf=r size=32 type=f align=16 words (r9.0)
//.declare V0296 (309)  rf=r size=32 type=d align=16 words (r10.0)
//.declare V0297 (310)  rf=r size=32 type=ud alias=V0296+0 align=16 words (r10.0)
//.declare V0298 (311)  rf=r size=32 type=d align=16 words (r11.0)
//.declare V0299 (312)  rf=r size=32 type=ud alias=V0298+0 align=16 words (r11.0)
//.declare V0300 (313)  rf=r size=4 type=f align=2 words (r12.0)
//.declare V0301 (314)  rf=r size=4 type=ud alias=V0291+0 align=2 words (r4.0)
//.declare V0302 (315)  rf=r size=32 type=f align=16 words (r13.0)
//.declare V0303 (316)  rf=r size=32 type=f align=16 words (r14.0)
//.declare V0305 (318)  rf=r size=32 type=f align=16 words (r15.0)
//.declare V0307 (320)  rf=r size=32 type=f align=16 words (r16.0)
//.declare V0308 (321)  rf=r size=32 type=f align=16 words (r17.0)
//.declare V0309 (322)  rf=r size=32 type=f align=16 words (r18.0)
//.declare V0310 (323)  rf=r size=32 type=d align=16 words (r19.0)
//.declare V0311 (324)  rf=r size=32 type=ud alias=V0310+0 align=16 words (r19.0)
//.declare V0312 (325)  rf=r size=32 type=d align=16 words (r20.0)
//.declare V0313 (326)  rf=r size=32 type=d align=16 words (r21.0)
//.declare V0314 (327)  rf=r size=32 type=ud alias=V0312+0 align=16 words (r20.0)
//.declare V0315 (328)  rf=r size=32 type=ud alias=V0313+0 align=16 words (r21.0)
//.declare  (329)  rf=f16  size=2 type=uw align=1 words (f1.1)
//.declare V0316 (330)  rf=r size=32 type=d align=16 words (r2.0)
//.declare V0317 (331)  rf=r size=32 type=d align=16 words (r3.0)
//.declare V0318 (332)  rf=r size=4 type=d align=16 words (r4.0)
//.declare V0319 (333)  rf=r size=4 type=d align=2 words (r6.0)
//.declare V0320 (334)  rf=r size=4 type=ud alias=V0318+0 align=2 words (r4.0)
//.declare V0321 (335)  rf=r size=4 type=ud alias=V0319+0 align=2 words (r6.0)
//.declare V0322 (336)  rf=r size=4 type=ud alias=V0100+0 align=2 words (r107.0)
//.declare V0323 (337)  rf=r size=8 type=ud alias=V0077+0 align=2 words (r106.2)
//.declare V0324 (338)  rf=r size=4 type=ud align=16 words (r5.0)
//.declare V0326 (340)  rf=r size=4 type=d align=2 words (r7.0)
//.declare V0327 (341)  rf=r size=4 type=d align=2 words (r8.0)
//.declare V0328 (342)  rf=r size=4 type=d align=2 words (r9.0)
//.declare V0329 (343)  rf=r size=4 type=d align=2 words (r10.0)
//.declare V0330 (344)  rf=r size=4 type=d align=2 words (r11.0)
//.declare V0331 (345)  rf=r size=4 type=d align=2 words (r12.0)
//.declare V0332 (346)  rf=r size=32 type=ud alias=V0317+0 align=16 words (r3.0)
//.declare V0333 (347)  rf=r size=4 type=ud align=16 words (r13.0)
//.declare V0334 (348)  rf=r size=32 type=d align=16 words (r22.0)
//.declare V0335 (349)  rf=r size=32 type=ud alias=V0334+0 align=16 words (r22.0)
//.declare V0336 (350)  rf=r size=4 type=ud align=16 words (r23.0)
//.declare V0337 (351)  rf=r size=32 type=d align=16 words (r24.0)
//.declare V0338 (352)  rf=r size=32 type=ud alias=V0337+0 align=16 words (r24.0)
//.declare V0339 (353)  rf=r size=4 type=ud align=16 words (r25.0)
//.declare V0340 (354)  rf=r size=32 type=d align=16 words (r26.0)
//.declare V0341 (355)  rf=r size=32 type=ud alias=V0340+0 align=16 words (r26.0)
//.declare V0342 (356)  rf=r size=4 type=ud align=16 words (r27.0)
//.declare V0343 (357)  rf=r size=32 type=d align=16 words (r28.0)
//.declare V0344 (358)  rf=r size=32 type=ud alias=V0343+0 align=16 words (r28.0)
//.declare V0345 (359)  rf=r size=4 type=ud align=16 words (r29.0)
//.declare V0346 (360)  rf=r size=32 type=d align=16 words (r30.0)
//.declare V0347 (361)  rf=r size=32 type=ud alias=V0346+0 align=16 words (r30.0)
//.declare V0348 (362)  rf=r size=4 type=ud align=16 words (r31.0)
//.declare V0349 (363)  rf=r size=32 type=d align=16 words (r32.0)
//.declare V0350 (364)  rf=r size=32 type=ud alias=V0349+0 align=16 words (r32.0)
//.declare V0351 (365)  rf=r size=4 type=ud align=16 words (r33.0)
//.declare V0352 (366)  rf=r size=32 type=d align=16 words (r34.0)
//.declare V0353 (367)  rf=r size=32 type=ud alias=V0352+0 align=16 words (r34.0)
//.declare V0354 (368)  rf=r size=4 type=ud align=16 words (r35.0)
//.declare V0355 (369)  rf=r size=4 type=d align=2 words (r36.0)
//.declare V0358 (372)  rf=r size=4 type=ud align=16 words (r37.0)
//.declare V0361 (375)  rf=r size=4 type=ud align=16 words (r46.0)
//.declare V0364 (378)  rf=r size=4 type=ud align=16 words (r47.0)
//.declare V0367 (381)  rf=r size=4 type=ud align=16 words (r48.0)
//.declare V0370 (384)  rf=r size=4 type=ud align=16 words (r49.0)
//.declare V0373 (387)  rf=r size=4 type=ud align=16 words (r50.0)
//.declare V0376 (390)  rf=r size=4 type=ud align=16 words (r51.0)
//.declare V0379 (393)  rf=r size=4 type=ud align=16 words (r52.0)
//.declare V0380 (394)  rf=r size=256 type=f align=16 words (r62.0)
//.declare V0381 (395)  rf=r size=32 type=f align=16 words (r53.0)
//.declare V0382 (396)  rf=r size=256 type=f align=16 words (r54.0)
//.declare V0383 (397)  rf=r size=4 type=d align=2 words (r70.0)
//.declare V0384 (398)  rf=r size=4 type=d align=2 words (r71.0)
//.declare V0387 (401)  rf=r size=4 type=ud align=16 words (r72.0)
//.declare V0390 (404)  rf=r size=4 type=ud align=16 words (r81.0)
//.declare V0393 (407)  rf=r size=4 type=ud align=16 words (r82.0)
//.declare V0396 (410)  rf=r size=4 type=ud align=16 words (r83.0)
//.declare V0399 (413)  rf=r size=4 type=ud align=16 words (r84.0)
//.declare V0402 (416)  rf=r size=4 type=ud align=16 words (r85.0)
//.declare V0405 (419)  rf=r size=4 type=ud align=16 words (r86.0)
//.declare V0408 (422)  rf=r size=4 type=ud align=16 words (r87.0)
//.declare V0409 (423)  rf=r size=256 type=f align=16 words (r96.0)
//.declare V0411 (425)  rf=r size=256 type=f align=16 words (r88.0)
//.declare V0412 (426)  rf=r size=32 type=d align=16 words (r3.0)
//.declare V0413 (427)  rf=r size=32 type=ud alias=V0412+0 align=16 words (r3.0)
//.declare V0414 (428)  rf=r size=4 type=ud align=16 words (r4.0)
//.declare V0415 (429)  rf=r size=32 type=d align=16 words (r17.0)
//.declare V0416 (430)  rf=r size=32 type=ud alias=V0415+0 align=16 words (r17.0)
//.declare V0417 (431)  rf=r size=4 type=ud align=16 words (r18.0)
//.declare V0418 (432)  rf=r size=32 type=d align=16 words (r19.0)
//.declare V0419 (433)  rf=r size=32 type=ud alias=V0418+0 align=16 words (r19.0)
//.declare V0420 (434)  rf=r size=4 type=ud align=16 words (r20.0)
//.declare V0421 (435)  rf=r size=32 type=d align=16 words (r21.0)
//.declare V0422 (436)  rf=r size=32 type=ud alias=V0421+0 align=16 words (r21.0)
//.declare V0423 (437)  rf=r size=4 type=ud align=16 words (r22.0)
//.declare V0424 (438)  rf=r size=32 type=d align=16 words (r23.0)
//.declare V0425 (439)  rf=r size=32 type=ud alias=V0424+0 align=16 words (r23.0)
//.declare V0426 (440)  rf=r size=4 type=ud align=16 words (r24.0)
//.declare V0427 (441)  rf=r size=32 type=d align=16 words (r25.0)
//.declare V0428 (442)  rf=r size=32 type=ud alias=V0427+0 align=16 words (r25.0)
//.declare V0429 (443)  rf=r size=4 type=ud align=16 words (r26.0)
//.declare V0430 (444)  rf=r size=32 type=d align=16 words (r27.0)
//.declare V0431 (445)  rf=r size=32 type=ud alias=V0430+0 align=16 words (r27.0)
//.declare V0432 (446)  rf=r size=4 type=ud align=16 words (r28.0)
//.declare V0433 (447)  rf=r size=32 type=d align=16 words (r29.0)
//.declare V0434 (448)  rf=r size=32 type=ud alias=V0433+0 align=16 words (r29.0)
//.declare V0435 (449)  rf=r size=4 type=ud align=16 words (r30.0)
//.declare V0436 (450)  rf=r size=4 type=d align=2 words (r31.0)
//.declare V0439 (453)  rf=r size=4 type=ud align=16 words (r32.0)
//.declare V0442 (456)  rf=r size=4 type=ud align=16 words (r45.0)
//.declare V0445 (459)  rf=r size=4 type=ud align=16 words (r46.0)
//.declare V0448 (462)  rf=r size=4 type=ud align=16 words (r47.0)
//.declare V0451 (465)  rf=r size=4 type=ud align=16 words (r48.0)
//.declare V0454 (468)  rf=r size=4 type=ud align=16 words (r49.0)
//.declare V0457 (471)  rf=r size=4 type=ud align=16 words (r50.0)
//.declare V0460 (474)  rf=r size=4 type=ud align=16 words (r51.0)
//.declare V0463 (477)  rf=r size=4 type=ud align=16 words (r52.0)
//.declare V0466 (480)  rf=r size=4 type=ud align=16 words (r61.0)
//.declare V0469 (483)  rf=r size=4 type=ud align=16 words (r70.0)
//.declare V0472 (486)  rf=r size=4 type=ud align=16 words (r72.0)
//.declare V0475 (489)  rf=r size=4 type=ud align=16 words (r73.0)
//.declare V0478 (492)  rf=r size=4 type=ud align=16 words (r74.0)
//.declare V0481 (495)  rf=r size=4 type=ud align=16 words (r75.0)
//.declare V0484 (498)  rf=r size=4 type=ud align=16 words (r76.0)
//.declare V0485 (499)  rf=r size=4 type=d align=16 words (r77.0)
//.declare V0486 (500)  rf=r size=4 type=d align=2 words (r79.0)
//.declare V0487 (501)  rf=r size=4 type=ud alias=V0485+0 align=2 words (r77.0)
//.declare V0488 (502)  rf=r size=4 type=ud alias=V0486+0 align=2 words (r79.0)
//.declare V0489 (503)  rf=r size=4 type=ud alias=V0114+0 align=2 words (r107.5)
//.declare V0490 (504)  rf=r size=8 type=ud alias=V0083+0 align=2 words (r106.4)
//.declare V0491 (505)  rf=r size=4 type=ud align=16 words (r78.0)
//.declare V0493 (507)  rf=r size=4 type=d align=2 words (r80.0)
//.declare V0494 (508)  rf=r size=4 type=d align=2 words (r81.0)
//.declare V0495 (509)  rf=r size=4 type=d align=2 words (r82.0)
//.declare V0496 (510)  rf=r size=4 type=d align=2 words (r83.0)
//.declare V0497 (511)  rf=r size=4 type=d align=2 words (r84.0)
//.declare V0498 (512)  rf=r size=4 type=d align=2 words (r85.0)
//.declare V0499 (513)  rf=r size=32 type=d align=16 words (r86.0)
//.declare V0500 (514)  rf=r size=32 type=d align=16 words (r87.0)
//.declare V0501 (515)  rf=r size=4 type=d align=2 words (r88.0)
//.declare V0502 (516)  rf=r size=32 type=d align=16 words (r89.0)
//.declare V0503 (517)  rf=r size=32 type=f align=16 words (r90.0)
//.declare V0504 (518)  rf=r size=32 type=d alias=V0503+0 align=16 words (r90.0)
//.declare V0505 (519)  rf=r size=32 type=ud alias=V0502+0 align=16 words (r89.0)
//.declare V0506 (520)  rf=r size=4 type=ud align=16 words (r91.0)
//.declare V0507 (521)  rf=r size=32 type=d align=16 words (r92.0)
//.declare V0508 (522)  rf=r size=32 type=f align=16 words (r93.0)
//.declare V0509 (523)  rf=r size=32 type=d alias=V0508+0 align=16 words (r93.0)
//.declare V0510 (524)  rf=r size=32 type=ud alias=V0507+0 align=16 words (r92.0)
//.declare V0511 (525)  rf=r size=4 type=ud align=16 words (r94.0)
//.declare V0512 (526)  rf=r size=32 type=d align=16 words (r95.0)
//.declare V0513 (527)  rf=r size=32 type=f align=16 words (r104.0)
//.declare V0514 (528)  rf=r size=32 type=d alias=V0513+0 align=16 words (r104.0)
//.declare V0515 (529)  rf=r size=32 type=ud alias=V0512+0 align=16 words (r95.0)
//.declare V0516 (530)  rf=r size=4 type=ud align=16 words (r2.0)
//.declare V0517 (531)  rf=r size=32 type=d align=16 words (r3.0)
//.declare V0518 (532)  rf=r size=32 type=f align=16 words (r4.0)
//.declare V0519 (533)  rf=r size=32 type=d alias=V0518+0 align=16 words (r4.0)
//.declare V0520 (534)  rf=r size=32 type=ud alias=V0517+0 align=16 words (r3.0)
//.declare V0521 (535)  rf=r size=4 type=ud align=16 words (r5.0)
//.declare V0522 (536)  rf=r size=32 type=d align=16 words (r6.0)
//.declare V0523 (537)  rf=r size=32 type=f align=16 words (r7.0)
//.declare V0524 (538)  rf=r size=32 type=d alias=V0523+0 align=16 words (r7.0)
//.declare V0525 (539)  rf=r size=32 type=ud alias=V0522+0 align=16 words (r6.0)
//.declare V0526 (540)  rf=r size=4 type=ud align=16 words (r8.0)
//.declare V0527 (541)  rf=r size=32 type=d align=16 words (r9.0)
//.declare V0528 (542)  rf=r size=32 type=f align=16 words (r10.0)
//.declare V0529 (543)  rf=r size=32 type=d alias=V0528+0 align=16 words (r10.0)
//.declare V0530 (544)  rf=r size=32 type=ud alias=V0527+0 align=16 words (r9.0)
//.declare V0531 (545)  rf=r size=4 type=ud align=16 words (r11.0)
//.declare V0532 (546)  rf=r size=32 type=d align=16 words (r12.0)
//.declare V0533 (547)  rf=r size=32 type=f align=16 words (r13.0)
//.declare V0534 (548)  rf=r size=32 type=d alias=V0533+0 align=16 words (r13.0)
//.declare V0535 (549)  rf=r size=32 type=ud alias=V0532+0 align=16 words (r12.0)
//.declare V0536 (550)  rf=r size=4 type=ud align=16 words (r14.0)
//.declare V0537 (551)  rf=r size=32 type=d align=16 words (r15.0)
//.declare V0538 (552)  rf=r size=32 type=f align=16 words (r16.0)
//.declare V0539 (553)  rf=r size=32 type=d alias=V0538+0 align=16 words (r16.0)
//.declare V0540 (554)  rf=r size=32 type=ud alias=V0537+0 align=16 words (r15.0)
//.declare V0541 (555)  rf=r size=4 type=ud align=16 words (r17.0)
//.declare V0542 (556)  rf=r size=4 type=d align=2 words (r18.0)
//.declare V0543 (557)  rf=r size=4 type=d align=2 words (r19.0)
//.declare V0544 (558)  rf=r size=32 type=d align=16 words (r20.0)
//.declare V0545 (559)  rf=r size=32 type=f align=16 words (r21.0)
//.declare V0546 (560)  rf=r size=32 type=d alias=V0545+0 align=16 words (r21.0)
//.declare V0547 (561)  rf=r size=32 type=ud alias=V0544+0 align=16 words (r20.0)
//.declare V0548 (562)  rf=r size=4 type=ud align=16 words (r22.0)
//.declare V0549 (563)  rf=r size=32 type=d align=16 words (r23.0)
//.declare V0550 (564)  rf=r size=32 type=f align=16 words (r24.0)
//.declare V0551 (565)  rf=r size=32 type=d alias=V0550+0 align=16 words (r24.0)
//.declare V0552 (566)  rf=r size=32 type=ud alias=V0549+0 align=16 words (r23.0)
//.declare V0553 (567)  rf=r size=4 type=ud align=16 words (r25.0)
//.declare V0554 (568)  rf=r size=32 type=d align=16 words (r26.0)
//.declare V0555 (569)  rf=r size=32 type=f align=16 words (r27.0)
//.declare V0556 (570)  rf=r size=32 type=d alias=V0555+0 align=16 words (r27.0)
//.declare V0557 (571)  rf=r size=32 type=ud alias=V0554+0 align=16 words (r26.0)
//.declare V0558 (572)  rf=r size=4 type=ud align=16 words (r28.0)
//.declare V0559 (573)  rf=r size=32 type=d align=16 words (r29.0)
//.declare V0560 (574)  rf=r size=32 type=f align=16 words (r30.0)
//.declare V0561 (575)  rf=r size=32 type=d alias=V0560+0 align=16 words (r30.0)
//.declare V0562 (576)  rf=r size=32 type=ud alias=V0559+0 align=16 words (r29.0)
//.declare V0563 (577)  rf=r size=4 type=ud align=16 words (r31.0)
//.declare V0564 (578)  rf=r size=32 type=d align=16 words (r32.0)
//.declare V0565 (579)  rf=r size=32 type=f align=16 words (r33.0)
//.declare V0566 (580)  rf=r size=32 type=d alias=V0565+0 align=16 words (r33.0)
//.declare V0567 (581)  rf=r size=32 type=ud alias=V0564+0 align=16 words (r32.0)
//.declare V0568 (582)  rf=r size=4 type=ud align=16 words (r34.0)
//.declare V0569 (583)  rf=r size=32 type=d align=16 words (r35.0)
//.declare V0570 (584)  rf=r size=32 type=f align=16 words (r36.0)
//.declare V0571 (585)  rf=r size=32 type=d alias=V0570+0 align=16 words (r36.0)
//.declare V0572 (586)  rf=r size=32 type=ud alias=V0569+0 align=16 words (r35.0)
//.declare V0573 (587)  rf=r size=4 type=ud align=16 words (r37.0)
//.declare V0574 (588)  rf=r size=32 type=d align=16 words (r38.0)
//.declare V0575 (589)  rf=r size=32 type=f align=16 words (r39.0)
//.declare V0576 (590)  rf=r size=32 type=d alias=V0575+0 align=16 words (r39.0)
//.declare V0577 (591)  rf=r size=32 type=ud alias=V0574+0 align=16 words (r38.0)
//.declare V0578 (592)  rf=r size=4 type=ud align=16 words (r40.0)
//.declare V0579 (593)  rf=r size=32 type=d align=16 words (r41.0)
//.declare V0580 (594)  rf=r size=32 type=f align=16 words (r42.0)
//.declare V0581 (595)  rf=r size=32 type=d alias=V0580+0 align=16 words (r42.0)
//.declare V0582 (596)  rf=r size=32 type=ud alias=V0579+0 align=16 words (r41.0)
//.declare V0583 (597)  rf=r size=4 type=ud align=16 words (r43.0)
//.declare V0584 (598)  rf=r size=8 type=uq align=4 words (r5.0)
//.declare V0585 (599)  rf=r size=8 type=uq align=4 words (r5.1)
//.declare V0586 (600)  rf=r size=8 type=uq align=4 words (r5.2)
//.declare  (601)  rf=r size=32 type=ud align=16 words (r127.0)
//.declare  (602)  rf=r size=2 type=uw align=1 words (r38.0)
//.declare  (603)  rf=r size=4 type=d align=2 words (r7.0)
//.declare  (605)  rf=r size=2 type=w align=1 words (r88.0)
//.declare  (606)  rf=r size=4 type=f align=2 words (r7.0)
//.declare r0 (607)  rf=r size=32 type=ud align=16 words (r0.0)
//.declare rtmp (608)  rf=r size=32 type=ud align=16 words (r127.0)
//.declare inlineRegFromTDL (609)  rf=r size=32 type=ud align=16 words (r1.0)
//.declare inlineRegExpectedLocation (610)  rf=r size=32 type=ud align=16 words (r4.0)
//.declare  (611)  rf=r size=64 type=ud align=16 words (r1.0)
//.declare  (612)  rf=r size=32 type=ud align=16 words (r3.0)
//.declare  (613)  rf=r size=128 type=ud align=16 words (r5.0)
//.declare  (614)  rf=r size=64 type=ud align=16 words (r9.0)
//.declare  (615)  rf=r size=256 type=d align=16 words (r62.0)
//.declare  (616)  rf=r size=256 type=d align=16 words (r62.0)

// .inputs
// +----------+----------+--------+----------+------------------+
// | id       | type     |  bytes | at       | from             |
// +----------+----------+--------+----------+------------------+
// | V0040    | :w x 16  |   0x20 | r1       | pti[tid]+0x0     |
// | V0041    | :w x 16  |   0x20 | r2       | pti[tid]+0x20    |
// | V0042    | :w x 16  |   0x20 | r3       | pti[tid]+0x40    |
// | V0037    | :d x 8   |   0x20 | r4       | inline+0x0       |
// | V0584    | :uq      |    0x8 | r5       | cti+0x0          |
// | V0585    | :uq      |    0x8 | r5+0x8   | cti+0x8          |
// | V0586    | :uq      |    0x8 | r5+0x10  | cti+0x10         |
// | V0034    | :q       |    0x8 | r5+0x18  | cti+0x18         |
// | V0044    | :q       |    0x8 | r6       | cti+0x20         |
// | V0045    | :q       |    0x8 | r6+0x8   | cti+0x28         |
// | V0046    | :q       |    0x8 | r6+0x10  | cti+0x30         |
// | V0047    | :q       |    0x8 | r6+0x18  | cti+0x38         |
// | V0048    | :q       |    0x8 | r7       | cti+0x40         |
// | V0049    | :q       |    0x8 | r7+0x8   | cti+0x48         |
// | V0050    | :q       |    0x8 | r7+0x10  | cti+0x50         |
// | V0051    | :q       |    0x8 | r7+0x18  | cti+0x58         |
// | V0052    | :q       |    0x8 | r8       | cti+0x60         |
// | V0053    | :q       |    0x8 | r8+0x8   | cti+0x68         |
// | V0054    | :q       |    0x8 | r8+0x10  | cti+0x70         |
// | V0055    | :q       |    0x8 | r8+0x18  | cti+0x78         |
// | V0043    | :uq      |    0x8 | r9       | cti+0x80         |
// | V0056    | :d       |    0x4 | r9+0x8   | cti+0x88         |
// | V0057    | :d       |    0x4 | r9+0xC   | cti+0x8C         |
// | V0058    | :d       |    0x4 | r9+0x10  | cti+0x90         |
// | V0038    | :d x 3   |    0xC | r9+0x14  | cti+0x94         |
// | V0039    | :d x 3   |    0xC | r10      | cti+0xA0         |
// +----------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (8|M0)               r127.0<1>:ud  0x0:ud                                                //  ALU pipe: int; 
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0xC0:ud              {I@2}          //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     mad (1|M0)               r127.2<1>:ud  r127.2<0;0>:ud    r127.0<0;0>:uw    0x60:uw              {I@1} //  ALU pipe: int; 
(W)     mov (8|M0)               r4.0<1>:ud    r1.0<1;1,0>:ud                                        //  ALU pipe: int; 
(W)     send.dc0 (8|M0)          r1       r127    null:0  0x0            0x022843FD           {A@1,$0} // wr:1h+0, rd:2; oword aligned block read x4 // 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x40:uw              {$0.src}       //  ALU pipe: int; 
(W)     send.dc0 (8|M0)          r3       r127    null:0  0x0            0x021842FD           {A@1,$1} // wr:1h+0, rd:1; oword aligned block read x2 // 
        nop                                                                                          // 
        nop                                                                                          // 
// B001: Preds:{B000},  Succs:{B002}
// cross_thread_prolog:
(W)     mov (8|M0)               r127.0<1>:ud  0x0:ud                              {$1.src}          //  ALU pipe: int; 
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     send.dc0 (8|M0)          r5       r127    null:0  0x0            0x024844FD           {A@1,$2} // wr:1h+0, rd:4; oword aligned block read x8 // 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x80:uw              {$2.src}       //  ALU pipe: int; 
(W)     send.dc0 (8|M0)          r9       r127    null:0  0x0            0x022843FD           {A@1,$3} // wr:1h+0, rd:2; oword aligned block read x4 // 
// B002: Preds:{B001},  Succs:{B003, B004}
// _main:
(W)     mov (8|M0)               r105.0<1>:ud  r0.0<1;1,0>:ud                                        //  ALU pipe: int; $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {A@1}         // $1
        sync.nop                             null                             {Compacted,A@1}        // $3
        sync.allwr                           ($1,$2)                                                 // $3
(W)     mov (2|M0)               r3.0<1>:d     r6.2<1;1,0>:d                    {A@1}                //  ALU pipe: int; $3
(W)     mov (2|M0)               r5.0<1>:d     r6.4<1;1,0>:d                                         //  ALU pipe: int; $4
(W)     mov (2|M0)               r7.0<1>:d     r7.2<1;1,0>:d                                         //  ALU pipe: int; $6
(W)     mul (1|M0)               acc0.0<1>:ud  r5.0<0;1,0>:ud    r3.0<0;1,0>:uw   {I@2}              //  ALU pipe: int; $12
(W)     mach (1|M0)              r12.0<1>:ud   r5.0<0;1,0>:ud    r3.0<0;1,0>:ud   {AccWrEn}          //  ALU pipe: int; 
(W)     mov (2|M0)               r8.0<1>:d     r7.4<1;1,0>:d                                         //  ALU pipe: int; $7
(W)     mov (1|M0)               r11.0<1>:ud   acc0.0<0;1,0>:ud                 {Compacted}          //  ALU pipe: int; 
(W)     mul (1|M0)               acc0.0<1>:d   r5.0<0;1,0>:ud    r3.2<0;1,0>:uw                      //  ALU pipe: int; $15
(W)     mach (1|M0)              r13.0<1>:d    r5.0<0;1,0>:ud    r3.1<0;1,0>:d                       //  ALU pipe: int; $16
(W)     mul (1|M0)               acc0.0<1>:d   r3.0<0;1,0>:ud    r5.2<0;1,0>:uw                      //  ALU pipe: int; $17
(W)     add (1|M0)               r14.0<1>:d    r12.0<0;1,0>:d    r13.0<0;1,0>:d   {Compacted,I@2}    //  ALU pipe: int; $16 R{} IR{}{E:3,E:3,},  {BC=1}
(W)     mach (1|M0)              r13.0<1>:d    r3.0<0;1,0>:ud    r5.1<0;1,0>:d                       //  ALU pipe: int; $19
(W)     mul (1|M0)               acc0.0<1>:ud  r8.0<0;1,0>:ud    r7.0<0;1,0>:uw                      //  ALU pipe: int; $24
(W)     mach (1|M0)              r20.0<1>:ud   r8.0<0;1,0>:ud    r7.0<0;1,0>:ud   {AccWrEn}          //  ALU pipe: int; 
(W)     mov (2|M0)               r9.0<1>:d     r8.2<1;1,0>:d                    {$3.dst}             //  ALU pipe: int; $9
(W)     mov (1|M0)               r19.0<1>:ud   acc0.0<0;1,0>:ud                 {Compacted}          //  ALU pipe: int; 
(W)     mul (1|M0)               acc0.0<1>:d   r8.0<0;1,0>:ud    r7.2<0;1,0>:uw                      //  ALU pipe: int; $27
(W)     mov (2|M0)               r10.3<1>:d    r8.4<1;1,0>:d                                         //  ALU pipe: int; $10
(W)     mach (1|M0)              r21.0<1>:d    r8.0<0;1,0>:ud    r7.1<0;1,0>:d                       //  ALU pipe: int; $28
(W)     mul (1|M0)               acc0.0<1>:d   r7.0<0;1,0>:ud    r8.2<0;1,0>:uw                      //  ALU pipe: int; $29
(W)     add (1|M0)               r22.0<1>:d    r20.0<0;1,0>:d    r21.0<0;1,0>:d   {Compacted,I@2}    //  ALU pipe: int; $28 R{} IR{}{E:5,E:5,},  {BC=1}
(W)     mach (1|M0)              r21.0<1>:d    r7.0<0;1,0>:ud    r8.1<0;1,0>:d                       //  ALU pipe: int; $31
(W)     mul (1|M0)               acc0.0<1>:ud  r10.3<0;1,0>:ud   r9.0<0;1,0>:uw                      //  ALU pipe: int; $36
(W)     mach (1|M0)              r25.0<1>:ud   r10.3<0;1,0>:ud   r9.0<0;1,0>:ud   {AccWrEn}          //  ALU pipe: int; 
        mov (8|M0)               r33.0<2>:d    r2.0<1;1,0>:uw                   {$0.dst}             //  ALU pipe: int; $54
(W)     mov (1|M0)               r24.0<1>:ud   acc0.0<0;1,0>:ud                 {Compacted}          //  ALU pipe: int; 
(W)     mul (1|M0)               acc0.0<1>:d   r10.3<0;1,0>:ud   r9.2<0;1,0>:uw                      //  ALU pipe: int; $39
(W)     mach (1|M0)              r26.0<1>:d    r10.3<0;1,0>:ud   r9.1<0;1,0>:d                       //  ALU pipe: int; $40
        mov (8|M0)               r33.1<2>:d    0:w                                                   //  ALU pipe: int; $56
(W)     mul (1|M0)               acc0.0<1>:d   r9.0<0;1,0>:ud    r10.8<0;1,0>:uw                     //  ALU pipe: int; $41
(W)     add (1|M0)               r27.0<1>:d    r25.0<0;1,0>:d    r26.0<0;1,0>:d   {Compacted,I@3}    //  ALU pipe: int; $40
(W)     mach (1|M0)              r26.0<1>:d    r9.0<0;1,0>:ud    r10.4<0;1,0>:d                      //  ALU pipe: int; $43
(W)     mul (1|M0)               acc0.0<1>:d   r10.1<0;1,0>:d    r105.12<0;1,0>:uw                   //  ALU pipe: int; $48
(W)     mov (1|M0)               r35.0<2>:f    r33.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $59
(W)     mov (1|M0)               r35.1<2>:f    r33.1<0;1,0>:f                   {Compacted,I@5}      //  ALU pipe: float; $60
(W)     mach (1|M0)              r29.0<1>:d    r10.1<0;1,0>:d    r105.6<0;1,0>:d                     //  ALU pipe: int; $50
(W)     mul (1|M0)               acc0.0<1>:d   r10.0<0;1,0>:d    r105.2<0;1,0>:uw {Compacted}        //  ALU pipe: int; $51
(W)     mov (2|M0)               r36.0<1>:f    r35.0<1;1,0>:f                   {Compacted,F@1}      //  ALU pipe: float; $61
        add3 (8|M0)              r30.0<1>:d    r29.0<0;0>:d      r2.0<1;0>:uw      r4.1<0>:d        {I@2} //  ALU pipe: int; $50
(W)     mach (1|M0)              r31.0<1>:d    r10.0<0;1,0>:d    r105.1<0;1,0>:d  {Compacted}        //  ALU pipe: int; $53
        mov (8|M0)               r39.0<1>:d    r1.0<1;1,0>:uw                                        //  ALU pipe: int; $64
(W)     subb (1|M0)              r109.0<1>:ud  r30.0<0;1,0>:ud   r36.0<0;1,0>:ud  {AccWrEn,A@1}      //  ALU pipe: int; $62
        add3 (8|M0)              r32.0<1>:d    r31.0<0;0>:d      r1.0<1;0>:uw      r4.0<0>:d        {I@3} //  ALU pipe: int; $53
(W)     mov (1|M0)               r37.0<1>:ud   acc0.0<0;1,0>:ud                 {Compacted}          //  ALU pipe: int; $62
        subb (8|M0)              r110.0<1>:ud  r32.0<1;1,0>:ud   r39.0<1;1,0>:ud  {AccWrEn,Compacted,I@2} //  ALU pipe: int; $65
(W)     mov (2|M0)               r106.0<1>:d   r5.6<1;1,0>:d                                         //  ALU pipe: int; $2
        mov (8|M0)               r111.0<1>:ud  -acc0.0<1;1,0>:ud                {Compacted}          //  ALU pipe: int; $66
(W)     mov (2|M0)               r6.0<1>:d     r6.6<1;1,0>:d                                         //  ALU pipe: int; $5
        or (8|M0)                r43.0<1>:d    r111.0<1;1,0>:d   r106.1<0;1,0>:d  {Compacted,I@2}    //  ALU pipe: int; $72
(W)     mov (2|M0)               r106.2<1>:d   r7.6<1;1,0>:d                                         //  ALU pipe: int; $8
(W)     mov (2|M0)               r106.4<1>:d   r8.6<1;1,0>:d                                         //  ALU pipe: int; $11
(W)     shl (1|M0)               r16.0<1>:d    r11.0<0;1,0>:d    1:w               {Compacted}       //  ALU pipe: int; $20
(W)     mov (1|M0)               r106.7<1>:ud  r19.0<0;1,0>:ud                                       //  ALU pipe: int; $25
(W)     mov (1|M0)               r107.4<1>:ud  r24.0<0;1,0>:ud                                       //  ALU pipe: int; $37
(W)     addc (1|M0)              r40.0<1>:ud   r11.0<0;1,0>:ud   r6.0<0;1,0>:ud   {AccWrEn,I@7}      //  ALU pipe: int; $67
        cmp (8|M0)    (lt)f0.1   null<1>:ud    r43.0<1;1,0>:ud   0x1:uw              {I@7}           //  ALU pipe: int; $73
(W)     mov (1|M0)               r38.0<1>:hf   0x0:hf                                                //  ALU pipe: float; $63
(W)     add (1|M0)               r15.0<1>:d    r14.0<0;1,0>:d    r13.0<0;1,0>:d   {Compacted}        //  ALU pipe: int; $19
(W)     shl (1|M0)               r18.0<1>:d    r6.0<0;1,0>:d     1:w               {Compacted}       //  ALU pipe: int; $22
(W)     shl (1|M0)               r107.2<1>:d   r106.2<0;1,0>:d   1:w               {I@7}             //  ALU pipe: int; $34
(W)     shl (1|M0)               r107.7<1>:d   r106.4<0;1,0>:d   2:w               {I@7}             //  ALU pipe: int; $46
(W)     add (1|M0)               r17.0<1>:d    r16.0<0;1,0>:d    r9.2<0;1,0>:d    {Compacted,I@7}    //  ALU pipe: int; $21
(W)     shl (1|M0)               r23.0<1>:d    r106.7<0;1,0>:d   1:w               {Compacted,I@7}   //  ALU pipe: int; $32
(W)     shl (1|M0)               r28.0<1>:d    r107.4<0;1,0>:d   2:w               {Compacted,I@7}   //  ALU pipe: int; $44
(W)     mov (1|M0)               r41.0<1>:ud   acc0.0<0;1,0>:ud                 {Compacted}          //  ALU pipe: int; $67
(W)     shl (1|M0)               r108.2<1>:d   -r40.0<0;1,0>:d   1:w               {Compacted,I@7}   //  ALU pipe: int; $70
(W)     add (1|M0)               r107.0<1>:d   r22.0<0;1,0>:d    r21.0<0;1,0>:d   {Compacted}        //  ALU pipe: int; $31
(W)     add (1|M0)               r107.5<1>:d   r27.0<0;1,0>:d    r26.0<0;1,0>:d   {Compacted}        //  ALU pipe: int; $43 R{} IR{}{O:6,O:6,},  {BC=1}
(W)     add3 (1|M0)              r108.1<1>:ud  -r37.0<0;0>:ud    r38.0<0;0>:uw     -r36.1<0>:ud     {F@1} //  ALU pipe: int; $63 R{} IR{}{E:9,O:9,E:9,},  {BC=1}
(W)     add3 (1|M0)              r106.6<1>:d   r16.0<0;0>:d      r9.2<0;0>:d       r18.0<0>:d       {I@7} //  ALU pipe: int; $23
(W)     add (1|M0)               r107.1<1>:d   r23.0<0;1,0>:d    r9.3<0;1,0>:d    {I@7}              //  ALU pipe: int; $33
(W)     add3 (1|M0)              r107.3<1>:d   r23.0<0;0>:d      r9.3<0;0>:d       r107.2<0>:d       //  ALU pipe: int; $35
(W)     add (1|M0)               r107.6<1>:d   r28.0<0;1,0>:d    r9.4<0;1,0>:d    {I@7}              //  ALU pipe: int; $45
(W)     add3 (1|M0)              r108.0<1>:d   r28.0<0;0>:d      r9.4<0;0>:d       r107.7<0>:d       //  ALU pipe: int; $47
(W)     add3 (1|M0)              r42.0<1>:ud   r41.0<0;0>:ud     r15.0<0;0>:ud     r6.1<0>:ud       {I@7} //  ALU pipe: int; $68
(W)     add3 (1|M0)              r108.3<1>:d   r17.0<0;0>:d      r18.0<0;0>:d      r108.2<0>:d      {I@7} //  ALU pipe: int; $71
(f0.1)  goto (8|M0)                          _0_007            _0_007                                //  ALU pipe: int; $74
// B003: [inDivergent],  Preds:{B002},  Succs:{B007}
_0_008:
(W)     shr (1|M0)               r2.0<1>:d     r106.0<0;1,0>:ud  20:w               {Compacted}      //  ALU pipe: int; $76
(W)     shl (1|M0)               r3.0<1>:d     r106.1<0;1,0>:d   12:w               {Compacted}      //  ALU pipe: int; $77
(W)     mov (1|M0)               r7.0<1>:d     1044480:d                                             //  ALU pipe: int; $83
(W)     or (1|M0)                r4.0<1>:d     r2.0<0;1,0>:d     r3.0<0;1,0>:d    {Compacted,I@2}    //  ALU pipe: int; $79 R{} IR{}{O:0,O:0,},  {BC=1}
(W)     and (1|M0)               r10.0<1>:d    r106.0<0;1,0>:d   1048575:d                           //  ALU pipe: int; $85
(W)     bfn.(s0&s1|s2) (1|M0)    r8.0<1>:ud    r4.0<0;0>:ud      r7.0<0;0>:ud      r2.0<0>:ud       {I@2} //  ALU pipe: int; $83
(W)     shr (1|M0)               r5.0<1>:d     r106.1<0;1,0>:ud  8:w               {Compacted}       //  ALU pipe: int; $80
(W)     mov (1|M0)               r12.0<1>:ud   0x49800000:ud                                         //  ALU pipe: int; $87
(W)     mov (1|M0)               r11.0<1>:f    r10.0<0;1,0>:ud                  {Compacted,I@4}      //  ALU pipe: float; $86
(W)     mov (1|M0)               r9.0<1>:f     r8.0<0;1,0>:ud                   {Compacted,I@3}      //  ALU pipe: float; $84
(W)     mov (1|M0)               r14.0<1>:ud   0x53800000:ud                                         //  ALU pipe: int; $89
(W)     mov (1|M0)               r6.0<1>:f     r5.0<0;1,0>:ud                   {Compacted,I@3}      //  ALU pipe: float; $81
(W)     mad (1|M0)               r13.0<1>:f    r11.0<0;0>:f      r12.0<0;0>:f      r9.0<0>:f        {Compacted,A@2} //  ALU pipe: float; $88
        shr (8|M0)               r16.0<1>:d    r111.0<1;1,0>:ud  8:w               {Compacted}       //  ALU pipe: int; $91
(W)     mad (1|M0)               r15.0<1>:f    r13.0<0;0>:f      r14.0<0;0>:f      r6.0<0>:f        {Compacted,A@1} //  ALU pipe: float; $90
(W)     mov (1|M0)               r19.0<1>:ud   0xB4E00000:ud                                         //  ALU pipe: int; $94
        sync.nop                             null                             {Compacted,F@1}        // $93
(W)     math.inv (1|M0)          r18.0<1>:f    r15.0<0;1,0>:f                   {$4}                 //  ALU pipe: math; $93
        mov (8|M0)               r17.0<1>:f    r16.0<1;1,0>:ud                  {Compacted,I@2}      //  ALU pipe: float; $92
        sync.nop                             null                             {Compacted,$4.dst}     // $95
(W)     mad (1|M0)               r20.0<1>:f    r18.0<0;0>:f      r19.0<0;0>:f      r18.0<0>:f       {Compacted,I@1} //  ALU pipe: float; $95 R{} IR{r18,}{O:4,O:4,},  {BC=1}
        shr (8|M0)               r26.0<1>:d    r110.0<1;1,0>:ud  20:w               {Compacted}      //  ALU pipe: int; $101
        shl (8|M0)               r27.0<1>:d    r111.0<1;1,0>:d   12:w               {Compacted}      //  ALU pipe: int; $102
        mul (8|M0)               acc0.0<1>:f   r20.0<0;1,0>:f    r17.0<1;1,0>:f   {Compacted,F@1}    //  ALU pipe: float; $96
        or (8|M0)                r28.0<1>:d    r26.0<1;1,0>:d    r27.0<1;1,0>:d   {Compacted,I@1}    //  ALU pipe: int; $104 R{} IR{}{O:6,O:6,},  {BC=1}
        rndd (8|M0)              r22.0<1>:f    acc0.0<1;1,0>:f                  {Compacted}          //  ALU pipe: float; $97
(W)     mul (1|M0)               r24.0<1>:f    r9.0<0;1,0>:f     0xC9800000:f               {Compacted} //  ALU pipe: float; $99
        bfn.(s0&s1|s2) (8|M0)    r26.0<1>:ud   r28.0<1;0>:ud     r7.0<0;0>:ud      r26.0<1>:ud      {I@1} //  ALU pipe: int; $105
        mad (8|M0)               acc0.0<1>:f   r17.0<1;0>:f      r22.0<1;0>:f      -r11.0<0>:f      {Compacted,F@2} //  ALU pipe: float; $98
        mov (8|M0)               r29.0<1>:f    r26.0<1;1,0>:ud                  {Compacted,I@1}      //  ALU pipe: float; $106
        mad (8|M0)               acc0.0<1>:f   acc0.0<1;0>:f     r22.0<1;0>:f      r24.0<0>:f       {Compacted,F@3} //  ALU pipe: float; $100
(W)     mul (1|M0)               r35.0<1>:f    r11.0<0;1,0>:f    0x35800000:f               {Compacted} //  ALU pipe: float; $113
        mad (8|M0)               acc1.0<1>:f   r29.0<1;0>:f      acc0.0<1;0>:f     r12.0<0>:f       {Compacted,F@3} //  ALU pipe: float; $107
        and (8|M0)               r42.0<1>:d    r110.0<1;1,0>:d   1048575:d                           //  ALU pipe: int; $121
        mul (8|M0)               acc1.0<1>:f   r20.0<0;1,0>:f    acc1.0<1;1,0>:f                     //  ALU pipe: float; $108
(W)     mul (1|M0)               r33.0<1>:f    r6.0<0;1,0>:f     0xC9800000:f               {Compacted} //  ALU pipe: float; $111
        rndd (8|M0)              r31.0<1>:f    acc1.0<1;1,0>:f                  {Compacted}          //  ALU pipe: float; $109
(W)     mov (1|M0)               r39.0<1>:ud   0xBF800000:ud                                         //  ALU pipe: int; $117
        mul (8|M0)               acc1.0<1>:f   r35.0<0;1,0>:f    r31.0<1;1,0>:f   {Compacted,F@1}    //  ALU pipe: float; $114
        mad (8|M0)               acc0.0<1>:f   acc0.0<1;0>:f     r31.0<1;0>:f      -r9.0<0>:f        //  ALU pipe: float; $110
        rndd (8|M0)              acc1.0<1>:f   acc1.0<1;1,0>:f                                       //  ALU pipe: float; $115
        mov (8|M0)               r43.0<1>:f    r42.0<1;1,0>:ud                  {Compacted,I@2}      //  ALU pipe: float; $122
        mad (8|M0)               acc2.0<1>:f   -acc1.0<1;0>:f    r31.0<1;0>:f      r35.0<0>:f        //  ALU pipe: float; $116
        mad (8|M0)               acc0.0<1>:f   acc0.0<1;0>:f     r31.0<1;0>:f      r33.0<0>:f       {Compacted} //  ALU pipe: float; $112
        mad (8|M0)               acc2.0<1>:f   r29.0<1;0>:f      acc2.0<1;0>:f     -r12.0<0>:f       //  ALU pipe: float; $120
        mad (8|M0)               acc0.0<1>:f   acc0.0<1;0>:f     acc1.0<1;0>:f     r39.0<0>:f       {I@1} //  ALU pipe: float; $118
        mad (8|M0)               acc1.0<1>:f   r43.0<1;0>:f      acc2.0<1;0>:f     r12.0<0>:f       {Compacted,F@5} //  ALU pipe: float; $123
(W)     mul (1|M0)               r53.0<1>:f    r9.0<0;1,0>:f     0x35800000:f               {Compacted} //  ALU pipe: float; $133
        mad (8|M0)               acc1.0<1>:f   acc1.0<1;0>:f     acc0.0<1;0>:f     r14.0<0>:f        //  ALU pipe: float; $124
(W)     mov (1|M0)               r76.0<1>:ud   0xCF800000:ud                                         //  ALU pipe: int; $160
        mul (8|M0)               acc1.0<1>:f   r20.0<0;1,0>:f    acc1.0<1;1,0>:f                     //  ALU pipe: float; $125
(W)     mov (1|M0)               r88.0<1>:w    -1:w                                                  //  ALU pipe: int; $178
        rndd (8|M0)              r46.0<1>:f    acc1.0<1;1,0>:f                  {Compacted}          //  ALU pipe: float; $126
        mov (8|M0)               r60.0<1>:d    r31.0<1;1,0>:f                   {Compacted}          //  ALU pipe: int; $140
        mul (8|M0)               acc1.0<1>:f   r35.0<0;1,0>:f    r46.0<1;1,0>:f   {Compacted,F@1}    //  ALU pipe: float; $128
        mul (8|M0)               acc3.0<1>:f   r53.0<0;1,0>:f    r46.0<1;1,0>:f   {Compacted}        //  ALU pipe: float; $134
        rndd (8|M0)              acc1.0<1>:f   acc1.0<1;1,0>:f                                       //  ALU pipe: float; $129
        rndd (8|M0)              acc3.0<1>:f   acc3.0<1;1,0>:f                                       //  ALU pipe: float; $135
        mad (8|M0)               r47.0<1>:f    acc0.0<1;0>:f     r46.0<1;0>:f      -r6.0<0>:f        //  ALU pipe: float; $127
        mad (8|M0)               acc2.0<1>:f   acc2.0<1;0>:f     acc1.0<1;0>:f     r39.0<0>:f        //  ALU pipe: float; $131
        mad (8|M0)               acc0.0<1>:f   -acc3.0<1;0>:f    r46.0<1;0>:f      r53.0<0>:f        //  ALU pipe: float; $136
        mad (8|M0)               acc1.0<1>:f   -acc1.0<1;0>:f    r46.0<1;0>:f      r35.0<0>:f        //  ALU pipe: float; $130
        mad (8|M0)               acc3.0<1>:f   r47.0<1;0>:f      acc3.0<1;0>:f     r39.0<0>:f       {Compacted,F@4} //  ALU pipe: float; $137
        mad (8|M0)               acc0.0<1>:f   acc2.0<1;0>:f     acc0.0<1;0>:f     -r12.0<0>:f       //  ALU pipe: float; $138
        mad (8|M0)               acc1.0<1>:f   r43.0<1;0>:f      acc1.0<1;0>:f     -r12.0<0>:f       //  ALU pipe: float; $132
        mad (8|M0)               acc2.0<1>:f   acc0.0<1;0>:f     acc3.0<1;0>:f     r12.0<0>:f        //  ALU pipe: float; $142
        mov (8|M0)               r59.0<1>:d    r22.0<1;1,0>:f                   {Compacted}          //  ALU pipe: int; $139
        mad (8|M0)               acc2.0<1>:f   acc1.0<1;0>:f     acc2.0<1;0>:f     r12.0<0>:f        //  ALU pipe: float; $143
        mov (8|M0)               r61.0<1>:d    r46.0<1;1,0>:f                   {Compacted}          //  ALU pipe: int; $141
        mul (8|M0)               acc2.0<1>:f   r20.0<0;1,0>:f    acc2.0<1;1,0>:f                     //  ALU pipe: float; $144
        shl (8|M0)               r84.0<1>:d    r60.0<1;1,0>:d    20:w               {Compacted,I@3}  //  ALU pipe: int; $169
        rndd (8|M0)              r64.0<1>:f    acc2.0<1;1,0>:f                  {Compacted}          //  ALU pipe: float; $145
        shr (8|M0)               r85.0<1>:d    r60.0<1;1,0>:ud   12:w               {Compacted}      //  ALU pipe: int; $170
        mad (8|M0)               acc2.0<1>:f   acc3.0<1;0>:f     -r64.0<1;0>:f     r6.0<0>:f        {F@1} //  ALU pipe: float; $146
        mad (8|M0)               r66.0<1>:f    acc0.0<1;0>:f     -r64.0<1;0>:f     r9.0<0>:f         //  ALU pipe: float; $147
        mul (8|M0)               acc0.0<1>:f   acc2.0<1;1,0>:f   0x2F800000:f               {Compacted} //  ALU pipe: float; $158
        mad (8|M0)               r67.0<1>:f    acc1.0<1;0>:f     -r64.0<1;0>:f     r11.0<0>:f        //  ALU pipe: float; $148
        rndz (8|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f                                       //  ALU pipe: float; $159
        mov (8|M0)               r69.0<1>:d    r66.0<1;1,0>:f                   {Compacted,F@4}      //  ALU pipe: int; $150
        mad (8|M0)               r77.0<1>:f    acc2.0<1;0>:f     acc0.0<1;0>:f     r76.0<0>:f        //  ALU pipe: float; $161
        mov (8|M0)               r70.0<1>:d    r67.0<1;1,0>:f                   {Compacted,F@3}      //  ALU pipe: int; $151
        mov (8|M0)               r78.0<1>:ud   r77.0<1;1,0>:f                   {F@1}                //  ALU pipe: int; $162
        asr (8|M0)               r71.0<1>:d    r69.0<1;1,0>:d    31:w               {Compacted,I@3}  //  ALU pipe: int; $152
        shl (8|M0)               r72.0<1>:d    r69.0<1;1,0>:d    20:w               {Compacted}      //  ALU pipe: int; $153
        asr (8|M0)               r73.0<1>:d    r70.0<1;1,0>:d    31:w               {Compacted,I@4}  //  ALU pipe: int; $157
        addc (8|M0)              r79.0<1>:ud   r70.0<1;1,0>:ud   0x0:ud              {AccWrEn,Compacted} //  ALU pipe: int; $164
        shl (8|M0)               r78.0<1>:d    r78.0<1;1,0>:d    8:w               {Compacted,I@5}   //  ALU pipe: int; $163
        shr (8|M0)               acc2.0<1>:d   r69.0<1;1,0>:ud   12:w                                //  ALU pipe: int; $154
        shl (8|M0)               r71.0<1>:d    r71.0<1;1,0>:d    20:w               {Compacted,I@6}  //  ALU pipe: int; $155
        add3 (8|M0)              r80.0<1>:ud   acc0.0<1;0>:ud    r78.0<1;0>:ud     r73.0<1>:ud      {I@3} //  ALU pipe: int; $165
        addc (8|M0)              r81.0<1>:ud   r79.0<1;1,0>:ud   r72.0<1;1,0>:ud  {AccWrEn,Compacted} //  ALU pipe: int; $166
        or (8|M0)                r71.0<1>:d    r71.0<1;1,0>:d    acc2.0<1;1,0>:d  {I@3}              //  ALU pipe: int; $156
        cmp (8|M0)    (ge)f1.0   null<1>:ud    r81.0<1;1,0>:ud   r106.0<0;1,0>:ud {I@2}              //  ALU pipe: int; $173
        add3 (8|M0)              r82.0<1>:ud   acc0.0<1;0>:ud    r80.0<1;0>:ud     r71.0<1>:ud      {I@2} //  ALU pipe: int; $167
        mov (8|M0)               r68.0<1>:d    r64.0<1;1,0>:f                   {Compacted}          //  ALU pipe: int; $149
        shl (8|M0)               r83.0<1>:d    r59.0<1;1,0>:d    8:w               {Compacted}       //  ALU pipe: int; $168
(f1.0)  cmp (8|M0)    (eq)f1.0   null<1>:d     r82.0<1;1,0>:d    r106.1<0;1,0>:d  {I@3}              //  ALU pipe: int; $174
        addc (8|M0)              r86.0<1>:ud   r84.0<1;1,0>:ud   0x0:ud              {AccWrEn,Compacted} //  ALU pipe: int; $171
        add3 (8|M0)              r87.0<1>:ud   acc0.0<1;0>:ud    r85.0<1;0>:ud     r83.0<1>:ud      {I@3} //  ALU pipe: int; $172
(~f1.0) cmp (8|M0)    (gt)f1.0   null<1>:ud    r82.0<1;1,0>:ud   r106.1<0;1,0>:ud                    //  ALU pipe: int; $176
(f1.0)  sel (8|M0)               r89.0<1>:d    r88.0<0;1,0>:w    0:w                                 //  ALU pipe: int; $178
        add3 (8|M0)              acc0.0<1>:d   r68.0<1;0>:d      r61.0<1;0>:d      -r89.0<1>:d      {I@1} //  ALU pipe: int; $179 R{} IR{}{E:1,E:15,E:6,},  {BC=1}
        add (8|M0)               r112.0<1>:d   r86.0<1;1,0>:d    acc0.0<1;1,0>:d                     //  ALU pipe: int; $180
        goto (8|M0)                          _0_007            _0_009                                // $181
// B004: [inDivergent],  Preds:{B002},  Succs:{B005, B006}
_0_007:
        join (8|M0)                          _0_009                                                  // 
L2616:
(W)     cmp (8|M0)    (eq)f0.0   null<1>:d     r106.0<0;1,0>:d   0:w                                 //  ALU pipe: int; $183
(f0.0)  goto (8|M0)                          _0_010            _0_010                                //  ALU pipe: int; $184
// B005: [inDivergent],  Preds:{B004},  Succs:{B007}
_0_011:
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}          // $186
(W)     mov (1|M0)               r2.0<1>:f     r106.0<0;1,0>:ud                 {Compacted,A@1}      //  ALU pipe: float; $187
(W)     mov (1|M0)               r7.0<1>:f     0xB4C00000:f                               {Compacted} //  ALU pipe: float; $192
(W)     math.inv (1|M0)          r6.0<1>:f     r2.0<0;1,0>:f                    {@2,$5}              //  ALU pipe: math; $191
        mov (8|M0)               r5.0<1>:f     r110.0<1;1,0>:ud                 {Compacted}          //  ALU pipe: float; $190
        sync.nop                             null                             {Compacted,$5.dst}     // $192
(W)     mad (1|M0)               r8.0<1>:f     r6.0<0;0>:f       r7.0<0;0>:f       r6.0<0>:f        {Compacted,F@2} //  ALU pipe: float; $192 R{} IR{r6,}{O:1,O:1,},  {BC=1}
(W)     mov (1|M0)               r3.0<1>:ud    r2.0<0;1,0>:f                                         //  ALU pipe: int; $188
        mov (8|M0)               r10.0<1>:ud   r5.0<1;1,0>:f                    {F@2}                //  ALU pipe: int; $194
        mul (8|M0)               r9.0<1>:f     r5.0<1;1,0>:f     r8.0<0;1,0>:f    {Compacted,F@1}    //  ALU pipe: float; $193
(W)     add (1|M0)               r4.0<1>:d     r106.0<0;1,0>:d   -r3.0<0;1,0>:d   {Compacted,I@2}    //  ALU pipe: int; $189
        add (8|M0)               r10.0<1>:d    r110.0<1;1,0>:d   -r10.0<1;1,0>:d  {Compacted,I@2}    //  ALU pipe: int; $195
        mov (8|M0)               r11.0<1>:ud   r9.0<1;1,0>:f                    {F@1}                //  ALU pipe: int; $196
(W)     mov (1|M0)               r12.0<1>:f    r4.0<0;1,0>:ud                   {Compacted,I@3}      //  ALU pipe: float; $197
        mov (8|M0)               r13.0<1>:f    r10.0<1;1,0>:ud                  {Compacted,I@2}      //  ALU pipe: float; $198
        mov (8|M0)               r14.0<1>:f    r11.0<1;1,0>:ud                  {Compacted,I@1}      //  ALU pipe: float; $199
        mad (8|M0)               acc0.0<1>:f   r5.0<1;0>:f       r14.0<1;0>:f      -r2.0<0>:f       {Compacted,F@1} //  ALU pipe: float; $201
        mad (8|M0)               acc1.0<1>:f   r13.0<1;0>:f      r14.0<1;0>:f      -r12.0<0>:f      {Compacted} //  ALU pipe: float; $203 R{r14,} IR{}{E:3,E:3,},  {BC=1}
        add (8|M0)               acc0.0<1>:f   acc0.0<1;1,0>:f   acc1.0<1;1,0>:f                     //  ALU pipe: float; $204
        mul (8|M0)               r18.0<1>:f    r8.0<0;1,0>:f     acc0.0<1;1,0>:f                     //  ALU pipe: float; $205
(W)     xor (1|M0)               cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x30:uw              {A@1}          // $206
        mov (8|M0)               r19.0<1>:ud   r18.0<1;1,0>:f                   {A@1}                //  ALU pipe: int; $207
        add (8|M0)               r20.0<1>:d    r19.0<1;1,0>:d    r11.0<1;1,0>:d   {Compacted,I@1}    //  ALU pipe: int; $208
(W)     mul (8|M0)               acc0.0<1>:d   r20.0<1;1,0>:d    r106.0<0;1,0>:uw {Compacted,I@1}    //  ALU pipe: int; $209
        mach (8|M0)              r20.0<1>:d    r20.0<1;1,0>:d    r106.0<0;1,0>:d  {Compacted}        //  ALU pipe: int; $210
        add (8|M0)               r20.0<1>:d    r110.0<1;1,0>:d   -r20.0<1;1,0>:d  {Compacted,I@1}    //  ALU pipe: int; $210
        cmp (8|M0)    (ge)f1.1   r21.0<1>:ud   r20.0<1;1,0>:ud   r106.0<0;1,0>:ud {I@1}              //  ALU pipe: int; $211
        add3 (8|M0)              r112.0<1>:d   r19.0<1;0>:d      r11.0<1;0>:d      -r21.0<1>:d      {I@1} //  ALU pipe: int; $212
        goto (8|M0)                          _0_010            _0_009                                // $213
// B006: [inDivergent],  Preds:{B004},  Succs:{B007}
_0_010:
        join (8|M0)                          _0_009                                                  // 
L2992:
        mov (8|M0)               r112.0<1>:d   -1:w                                                  //  ALU pipe: int; $215
// B007: Preds:{B006, B005, B003},  Succs:{}
_0_009:
        join (8|M0)                          L3024                                                   // 
L3024:
        shl (8|M0)               r2.0<1>:d     r112.0<1;1,0>:d   4:w               {Compacted,I@2}   //  ALU pipe: int; $217
(W)     addc (1|M0)              r4.0<1>:ud    r106.7<0;1,0>:ud  r106.2<0;1,0>:ud {AccWrEn}          //  ALU pipe: int; $220
(W)     shl (1|M0)               r9.0<1>:d     r109.0<0;1,0>:d   9:w               {Compacted}       //  ALU pipe: int; $225
        shl (8|M0)               r2.0<1>:d     r2.0<1;1,0>:d     1:w               {Compacted,I@3}   //  ALU pipe: int; $218
(W)     shl (1|M0)               r7.0<1>:d     -r4.0<0;1,0>:d    1:w               {Compacted,I@3}   //  ALU pipe: int; $223
(W)     add (1|M0)               r70.0<1>:d    r9.0<0;1,0>:d     256:w               {Compacted,I@3} //  ALU pipe: int; $287
(W)     shl (1|M0)               r36.0<1>:d    r9.0<0;1,0>:d     1:w               {Compacted}       //  ALU pipe: int; $252
        add3 (8|M0)              r3.0<1>:d     r106.6<0;0>:d     r108.2<0;0>:d     r2.0<1>:d        {I@4} //  ALU pipe: int; $219
        add3 (8|M0)              r22.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       64:w               //  ALU pipe: int; $231
        add3 (8|M0)              r24.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       128:w               //  ALU pipe: int; $234
        add3 (8|M0)              r26.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       192:w               //  ALU pipe: int; $237
        add3 (8|M0)              r28.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       256:w               //  ALU pipe: int; $240
        add3 (8|M0)              r30.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       320:w               //  ALU pipe: int; $243
        add3 (8|M0)              r32.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       384:w               //  ALU pipe: int; $246
        add3 (8|M0)              r34.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       448:w               //  ALU pipe: int; $249
(W)     add3 (1|M0)              r8.0<1>:d     r107.1<0;0>:d     r107.2<0;0>:d     r7.0<0>:d        {I@7} //  ALU pipe: int; $224
(W)     shl (1|M0)               r71.0<1>:d    r70.0<0;1,0>:d    1:w               {Compacted,I@7}   //  ALU pipe: int; $288
(W)     mov (1|M0)               r13.0<1>:f    r3.0<0;1,0>:f                    {Compacted,I@7}      //  ALU pipe: float; $229
(W)     mov (1|M0)               r23.0<1>:f    r22.0<0;1,0>:f                   {Compacted,I@7}      //  ALU pipe: float; $232
(W)     mov (1|M0)               r25.0<1>:f    r24.0<0;1,0>:f                   {Compacted,I@7}      //  ALU pipe: float; $235
(W)     mov (1|M0)               r27.0<1>:f    r26.0<0;1,0>:f                   {Compacted,I@7}      //  ALU pipe: float; $238
(W)     mov (1|M0)               r29.0<1>:f    r28.0<0;1,0>:f                   {Compacted,I@6}      //  ALU pipe: float; $241
(W)     mov (1|M0)               r31.0<1>:f    r30.0<0;1,0>:f                   {Compacted,I@5}      //  ALU pipe: float; $244
(W)     mov (1|M0)               r33.0<1>:f    r32.0<0;1,0>:f                   {Compacted,I@4}      //  ALU pipe: float; $247
(W)     mov (1|M0)               r35.0<1>:f    r34.0<0;1,0>:f                   {Compacted,I@3}      //  ALU pipe: float; $250
(W)     add (1|M0)               r37.0<1>:d    r8.0<0;1,0>:d     r36.0<0;1,0>:d   {Compacted,I@2}    //  ALU pipe: int; $253
(W)     add3 (1|M0)              r46.0<1>:d    r8.0<0;0>:d       r36.0<0;0>:d      64:w               //  ALU pipe: int; $256
(W)     add3 (1|M0)              r47.0<1>:d    r8.0<0;0>:d       r36.0<0;0>:d      128:w               //  ALU pipe: int; $259
(W)     add3 (1|M0)              r48.0<1>:d    r8.0<0;0>:d       r36.0<0;0>:d      192:w               //  ALU pipe: int; $262
(W)     add3 (1|M0)              r49.0<1>:d    r8.0<0;0>:d       r36.0<0;0>:d      256:w               //  ALU pipe: int; $265
(W)     add3 (1|M0)              r50.0<1>:d    r8.0<0;0>:d       r36.0<0;0>:d      320:w               //  ALU pipe: int; $268
(W)     add3 (1|M0)              r51.0<1>:d    r8.0<0;0>:d       r36.0<0;0>:d      384:w               //  ALU pipe: int; $271
(W)     add3 (1|M0)              r52.0<1>:d    r8.0<0;0>:d       r36.0<0;0>:d      448:w               //  ALU pipe: int; $274
(W)     add (1|M0)               r72.0<1>:d    r8.0<0;1,0>:d     r71.0<0;1,0>:d   {Compacted,I@7}    //  ALU pipe: int; $289
(W)     add3 (1|M0)              r81.0<1>:d    r8.0<0;0>:d       r71.0<0;0>:d      64:w               //  ALU pipe: int; $292
(W)     add3 (1|M0)              r82.0<1>:d    r8.0<0;0>:d       r71.0<0;0>:d      128:w               //  ALU pipe: int; $295
(W)     add3 (1|M0)              r83.0<1>:d    r8.0<0;0>:d       r71.0<0;0>:d      192:w               //  ALU pipe: int; $298
(W)     add3 (1|M0)              r84.0<1>:d    r8.0<0;0>:d       r71.0<0;0>:d      256:w               //  ALU pipe: int; $301
(W)     add3 (1|M0)              r85.0<1>:d    r8.0<0;0>:d       r71.0<0;0>:d      320:w               //  ALU pipe: int; $304
(W)     add3 (1|M0)              r86.0<1>:d    r8.0<0;0>:d       r71.0<0;0>:d      384:w               //  ALU pipe: int; $307
(W)     add3 (1|M0)              r87.0<1>:d    r8.0<0;0>:d       r71.0<0;0>:d      448:w               //  ALU pipe: int; $310
        sync.nop                             null                             {Compacted,F@7}        // $230
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r14:1 bti[0][r13:1]      {$6} // ex_desc:0x0; desc:0x6218C500 // $230
        sync.nop                             null                             {Compacted,F@7}        // $233
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r15:1 bti[0][r23:1]      {$7} // ex_desc:0x0; desc:0x6218C500 // $233
        sync.nop                             null                             {Compacted,F@6}        // $236
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r16:1 bti[0][r25:1]      {$8} // ex_desc:0x0; desc:0x6218C500 // $236
        sync.nop                             null                             {Compacted,F@5}        // $239
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r17:1 bti[0][r27:1]      {$9} // ex_desc:0x0; desc:0x6218C500 // $239
        sync.nop                             null                             {Compacted,F@4}        // $242
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r18:1 bti[0][r29:1]      {$10} // ex_desc:0x0; desc:0x6218C500 // $242
        sync.nop                             null                             {Compacted,F@3}        // $245
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r19:1 bti[0][r31:1]      {$11} // ex_desc:0x0; desc:0x6218C500 // $245
        sync.nop                             null                             {Compacted,F@2}        // $248
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r20:1 bti[0][r33:1]      {$12} // ex_desc:0x0; desc:0x6218C500 // $248
        sync.nop                             null                             {Compacted,F@1}        // $251
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r21:1 bti[0][r35:1]      {$13} // ex_desc:0x0; desc:0x6218C500 // $251
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r38:1 bti[1][r37:1]      {$14} // ex_desc:0x1000000; desc:0x6218C500 // $255
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r39:1 bti[1][r46:1]      {$15} // ex_desc:0x1000000; desc:0x6218C500 // $258
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r40:1 bti[1][r47:1]      {$0} // ex_desc:0x1000000; desc:0x6218C500 // $261
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r41:1 bti[1][r48:1]      {$1} // ex_desc:0x1000000; desc:0x6218C500 // $264
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r42:1 bti[1][r49:1]      {$2} // ex_desc:0x1000000; desc:0x6218C500 // $267
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r43:1 bti[1][r50:1]      {A@7,$3} // ex_desc:0x1000000; desc:0x6218C500 // $270
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r44:1 bti[1][r51:1]      {A@7,$4} // ex_desc:0x1000000; desc:0x6218C500 // $273
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r45:1 bti[1][r52:1]      {A@7,$5} // ex_desc:0x1000000; desc:0x6218C500 // $276
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r73:1 bti[1][r72:1]      {A@7,$6} // ex_desc:0x1000000; desc:0x6218C500 // $291
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r74:1 bti[1][r81:1]      {A@7,$7} // ex_desc:0x1000000; desc:0x6218C500 // $294
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r75:1 bti[1][r82:1]      {A@6,$8} // ex_desc:0x1000000; desc:0x6218C500 // $297
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r76:1 bti[1][r83:1]      {A@5,$9} // ex_desc:0x1000000; desc:0x6218C500 // $300
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r77:1 bti[1][r84:1]      {A@4,$10} // ex_desc:0x1000000; desc:0x6218C500 // $303
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r78:1 bti[1][r85:1]      {A@3,$11} // ex_desc:0x1000000; desc:0x6218C500 // $306
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r79:1 bti[1][r86:1]      {A@2,$12} // ex_desc:0x1000000; desc:0x6218C500 // $309
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r80:1 bti[1][r87:1]      {A@1,$13} // ex_desc:0x1000000; desc:0x6218C500 // $312
(W)     mov (8|M0)               r53.0<1>:f    0x3F800000:f                               {Compacted} //  ALU pipe: float; $277
        add3 (8|M0)              r3.0<1>:d     r108.3<0;0>:d     r2.0<1;0>:d       512:w               //  ALU pipe: int; $323
        mov (8|M0)               r54.0<1>:f    r53.0<0;1,0>:f                   {Compacted,F@1}      //  ALU pipe: float; $278
        mov (8|M0)               r55.0<1>:f    r53.1<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $279
        mov (8|M0)               r56.0<1>:f    r53.2<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $280
        mov (8|M0)               r57.0<1>:f    r53.3<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $281
        mov (8|M0)               r58.0<1>:f    r53.4<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $282
        mov (8|M0)               r59.0<1>:f    r53.5<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $283
        mov (8|M0)               r60.0<1>:f    r53.6<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $284
        mov (8|M0)               r61.0<1>:f    r53.7<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $285
        mov (8|M0)               r88.0<1>:f    r53.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $314
        mov (8|M0)               r89.0<1>:f    r53.1<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $315
        mov (8|M0)               r90.0<1>:f    r53.2<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $316
        mov (8|M0)               r91.0<1>:f    r53.3<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $317
        mov (8|M0)               r92.0<1>:f    r53.4<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $318
        mov (8|M0)               r93.0<1>:f    r53.5<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $319
        mov (8|M0)               r94.0<1>:f    r53.6<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $320
        mov (8|M0)               r95.0<1>:f    r53.7<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $321
        add3 (8|M0)              r23.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       768:w               {$7.src} //  ALU pipe: int; $335
        add3 (8|M0)              r25.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       832:w               {$8.src} //  ALU pipe: int; $338
        add3 (8|M0)              r27.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       896:w               {$9.src} //  ALU pipe: int; $341
        add3 (8|M0)              r29.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       960:w               {$10.src} //  ALU pipe: int; $344
(W)     add3 (1|M0)              r31.0<1>:d    r107.3<0;0>:d     r7.0<0;0>:d       32:w               {$11.src} //  ALU pipe: int; $347
        mov (8|M0)               r14.0<1>:ud   r14.0<8;8,1>:ud                  {$6.dst}             //  ALU pipe: int; $286
        sync.nop                             null                             {Compacted,A@1}        // $286
        sync.allwr                           null                                                    // $286
        dpas.8x8 (8|M0)          r62:f         null:f            r62:bf            r62.0:bf         {Atomic} // $286
        dpas.8x8 (8|M0)          r62:f         r54:f             r14:bf            r38.0:bf         {Atomic} // $286 R{} IR{}{O:13,O:3,O:9,},  R{} IR{}{O:13,O:3,O:9,},  {BC=2}
        dpas.8x8 (8|M0)          r96:f         r88:f             r14:bf            r73.0:bf         {$15} // $322
        add3 (8|M0)              r17.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       576:w               {$15.src} //  ALU pipe: int; $326
        add3 (8|M0)              r19.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       640:w               //  ALU pipe: int; $329
        add3 (8|M0)              r21.0<1>:d    r108.3<0;0>:d     r2.0<1;0>:d       704:w               //  ALU pipe: int; $332
(W)     shr (1|M0)               r10.0<1>:d    r109.0<0;1,0>:ud  23:w               {Compacted}      //  ALU pipe: int; $226
(W)     shl (1|M0)               r11.0<1>:d    r108.1<0;1,0>:d   9:w               {Compacted}       //  ALU pipe: int; $227
(W)     add3 (1|M0)              r32.0<1>:d    r8.0<0;0>:d       r36.0<0;0>:d      32:w               //  ALU pipe: int; $348
(W)     add3 (1|M0)              r52.0<1>:d    r8.0<0;0>:d       r71.0<0;0>:d      32:w               //  ALU pipe: int; $373
(W)     mov (1|M0)               r4.0<1>:f     r3.0<0;1,0>:f                    {Compacted}          //  ALU pipe: float; $324
(W)     mov (1|M0)               r24.0<1>:f    r23.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $336
(W)     mov (1|M0)               r26.0<1>:f    r25.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $339
(W)     mov (1|M0)               r28.0<1>:f    r27.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $342
(W)     mov (1|M0)               r30.0<1>:f    r29.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $345
(W)     add3 (1|M0)              r45.0<1>:d    r31.0<0;0>:d      r36.0<0;0>:d      64:w               //  ALU pipe: int; $351
(W)     add3 (1|M0)              r46.0<1>:d    r31.0<0;0>:d      r36.0<0;0>:d      128:w               //  ALU pipe: int; $354
(W)     add3 (1|M0)              r47.0<1>:d    r31.0<0;0>:d      r36.0<0;0>:d      192:w               //  ALU pipe: int; $357
(W)     add3 (1|M0)              r48.0<1>:d    r31.0<0;0>:d      r36.0<0;0>:d      256:w               //  ALU pipe: int; $360
(W)     add3 (1|M0)              r49.0<1>:d    r31.0<0;0>:d      r36.0<0;0>:d      320:w               //  ALU pipe: int; $363
(W)     add3 (1|M0)              r50.0<1>:d    r31.0<0;0>:d      r36.0<0;0>:d      384:w               //  ALU pipe: int; $366
(W)     add3 (1|M0)              r51.0<1>:d    r31.0<0;0>:d      r36.0<0;0>:d      448:w               //  ALU pipe: int; $369
(W)     add3 (1|M0)              r61.0<1>:d    r31.0<0;0>:d      r71.0<0;0>:d      64:w               //  ALU pipe: int; $376
(W)     add3 (1|M0)              r70.0<1>:d    r31.0<0;0>:d      r71.0<0;0>:d      128:w               //  ALU pipe: int; $379
(W)     add3 (1|M0)              r72.0<1>:d    r31.0<0;0>:d      r71.0<0;0>:d      192:w               //  ALU pipe: int; $382
(W)     add3 (1|M0)              r73.0<1>:d    r31.0<0;0>:d      r71.0<0;0>:d      256:w               //  ALU pipe: int; $385
(W)     add3 (1|M0)              r74.0<1>:d    r31.0<0;0>:d      r71.0<0;0>:d      320:w               //  ALU pipe: int; $388
(W)     add3 (1|M0)              r75.0<1>:d    r31.0<0;0>:d      r71.0<0;0>:d      384:w               //  ALU pipe: int; $391
(W)     add3 (1|M0)              r76.0<1>:d    r31.0<0;0>:d      r71.0<0;0>:d      448:w               //  ALU pipe: int; $394
(W)     mov (1|M0)               r18.0<1>:f    r17.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $327
(W)     mov (1|M0)               r20.0<1>:f    r19.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $330
(W)     mov (1|M0)               r22.0<1>:f    r21.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $333
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r37:1 bti[1][r32:1]      {$14} // ex_desc:0x1000000; desc:0x6218C500 // $350
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r53:1 bti[1][r52:1]      {$0} // ex_desc:0x1000000; desc:0x6218C500 // $375
        sync.nop                             null                             {Compacted,F@7}        // $325
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r9:1  bti[0][r4:1]       {$1} // ex_desc:0x0; desc:0x6218C500 // $325
        sync.nop                             null                             {Compacted,F@7}        // $337
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r13:1 bti[0][r24:1]      {$2} // ex_desc:0x0; desc:0x6218C500 // $337
        sync.nop                             null                             {Compacted,F@6}        // $340
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r14:1 bti[0][r26:1]      {$3} // ex_desc:0x0; desc:0x6218C500 // $340
        sync.nop                             null                             {Compacted,F@5}        // $343
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r15:1 bti[0][r28:1]      {$4} // ex_desc:0x0; desc:0x6218C500 // $343
        sync.nop                             null                             {Compacted,F@4}        // $346
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r16:1 bti[0][r30:1]      {$5} // ex_desc:0x0; desc:0x6218C500 // $346
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r38:1 bti[1][r45:1]      {$6} // ex_desc:0x1000000; desc:0x6218C500 // $353
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r39:1 bti[1][r46:1]      {$7} // ex_desc:0x1000000; desc:0x6218C500 // $356
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r40:1 bti[1][r47:1]      {$8} // ex_desc:0x1000000; desc:0x6218C500 // $359
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r41:1 bti[1][r48:1]      {A@7,$9} // ex_desc:0x1000000; desc:0x6218C500 // $362
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r42:1 bti[1][r49:1]      {A@7,$10} // ex_desc:0x1000000; desc:0x6218C500 // $365
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r43:1 bti[1][r50:1]      {A@7,$11} // ex_desc:0x1000000; desc:0x6218C500 // $368
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r44:1 bti[1][r51:1]      {A@7,$12} // ex_desc:0x1000000; desc:0x6218C500 // $371
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r54:1 bti[1][r61:1]      {A@7,$13} // ex_desc:0x1000000; desc:0x6218C500 // $378
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r55:1 bti[1][r70:1]      {A@6,$15} // ex_desc:0x1000000; desc:0x6218C500 // $381
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r56:1 bti[1][r72:1]      {A@5,$14} // ex_desc:0x1000000; desc:0x6218C500 // $384
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r57:1 bti[1][r73:1]      {A@4,$0} // ex_desc:0x1000000; desc:0x6218C500 // $387
        sync.nop                             null                             {Compacted,I@3}        // $390
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r58:1 bti[1][r74:1]      {$1} // ex_desc:0x1000000; desc:0x6218C500 // $390
        sync.nop                             null                             {Compacted,I@2}        // $393
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r59:1 bti[1][r75:1]      {$2} // ex_desc:0x1000000; desc:0x6218C500 // $393
        sync.nop                             null                             {Compacted,I@1}        // $396
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r60:1 bti[1][r76:1]      {$3} // ex_desc:0x1000000; desc:0x6218C500 // $396
(W)     or (1|M0)                r12.0<1>:d    r11.0<0;1,0>:d    r10.0<0;1,0>:d   {Compacted}        //  ALU pipe: int; $228 R{} IR{}{O:2,O:2,},  {BC=1}
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r10:1 bti[0][r18:1]      {A@1,$4} // ex_desc:0x0; desc:0x6218C500 // $328
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r11:1 bti[0][r20:1]      {A@2,$5} // ex_desc:0x0; desc:0x6218C500 // $331
        sync.nop                             null                             {Compacted,F@1}        // $334
(W)     load.ugm.d32x8t.a32.ca.ca (1|M0)  r12:1 bti[0][r22:1]      {$6} // ex_desc:0x0; desc:0x6218C500 // $334
(W)     mov (1|M0)               r5.0<1>:ud    acc0.0<0;1,0>:ud                 {Compacted}          //  ALU pipe: int; $220
(W)     shl (1|M0)               r82.0<1>:d    r109.0<0;1,0>:d   8:w               {Compacted}       //  ALU pipe: int; $403
        shl (8|M0)               r86.0<1>:d    r112.0<1;1,0>:d   3:w               {Compacted}       //  ALU pipe: int; $407
(W)     addc (1|M0)              r77.0<1>:ud   r107.4<0;1,0>:ud  r106.4<0;1,0>:ud {AccWrEn}          //  ALU pipe: int; $398 R{} IR{}{O:10,O:10,},  {BC=1}
        sync.nop                             null                             {Compacted,$4.src}     // $443
(W)     add (1|M0)               r18.0<1>:d    r82.0<0;1,0>:d    128:w               {Compacted,I@3} //  ALU pipe: int; $443
        shl (8|M0)               r86.0<1>:d    r86.0<1;1,0>:d    2:w               {Compacted,I@3}   //  ALU pipe: int; $408
(W)     shl (1|M0)               r80.0<1>:d    -r77.0<0;1,0>:d   2:w               {Compacted,I@3}   //  ALU pipe: int; $401
(W)     shl (1|M0)               r88.0<1>:d    r82.0<0;1,0>:d    2:w               {Compacted}       //  ALU pipe: int; $410
(W)     shl (1|M0)               r19.0<1>:d    r18.0<0;1,0>:d    2:w               {Compacted,I@4}   //  ALU pipe: int; $444
(W)     add3 (1|M0)              r81.0<1>:d    r107.6<0;0>:d     r107.7<0;0>:d     r80.0<0>:d       {I@3} //  ALU pipe: int; $402
        sync.allwr                           null                                                    // $372
        dpas.8x8 (8|M0)          r62:f         null:f            r62:bf            r62.0:bf         {Atomic} // $372
        dpas.8x8 (8|M0)          r62:f         r62:f             r9:bf             r37.0:bf         {Atomic} // $372 R{} IR{}{O:15,E:2,E:9,},  R{} IR{}{O:15,O:2,O:9,},  {BC=1}
        dpas.8x8 (8|M0)          r96:f         r96:f             r9:bf             r53.0:bf         {$8} // $397 R{} IR{}{E:8,E:2,E:13,},  R{} IR{}{E:8,O:2,O:13,},  {BC=1}
        add3 (8|M0)              acc2.0<1>:d   r108.0<0;0>:d     r80.0<0;0>:d      r86.0<1>:d        //  ALU pipe: int; $409
(W)     add3 (1|M0)              r6.0<1>:ud    r5.0<0;0>:ud      r107.0<0;0>:ud    r106.3<0>:ud      //  ALU pipe: int; $221 R{} IR{}{E:1,O:10,O:10,},  {BC=1}
        add3 (8|M0)              r89.0<1>:d    r81.0<0;0>:d      r86.0<1;0>:d      r88.0<0>:d       {I@3} //  ALU pipe: int; $411
        add3 (8|M0)              r20.0<1>:d    r81.0<0;0>:d      r86.0<1;0>:d      r19.0<0>:d        //  ALU pipe: int; $445
        add3 (8|M0)              r92.0<1>:d    acc2.0<1;0>:d     r88.0<0;0>:d      64:w               //  ALU pipe: int; $415
        add3 (8|M0)              r95.0<1>:d    acc2.0<1;0>:d     r88.0<0;0>:d      128:w               //  ALU pipe: int; $419
        add3 (8|M0)              r3.0<1>:d     acc2.0<1;0>:d     r88.0<0;0>:d      192:w               //  ALU pipe: int; $423
        add3 (8|M0)              r6.0<1>:d     acc2.0<1;0>:d     r88.0<0;0>:d      256:w               //  ALU pipe: int; $427
        add3 (8|M0)              r9.0<1>:d     acc2.0<1;0>:d     r88.0<0;0>:d      320:w               {$8.src} //  ALU pipe: int; $431
        add3 (8|M0)              r12.0<1>:d    acc2.0<1;0>:d     r88.0<0;0>:d      384:w               //  ALU pipe: int; $435
        add3 (8|M0)              r15.0<1>:d    acc2.0<1;0>:d     r88.0<0;0>:d      448:w               //  ALU pipe: int; $439
        add3 (8|M0)              r23.0<1>:d    acc2.0<1;0>:d     r19.0<0;0>:d      64:w               //  ALU pipe: int; $449
        add3 (8|M0)              r26.0<1>:d    acc2.0<1;0>:d     r19.0<0;0>:d      128:w               //  ALU pipe: int; $453
        add3 (8|M0)              r29.0<1>:d    acc2.0<1;0>:d     r19.0<0;0>:d      192:w               //  ALU pipe: int; $457
        add3 (8|M0)              r32.0<1>:d    acc2.0<1;0>:d     r19.0<0;0>:d      256:w               //  ALU pipe: int; $461
        add3 (8|M0)              r35.0<1>:d    acc2.0<1;0>:d     r19.0<0;0>:d      320:w               //  ALU pipe: int; $465
        add3 (8|M0)              r38.0<1>:d    acc2.0<1;0>:d     r19.0<0;0>:d      384:w               //  ALU pipe: int; $469
        add3 (8|M0)              r41.0<1>:d    acc2.0<1;0>:d     r19.0<0;0>:d      448:w               //  ALU pipe: int; $473
(W)     mov (8|M0)               r127.0<1>:f   r105.0<1;1,0>:f                  {Compacted}          //  ALU pipe: float; $477
(W)     shr (1|M0)               r83.0<1>:d    r109.0<0;1,0>:ud  24:w               {Compacted}      //  ALU pipe: int; $404
(W)     shl (1|M0)               r84.0<1>:d    r108.1<0;1,0>:d   8:w               {Compacted}       //  ALU pipe: int; $405
(W)     mov (1|M0)               r78.0<1>:ud   acc0.0<0;1,0>:ud                 {Compacted}          //  ALU pipe: int; $398
(W)     mov (1|M0)               r91.0<1>:f    r89.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $413
(W)     mov (1|M0)               r22.0<1>:f    r20.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $447
        mov (8|M0)               r90.0<1>:f    r62.0<1;1,0>:f                   {Compacted,$8.dst}   //  ALU pipe: float; $412
        mov (8|M0)               r93.0<1>:f    r63.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $416
(W)     mov (1|M0)               r94.0<1>:f    r92.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $417
        mov (8|M0)               r104.0<1>:f   r64.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $420
        mov (8|M0)               r4.0<1>:f     r65.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $424
        mov (8|M0)               r7.0<1>:f     r66.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $428
        mov (8|M0)               r10.0<1>:f    r67.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $432
        mov (8|M0)               r13.0<1>:f    r68.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $436
        mov (8|M0)               r16.0<1>:f    r69.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $440
(W)     mov (1|M0)               r2.0<1>:f     r95.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $421
        mov (8|M0)               r21.0<1>:f    r96.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $446
        mov (8|M0)               r24.0<1>:f    r97.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $450
        mov (8|M0)               r27.0<1>:f    r98.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $454
        mov (8|M0)               r30.0<1>:f    r99.0<1;1,0>:f                   {Compacted}          //  ALU pipe: float; $458
        mov (8|M0)               r33.0<1>:f    r100.0<1;1,0>:f                  {Compacted}          //  ALU pipe: float; $462
        mov (8|M0)               r36.0<1>:f    r101.0<1;1,0>:f                  {Compacted}          //  ALU pipe: float; $466
        mov (8|M0)               r39.0<1>:f    r102.0<1;1,0>:f                  {Compacted}          //  ALU pipe: float; $470
        mov (8|M0)               r42.0<1>:f    r103.0<1;1,0>:f                  {Compacted}          //  ALU pipe: float; $474
(W)     mov (1|M0)               r5.0<1>:f     r3.0<0;1,0>:f                    {Compacted}          //  ALU pipe: float; $425
(W)     mov (1|M0)               r8.0<1>:f     r6.0<0;1,0>:f                    {Compacted}          //  ALU pipe: float; $429
(W)     mov (1|M0)               r11.0<1>:f    r9.0<0;1,0>:f                    {Compacted}          //  ALU pipe: float; $433
(W)     mov (1|M0)               r14.0<1>:f    r12.0<0;1,0>:f                   {Compacted}          //  ALU pipe: float; $437
(W)     mov (1|M0)               r17.0<1>:f    r15.0<0;1,0>:f                   {Compacted,I@7}      //  ALU pipe: float; $441
(W)     mov (1|M0)               r25.0<1>:f    r23.0<0;1,0>:f                   {Compacted,I@7}      //  ALU pipe: float; $451
(W)     mov (1|M0)               r28.0<1>:f    r26.0<0;1,0>:f                   {Compacted,I@7}      //  ALU pipe: float; $455
(W)     mov (1|M0)               r31.0<1>:f    r29.0<0;1,0>:f                   {Compacted,I@7}      //  ALU pipe: float; $459
(W)     mov (1|M0)               r34.0<1>:f    r32.0<0;1,0>:f                   {Compacted,I@7}      //  ALU pipe: float; $463
(W)     mov (1|M0)               r37.0<1>:f    r35.0<0;1,0>:f                   {Compacted,I@6}      //  ALU pipe: float; $467
(W)     mov (1|M0)               r40.0<1>:f    r38.0<0;1,0>:f                   {Compacted,I@5}      //  ALU pipe: float; $471
(W)     mov (1|M0)               r43.0<1>:f    r41.0<0;1,0>:f                   {Compacted,I@4}      //  ALU pipe: float; $475
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r91:1] r90:1     {$7} // ex_desc:0x2000000; desc:0x620EB704 // $414
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r94:1] r93:1     {$8} // ex_desc:0x2000000; desc:0x620EB704 // $418
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r2:1] r104:1     {$9} // ex_desc:0x2000000; desc:0x620EB704 // $422
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r5:1] r4:1       {$10} // ex_desc:0x2000000; desc:0x620EB704 // $426
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r8:1] r7:1       {A@7,$11} // ex_desc:0x2000000; desc:0x620EB704 // $430
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r11:1] r10:1     {A@7,$12} // ex_desc:0x2000000; desc:0x620EB704 // $434
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r14:1] r13:1     {A@7,$13} // ex_desc:0x2000000; desc:0x620EB704 // $438
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r17:1] r16:1     {A@7,$14} // ex_desc:0x2000000; desc:0x620EB704 // $442
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r22:1] r21:1     {$15} // ex_desc:0x2000000; desc:0x620EB704 // $448
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r25:1] r24:1     {A@7,$0} // ex_desc:0x2000000; desc:0x620EB704 // $452
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r28:1] r27:1     {A@6,$1} // ex_desc:0x2000000; desc:0x620EB704 // $456
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r31:1] r30:1     {A@5,$2} // ex_desc:0x2000000; desc:0x620EB704 // $460
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r34:1] r33:1     {A@4,$3} // ex_desc:0x2000000; desc:0x620EB704 // $464
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r37:1] r36:1     {A@3,$4} // ex_desc:0x2000000; desc:0x620EB704 // $468
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r40:1] r39:1     {A@2,$5} // ex_desc:0x2000000; desc:0x620EB704 // $472
(W)     store.ugm.d64x4t.a32.wb.wb (1|M0)  bti[2][r43:1] r42:1     {A@1,$6} // ex_desc:0x2000000; desc:0x620EB704 // $476
(W)     or (1|M0)                r85.0<1>:d    r84.0<0;1,0>:d    r83.0<0;1,0>:d   {Compacted,I@2}    //  ALU pipe: int; $406
(W)     add3 (1|M0)              r79.0<1>:ud   r78.0<0;0>:ud     r107.5<0;0>:ud    r106.5<0>:ud     {I@2} //  ALU pipe: int; $399 R{} IR{}{O:3,O:10,O:10,},  {BC=1}
(W)     send.gtwy (1|M0)         null     r127    null:0  0x0            0x02000010           {EOT,A@1} // wr:1+0, rd:0; end of thread // $477
L6200:
        nop                                                                                          // $477


//.BankConflicts: 18
//.ByteRMWs: 0
//


//.numALUInst: 408
//.accSubDef: 37
//.accSubUse: 60
//.accSubCandidateDef: 40
//.accSubCandidateUse: 63
//
//
//.singlePipeAtOneDistNum: 35
//.allAtOneDistNum: 12
//.syncInstCount: 23
//.tokenReuseCount: 17
//.AfterWriteTokenDepCount: 56
//.AfterReadTokenDepCount: 11
