# ABS Census 2021 GCP - All G-Table Metadata

*Source Metadata File:* `Metadata_2021_GCP_DataPack_R1_R2.xlsx`
*Source Template File:* `2021_GCP_Sequential_Template_R2.xlsx`

This document summarizes the structure of all G-tables found in the metadata template file. Use this as a reference when implementing processing logic for specific tables.

## G01: G01: G01 SELECTED PERSON CHARACTERISTICS BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 15: Males, Females, Persons
  - Row 17: Total persons, 1, 2, 3
  - Row 20: 0-4 years, 4, 5, 6

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: 5-14 years | 7 | 8
  - Row 22: 15-19 years | 10 | 11
  - Row 23: 20-24 years | 13 | 14
  - Row 24: 25-34 years | 16 | 17
  - Row 25: 35-44 years | 19 | 20
  - Row 26: 45-54 years | 22 | 23
  - Row 27: 55-64 years | 25 | 26
  - Row 28: 65-74 years | 28 | 29
  - Row 29: 75-84 years | 31 | 32
  - Row 30: 85 years and over | 34 | 35
  - Row 32: Counted on Census Night:
  - Row 33: At home | 37 | 38
  - Row 34: Elsewhere in Australia | 40 | 41
  - Row 36: Aboriginal and/or Torres Strait Islander persons:
  - Row 37: Aboriginal | 43 | 44
  - Row 38: Torres Strait Islander | 46 | 47
  - Row 39: Both Aboriginal and Torres Strait Islander(a) | 49 | 50
  - Row 40: Total | 52 | 53
  - Row 42: Birthplace:
  - Row 43: Australia(b) | 55 | 56
  - Row 44: Elsewhere(c) | 58 | 59
  - Row 46: Language used at home:
  - Row 47: English only | 61 | 62
  - Row 48: Other language(d) | 64 | 65
  - Row 50: Australian citizen | 67 | 68

**Notes (from bottom of template sheet):**
  - Language used at home:

---

## G02: G02: G02 SELECTED MEDIANS AND AVERAGES

**Potential Headers:** (None detected in typical range)

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 2: Australia (AUS) 7688094.9 sq kms
  - Row 4: G02 SELECTED MEDIANS AND AVERAGES
  - Row 13: Median age of persons | 109.0
  - Row 15: Median total personal income ($/weekly) | 111.0
  - Row 17: Median total family income ($/weekly) | 113.0
  - Row 19: Median total household income ($/weekly) | 115.0
  - Row 22: Median age of persons excludes overseas visitors.
  - Row 24: Median total personal income is applicable to persons aged 15 years and over.
  - Row 26: Median total family income is applicable to families in family households. It excludes families where at least one member aged 15 years and over did not state an income and families where at
  - Row 27: least one member aged 15 years and over was temporarily absent on Census Night.
  - Row 29: Median total household income is applicable to occupied private dwellings. It excludes households where at least one member aged 15 years and over did not state an income and
  - Row 30: households where at least one member aged 15 years and over was temporarily absent on Census Night. It excludes 'Visitors only' and 'Other non-classifiable' households.

**Notes (from bottom of template sheet):**
  - and 'Other non-classifiable' households.
  - Median rent is applicable to occupied private dwellings being rented. It excludes 'Visitors only' and 'Other non-classifiable' households.
  - (a) For 2021, median rent calculations exclude dwellings being 'Occupied rent-free' and will not be comparable to 2016 Census data.
  - Average number of persons per bedroom is applicable to occupied private dwellings. It excludes 'Visitors only' and 'Other non-classifiable' households.
  - Average household size is applicable to number of persons usually resident in occupied private dwellings. It includes partners, children, and co-tenants (in group households) who were
  - temporarily absent on Census Night. A maximum of three temporary absentees can be counted in each household. It excludes 'Visitors only' and 'Other non-classifiable' households.

---

## G03: G03: G03 PLACE OF USUAL RESIDENCE, BY PLACE OF ENUMERATION ON CENSUS NIGHT(a), BY AGE

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 0-14, 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 10: years, years, years, years, years, years, years, years, and over, Total
  - Row 12: Counted at home on Census Night, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126
  - Row 15: Same Statistical Area Level 2 (SA2), 127, 128, 129, 130, 131, 132, 133, 134, 135, 136
  - Row 17: New South Wales, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146
  - Row 18: Victoria, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156
  - Row 19: Queensland, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166
  - Row 20: South Australia, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Western Australia | 177 | 178
  - Row 22: Tasmania | 187 | 188
  - Row 23: Northern Territory | 197 | 198
  - Row 24: Australian Capital Territory | 207 | 208
  - Row 25: Other Territories | 217 | 218
  - Row 26: Total visitors from a different SA2 | 227 | 228
  - Row 27: Total visitors | 237 | 238
  - Row 29: Total | 247 | 248
  - Row 31: This table is based on place of enumeration.
  - Row 32: (a) This table counts persons where they were staying on Census Night. It shows the number of persons who were at home on Census Night and the number of persons who were visiting away from their usual residence.
  - Row 33: (b) For persons visiting away from their usual residence, it shows which SA2 they were visiting from, i.e. either interstate, intrastate or from within the same SA2 area.
  - Row 35: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of enumeration.
  - (a) This table counts persons where they were staying on Census Night. It shows the number of persons who were at home on Census Night and the number of persons who were visiting away from their usual residence.
  - (b) For persons visiting away from their usual residence, it shows which SA2 they were visiting from, i.e. either interstate, intrastate or from within the same SA2 area.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G04: G04: G04 AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 7: Males, Females, Persons, Males, Females, Persons, Males, Females, Persons
  - Row 10: 0, 257, 258, 259, 30, 365, 366, 367, 60, 473, 474, 475
  - Row 11: 1, 260, 261, 262, 31, 368, 369, 370, 61, 476, 477, 478
  - Row 12: 2, 263, 264, 265, 32, 371, 372, 373, 62, 479, 480, 481
  - Row 13: 3, 266, 267, 268, 33, 374, 375, 376, 63, 482, 483, 484
  - Row 14: 4, 269, 270, 271, 34, 377, 378, 379, 64, 485, 486, 487
  - Row 15: 0-4 years, 272, 273, 274, 30-34 years, 380, 381, 382, 60-64 years, 488, 489, 490
  - Row 16: 5, 275, 276, 277, 35, 383, 384, 385, 65, 491, 492, 493
  - Row 17: 6, 278, 279, 280, 36, 386, 387, 388, 66, 494, 495, 496
  - Row 18: 7, 281, 282, 283, 37, 389, 390, 391, 67, 497, 498, 499
  - Row 19: 8, 284, 285, 286, 38, 392, 393, 394, 68, 500, 501, 502
  - Row 20: 9, 287, 288, 289, 39, 395, 396, 397, 69, 503, 504, 505

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: 5-9 years | 290 | 291
  - Row 27: 10-14 years | 308 | 309
  - Row 33: 15-19 years | 326 | 327
  - Row 39: 20-24 years | 344 | 345
  - Row 45: 25-29 years | 362 | 363
  - Row 47: This table is based on place of usual residence.
  - Row 49: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of usual residence.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G05: G05: G05 REGISTERED MARITAL STATUS(a) BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Married(b), Separated, Divorced, Widowed, Never married, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 9: MALES
  - Row 11: 15-19 years | 563 | 564
  - Row 12: 20-24 years | 569 | 570
  - Row 13: 25-34 years | 575 | 576
  - Row 14: 35-44 years | 581 | 582
  - Row 15: 45-54 years | 587 | 588
  - Row 16: 55-64 years | 593 | 594
  - Row 17: 65-74 years | 599 | 600
  - Row 18: 75-84 years | 605 | 606
  - Row 19: 85 years and over | 611 | 612
  - Row 21: Total | 617 | 618
  - Row 23: FEMALES
  - Row 25: 15-19 years | 623 | 624
  - Row 26: 20-24 years | 629 | 630
  - Row 27: 25-34 years | 635 | 636
  - Row 28: 35-44 years | 641 | 642
  - Row 29: 45-54 years | 647 | 648
  - Row 30: 55-64 years | 653 | 654
  - Row 31: 65-74 years | 659 | 660
  - Row 32: 75-84 years | 665 | 666
  - Row 33: 85 years and over | 671 | 672
  - Row 35: Total | 677 | 678
  - Row 37: PERSONS

**Notes:** (None detected)

---

## G06: G06: G06 SOCIAL MARITAL STATUS BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Married in a, Married in a, Not
  - Row 9: registered marriage(b), de facto marriage(c), married, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 10: MALES
  - Row 12: 15-19 years | 743 | 744
  - Row 13: 20-24 years | 747 | 748
  - Row 14: 25-34 years | 751 | 752
  - Row 15: 35-44 years | 755 | 756
  - Row 16: 45-54 years | 759 | 760
  - Row 17: 55-64 years | 763 | 764
  - Row 18: 65-74 years | 767 | 768
  - Row 19: 75-84 years | 771 | 772
  - Row 20: 85 years and over | 775 | 776
  - Row 22: Total | 779 | 780
  - Row 24: FEMALES
  - Row 26: 15-19 years | 783 | 784
  - Row 27: 20-24 years | 787 | 788
  - Row 28: 25-34 years | 791 | 792
  - Row 29: 35-44 years | 795 | 796
  - Row 30: 45-54 years | 799 | 800
  - Row 31: 55-64 years | 803 | 804
  - Row 32: 65-74 years | 807 | 808
  - Row 33: 75-84 years | 811 | 812
  - Row 34: 85 years and over | 815 | 816
  - Row 36: Total | 819 | 820
  - Row 38: PERSONS

**Notes:** (None detected)

---

## G07: G07: G07 INDIGENOUS STATUS BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: Aboriginal and/or Torres Strait Islander(a), Non-Indigenous, Indigenous status not stated, Total
  - Row 10: Males, Females, Persons, Males, Females, Persons, Males, Females, Persons, Males, Females, Persons
  - Row 12: 0-4 years, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874
  - Row 13: 5-9 years, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886
  - Row 14: 10-14 years, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898
  - Row 15: 15-19 years, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910
  - Row 16: 20-24 years, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922
  - Row 17: 25-29 years, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934
  - Row 18: 30-34 years, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946
  - Row 19: 35-39 years, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958
  - Row 20: 40-44 years, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: 45-49 years | 971 | 972
  - Row 22: 50-54 years | 983 | 984
  - Row 23: 55-59 years | 995 | 996
  - Row 24: 60-64 years | 1007 | 1008
  - Row 25: 65 years and over | 1019 | 1020
  - Row 27: Total | 1031 | 1032
  - Row 29: This table is based on place of usual residence.
  - Row 30: (a) Comprises persons who identified themselves as being of 'Aboriginal' or 'Torres Strait Islander' or 'Both Aboriginal and Torres Strait Islander' origin.
  - Row 32: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of usual residence.
  - (a) Comprises persons who identified themselves as being of 'Aboriginal' or 'Torres Strait Islander' or 'Both Aboriginal and Torres Strait Islander' origin.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G08: G08: G08 ANCESTRY(a) BY COUNTRY OF BIRTH OF PARENTS

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Both parents, Father only, Mother only, Both parents, Birthplace, Total
  - Row 9: born overseas, born overseas, born overseas, born in Australia, not stated(b), responses(c)
  - Row 11: Australian, 1043, 1044, 1045, 1046, 1047, 1048
  - Row 12: Australian Aboriginal, 1049, 1050, 1051, 1052, 1053, 1054
  - Row 13: Chinese, 1055, 1056, 1057, 1058, 1059, 1060
  - Row 14: Croatian, 1061, 1062, 1063, 1064, 1065, 1066
  - Row 15: Dutch, 1067, 1068, 1069, 1070, 1071, 1072
  - Row 16: English, 1073, 1074, 1075, 1076, 1077, 1078
  - Row 17: Filipino, 1079, 1080, 1081, 1082, 1083, 1084
  - Row 18: French, 1085, 1086, 1087, 1088, 1089, 1090
  - Row 19: German, 1091, 1092, 1093, 1094, 1095, 1096
  - Row 20: Greek, 1097, 1098, 1099, 1100, 1101, 1102

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Hungarian | 1103 | 1104
  - Row 22: Indian | 1109 | 1110
  - Row 23: Irish | 1115 | 1116
  - Row 24: Italian | 1121 | 1122
  - Row 25: Korean | 1127 | 1128
  - Row 26: Lebanese | 1133 | 1134
  - Row 27: Macedonian | 1139 | 1140
  - Row 28: Maltese | 1145 | 1146
  - Row 29: Maori | 1151 | 1152
  - Row 30: New Zealander | 1157 | 1158
  - Row 31: Polish | 1163 | 1164
  - Row 32: Russian | 1169 | 1170
  - Row 33: Samoan | 1175 | 1176
  - Row 34: Scottish | 1181 | 1182
  - Row 35: Serbian | 1187 | 1188
  - Row 36: South African | 1193 | 1194
  - Row 37: Spanish | 1199 | 1200
  - Row 38: Sri Lankan | 1205 | 1206
  - Row 39: Vietnamese | 1211 | 1212
  - Row 40: Welsh | 1217 | 1218
  - Row 41: Other(d) | 1223 | 1224
  - Row 42: Ancestry not stated | 1229 | 1230
  - Row 44: Total persons(c) | 1235 | 1236
  - Row 46: This table is based on place of usual residence.
  - Row 47: (a) This list of ancestries consists of the most common 30 Ancestry responses reported in the 2016 Census.
  - Row 48: (b) Includes birthplace for either or both parents not stated.
  - Row 49: (c) This table is a multi-response table and therefore the total responses count will not equal the total persons count.
  - Row 50: (d) If two responses from one person are categorised in the 'Other' category only one response is counted. Includes ancestries not identified individually and 'Inadequately described'.

**Notes (from bottom of template sheet):**
  - This table is based on place of usual residence.
  - (a) This list of ancestries consists of the most common 30 Ancestry responses reported in the 2016 Census.
  - (b) Includes birthplace for either or both parents not stated.
  - (c) This table is a multi-response table and therefore the total responses count will not equal the total persons count.
  - (d) If two responses from one person are categorised in the 'Other' category only one response is counted. Includes ancestries not identified individually and 'Inadequately described'.

---

## G09: Country of Birth of Person by Age by Sex

**Error processing this table:** Sheet not found in template.

---

## G10: G10: G10 COUNTRY OF BIRTH OF PERSON(a) BY YEAR OF ARRIVAL IN AUSTRALIA

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Before, 1951.0, 1961.0, 1971.0, 1981.0, 1991.0, 2001.0, 2011.0
  - Row 9: 1951, -1960.0, -1970.0, -1980.0, -1990.0, -2000.0, -2010.0, -2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0, 2021(c), Not stated, Total
  - Row 11: Afghanistan, 2831, 2832.0, 2833.0, 2834.0, 2835.0, 2836.0, 2837.0, 2838.0, 2839.0, 2840.0, 2841.0, 2842.0, 2843.0, 2844, 2845, 2846
  - Row 12: Bangladesh, 2847, 2848.0, 2849.0, 2850.0, 2851.0, 2852.0, 2853.0, 2854.0, 2855.0, 2856.0, 2857.0, 2858.0, 2859.0, 2860, 2861, 2862
  - Row 13: Canada, 2863, 2864.0, 2865.0, 2866.0, 2867.0, 2868.0, 2869.0, 2870.0, 2871.0, 2872.0, 2873.0, 2874.0, 2875.0, 2876, 2877, 2878
  - Row 14: China (excludes SARs and Taiwan)(d), 2879, 2880.0, 2881.0, 2882.0, 2883.0, 2884.0, 2885.0, 2886.0, 2887.0, 2888.0, 2889.0, 2890.0, 2891.0, 2892, 2893, 2894
  - Row 15: Croatia, 2895, 2896.0, 2897.0, 2898.0, 2899.0, 2900.0, 2901.0, 2902.0, 2903.0, 2904.0, 2905.0, 2906.0, 2907.0, 2908, 2909, 2910
  - Row 16: Egypt, 2911, 2912.0, 2913.0, 2914.0, 2915.0, 2916.0, 2917.0, 2918.0, 2919.0, 2920.0, 2921.0, 2922.0, 2923.0, 2924, 2925, 2926
  - Row 17: Fiji, 2927, 2928.0, 2929.0, 2930.0, 2931.0, 2932.0, 2933.0, 2934.0, 2935.0, 2936.0, 2937.0, 2938.0, 2939.0, 2940, 2941, 2942
  - Row 18: Germany, 2943, 2944.0, 2945.0, 2946.0, 2947.0, 2948.0, 2949.0, 2950.0, 2951.0, 2952.0, 2953.0, 2954.0, 2955.0, 2956, 2957, 2958
  - Row 19: Greece, 2959, 2960.0, 2961.0, 2962.0, 2963.0, 2964.0, 2965.0, 2966.0, 2967.0, 2968.0, 2969.0, 2970.0, 2971.0, 2972, 2973, 2974
  - Row 20: Hong Kong (SAR of China)(d), 2975, 2976.0, 2977.0, 2978.0, 2979.0, 2980.0, 2981.0, 2982.0, 2983.0, 2984.0, 2985.0, 2986.0, 2987.0, 2988, 2989, 2990

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: India | 2991 | 2992.0
  - Row 22: Indonesia | 3007 | 3008.0
  - Row 23: Iran | 3023 | 3024.0
  - Row 24: Iraq | 3039 | 3040.0
  - Row 25: Ireland | 3055 | 3056.0
  - Row 26: Italy | 3071 | 3072.0
  - Row 27: Japan | 3087 | 3088.0
  - Row 28: Korea, Republic of (South) | 3103 | 3104.0
  - Row 29: Lebanon | 3119 | 3120.0
  - Row 30: Malaysia | 3135 | 3136.0
  - Row 31: Nepal | 3151 | 3152.0
  - Row 32: Netherlands | 3167 | 3168.0
  - Row 33: New Zealand | 3183 | 3184.0
  - Row 34: North Macedonia | 3199 | 3200.0
  - Row 35: Pakistan | 3215 | 3216.0
  - Row 36: Philippines | 3231 | 3232.0
  - Row 37: Poland | 3247 | 3248.0
  - Row 38: Singapore | 3263 | 3264.0
  - Row 39: South Africa | 3279 | 3280.0
  - Row 40: Sri Lanka | 3295 | 3296.0
  - Row 41: Taiwan | 3311 | 3312.0
  - Row 42: Thailand | 3327 | 3328.0
  - Row 43: United Kingdom, Channel Islands and Isle of Man(e) | 3343 | 3344.0
  - Row 44: United States of America | 3359 | 3360.0
  - Row 45: Vietnam | 3375 | 3376.0
  - Row 46: Born elsewhere(f) | 3391 | 3392.0
  - Row 48: Total | 3407 | 3408.0
  - Row 50: This table is based on place of usual residence.

**Notes (from bottom of template sheet):**
  - United Kingdom, Channel Islands and Isle of Man(e)
  - United States of America
  - This table is based on place of usual residence.

---

## G11: Proficiency in Spoken English by Year of Arrival in Australia by Age

**Error processing this table:** Sheet not found in template.

---

## G12: Proficiency in Spoken English of Parents by Age of Dependent Children

**Error processing this table:** Sheet not found in template.

---

## G13: Language Used at Home by Proficiency in Spoken English by Sex

**Error processing this table:** Sheet not found in template.

---

## G14: G14: G14 RELIGIOUS AFFILIATION BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 7: Males, Females, Persons
  - Row 9: Buddhism, 5431, 5432, 5433
  - Row 11: Anglican, 5434, 5435, 5436
  - Row 12: Assyrian Apostolic, 5437, 5438, 5439
  - Row 13: Baptist, 5440, 5441, 5442
  - Row 14: Brethren, 5443, 5444, 5445
  - Row 15: Catholic, 5446, 5447, 5448
  - Row 16: Churches of Christ, 5449, 5450, 5451
  - Row 17: Eastern Orthodox, 5452, 5453, 5454
  - Row 18: Jehovah's Witnesses, 5455, 5456, 5457
  - Row 19: Latter-day Saints, 5458, 5459, 5460
  - Row 20: Lutheran, 5461, 5462, 5463

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Oriental Orthodox | 5464 | 5465
  - Row 22: Other Protestant | 5467 | 5468
  - Row 23: Pentecostal | 5470 | 5471
  - Row 24: Presbyterian and Reformed | 5473 | 5474
  - Row 25: Salvation Army | 5476 | 5477
  - Row 26: Seventh-day Adventist | 5479 | 5480
  - Row 27: Uniting Church | 5482 | 5483
  - Row 28: Christianity, nfd | 5485 | 5486
  - Row 29: Other Christian | 5488 | 5489
  - Row 30: Total | 5491 | 5492
  - Row 31: Hinduism | 5494 | 5495
  - Row 32: Islam | 5497 | 5498
  - Row 33: Judaism | 5500 | 5501
  - Row 34: Other Religions:
  - Row 35: Australian Aboriginal Traditional Religions | 5503 | 5504
  - Row 36: Sikhism | 5506 | 5507
  - Row 37: Other Religious Groups(a) | 5509 | 5510
  - Row 38: Total | 5512 | 5513
  - Row 39: Secular Beliefs and Other Spiritual Beliefs and No Religious Affiliation:
  - Row 40: No Religion, so described | 5515 | 5516
  - Row 41: Secular Beliefs(b) | 5518 | 5519
  - Row 42: Other Spiritual Beliefs(c) | 5521 | 5522
  - Row 43: Total | 5524 | 5525
  - Row 44: Religious affiliation not stated(d) | 5527 | 5528
  - Row 46: Total | 5530 | 5531
  - Row 48: This table is based on place of usual residence.
  - Row 49: (a) Comprises 'Baha'i', 'Chinese Religions', 'Druse', 'Japanese Religions', 'Nature Religions', 'Spiritualism' and 'Miscellaneous Religions'.
  - Row 50: (b) 'Secular Beliefs' includes 'Secular Beliefs, nfd', 'Agnosticism', 'Atheism', 'Humanism', 'Rationalism' and 'Secular Beliefs, nec'.

**Notes (from bottom of template sheet):**
  - Other Spiritual Beliefs(c)
  - Religious affiliation not stated(d)
  - This table is based on place of usual residence.
  - (a) Comprises 'Baha'i', 'Chinese Religions', 'Druse', 'Japanese Religions', 'Nature Religions', 'Spiritualism' and 'Miscellaneous Religions'.
  - (b) 'Secular Beliefs' includes 'Secular Beliefs, nfd', 'Agnosticism', 'Atheism', 'Humanism', 'Rationalism' and 'Secular Beliefs, nec'.

---

## G15: G15: G15 TYPE OF EDUCATION INSTITUTION ATTENDING (FULL-TIME/PART-TIME STUDENT STATUS BY AGE) BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Males, Females, Persons
  - Row 10: Preschool, 5533, 5534, 5535
  - Row 13: Government, 5536, 5537, 5538
  - Row 14: Catholic, 5539, 5540, 5541
  - Row 15: Other non-Government, 5542, 5543, 5544
  - Row 16: Total Primary(a), 5545, 5546, 5547
  - Row 19: Government, 5548, 5549, 5550
  - Row 20: Catholic, 5551, 5552, 5553

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Other non-Government | 5554 | 5555
  - Row 22: Total Secondary(b) | 5557 | 5558
  - Row 24: Tertiary:
  - Row 25: Vocational education (including TAFE and private training providers):
  - Row 26: Full-time student:
  - Row 27: Aged 15-24 years | 5560 | 5561
  - Row 28: Aged 25 years and over | 5563 | 5564
  - Row 29: Part-time student:
  - Row 30: Aged 15-24 years | 5566 | 5567
  - Row 31: Aged 25 years and over | 5569 | 5570
  - Row 32: Full-time/part-time student status not stated | 5572 | 5573
  - Row 33: Total Vocational education (including TAFE and private training providers) | 5575 | 5576
  - Row 35: University or other higher education:
  - Row 36: Full-time student:
  - Row 37: Aged 15-24 years | 5578 | 5579
  - Row 38: Aged 25 years and over | 5581 | 5582
  - Row 39: Part-time student:
  - Row 40: Aged 15-24 years | 5584 | 5585
  - Row 41: Aged 25 years and over | 5587 | 5588
  - Row 42: Full-time/part-time student status not stated | 5590 | 5591
  - Row 43: Total University or higher education | 5593 | 5594
  - Row 45: Total Tertiary(c) | 5596 | 5597
  - Row 47: Other type of education institution:
  - Row 48: Full-time student | 5599 | 5600
  - Row 49: Part-time student | 5602 | 5603
  - Row 50: Full-time/part-time student status not stated | 5605 | 5606

**Notes (from bottom of template sheet):**
  - Full-time/part-time student status not stated
  - Total University or higher education
  - Other type of education institution:
  - Full-time/part-time student status not stated

---

## G16: G16: G16 HIGHEST YEAR OF SCHOOL COMPLETED BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 15-19, 20-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 10: years, years, years, years, years, years, years, years, and over, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Year 12 or equivalent | 5617 | 5618
  - Row 14: Year 11 or equivalent | 5627 | 5628
  - Row 15: Year 10 or equivalent | 5637 | 5638
  - Row 16: Year 9 or equivalent | 5647 | 5648
  - Row 17: Year 8 or below | 5657 | 5658
  - Row 19: Did not go to school | 5667 | 5668
  - Row 21: Highest year of school not stated | 5677 | 5678
  - Row 23: Total | 5687 | 5688
  - Row 25: FEMALES
  - Row 27: Year 12 or equivalent | 5697 | 5698
  - Row 28: Year 11 or equivalent | 5707 | 5708
  - Row 29: Year 10 or equivalent | 5717 | 5718
  - Row 30: Year 9 or equivalent | 5727 | 5728
  - Row 31: Year 8 or below | 5737 | 5738
  - Row 33: Did not go to school | 5747 | 5748
  - Row 35: Highest year of school not stated | 5757 | 5758
  - Row 37: Total | 5767 | 5768
  - Row 39: PERSONS

**Notes (from bottom of template sheet):**
  - Year 11 or equivalent
  - Year 10 or equivalent
  - Highest year of school not stated

---

## G17: G17: G17 TOTAL PERSONAL INCOME (WEEKLY) BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 15-19, 20-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 10: years, years, years, years, years, years, years, years, and over, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Negative/Nil income | 5857 | 5858
  - Row 14: $1-$149 | 5867 | 5868
  - Row 15: $150-$299 | 5877 | 5878
  - Row 16: $300-$399 | 5887 | 5888
  - Row 17: $400-$499 | 5897 | 5898
  - Row 18: $500-$649 | 5907 | 5908
  - Row 19: $650-$799 | 5917 | 5918
  - Row 20: $800-$999 | 5927 | 5928
  - Row 21: $1,000-$1,249 | 5937 | 5938
  - Row 22: $1,250-$1,499 | 5947 | 5948
  - Row 23: $1,500-$1,749 | 5957 | 5958
  - Row 24: $1,750-$1,999 | 5967 | 5968
  - Row 25: $2,000-$2,999 | 5977 | 5978
  - Row 26: $3,000-$3,499 | 5987 | 5988
  - Row 27: $3,500 or more | 5997 | 5998
  - Row 29: Personal income not stated | 6007 | 6008
  - Row 31: Total | 6017 | 6018
  - Row 33: FEMALES
  - Row 35: Negative/Nil income | 6027 | 6028
  - Row 36: $1-$149 | 6037 | 6038
  - Row 37: $150-$299 | 6047 | 6048
  - Row 38: $300-$399 | 6057 | 6058
  - Row 39: $400-$499 | 6067 | 6068
  - Row 40: $500-$649 | 6077 | 6078

**Notes:** (None detected)

---

## G18: G18: G18 CORE ACTIVITY NEED FOR ASSISTANCE(a) BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 7: Has need, Does not have, Need for
  - Row 8: for assistance, need for assistance, assistance not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 9: MALES
  - Row 11: 0-4 years | 6367 | 6368
  - Row 12: 5-14 years | 6371 | 6372
  - Row 13: 15-19 years | 6375 | 6376
  - Row 14: 20-24 years | 6379 | 6380
  - Row 15: 25-34 years | 6383 | 6384
  - Row 16: 35-44 years | 6387 | 6388
  - Row 17: 45-54 years | 6391 | 6392
  - Row 18: 55-64 years | 6395 | 6396
  - Row 19: 65-74 years | 6399 | 6400
  - Row 20: 75-84 years | 6403 | 6404
  - Row 21: 85 years and over | 6407 | 6408
  - Row 23: Total | 6411 | 6412
  - Row 25: FEMALES
  - Row 27: 0-4 years | 6415 | 6416
  - Row 28: 5-14 years | 6419 | 6420
  - Row 29: 15-19 years | 6423 | 6424
  - Row 30: 20-24 years | 6427 | 6428
  - Row 31: 25-34 years | 6431 | 6432
  - Row 32: 35-44 years | 6435 | 6436
  - Row 33: 45-54 years | 6439 | 6440
  - Row 34: 55-64 years | 6443 | 6444
  - Row 35: 65-74 years | 6447 | 6448
  - Row 36: 75-84 years | 6451 | 6452
  - Row 37: 85 years and over | 6455 | 6456

**Notes:** (None detected)

---

## G19: G19: G19 TYPE OF LONG-TERM HEALTH CONDITION(a) BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 0-14, 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years, Total
  - Row 10: years, years, years, years, years, years, years, years, and over, responses(d)

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Arthritis | 6511 | 6512
  - Row 14: Asthma | 6521 | 6522
  - Row 15: Cancer (including remission) | 6531 | 6532
  - Row 16: Dementia (including Alzheimer's) | 6541 | 6542
  - Row 17: Diabetes (excluding gestational diabetes) | 6551 | 6552
  - Row 18: Heart disease (including heart attack or angina) | 6561 | 6562
  - Row 19: Kidney disease | 6571 | 6572
  - Row 20: Lung condition (including COPD or emphysema)(b) | 6581 | 6582
  - Row 21: Mental health condition (including depression or anxiety) | 6591 | 6592
  - Row 22: Stroke | 6601 | 6602
  - Row 23: Any other long-term health condition(s)(c) | 6611 | 6612
  - Row 24: No long-term health condition(s) | 6621 | 6622
  - Row 25: Not stated | 6631 | 6632
  - Row 27: Total males(d) | 6641 | 6642
  - Row 29: FEMALES
  - Row 31: Arthritis | 6651 | 6652
  - Row 32: Asthma | 6661 | 6662
  - Row 33: Cancer (including remission) | 6671 | 6672
  - Row 34: Dementia (including Alzheimer's) | 6681 | 6682
  - Row 35: Diabetes (excluding gestational diabetes) | 6691 | 6692
  - Row 36: Heart disease (including heart attack or angina) | 6701 | 6702
  - Row 37: Kidney disease | 6711 | 6712
  - Row 38: Lung condition (including COPD or emphysema)(b) | 6721 | 6722
  - Row 39: Mental health condition (including depression or anxiety) | 6731 | 6732
  - Row 40: Stroke | 6741 | 6742

**Notes (from bottom of template sheet):**
  - No long-term health condition(s)

---

## G20: G20: G20 COUNT OF SELECTED LONG-TERM HEALTH CONDITIONS(a) BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 0-14, 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 10: years, years, years, years, years, years, years, years, and over, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Count of selected long-term health conditions:
  - Row 14: None of the selected conditions | 6931 | 6932
  - Row 15: Has one or more selected long-term health conditions:
  - Row 16: One condition | 6941 | 6942
  - Row 17: Two conditions | 6951 | 6952
  - Row 18: Three or more conditions | 6961 | 6962
  - Row 19: Total | 6971 | 6972
  - Row 20: Not stated | 6981 | 6982
  - Row 22: Total males | 6991 | 6992
  - Row 24: FEMALES
  - Row 26: Count of selected long-term health conditions:
  - Row 27: None of the selected conditions | 7001 | 7002
  - Row 28: Has one or more selected long-term health conditions:
  - Row 29: One condition | 7011 | 7012
  - Row 30: Two conditions | 7021 | 7022
  - Row 31: Three or more conditions | 7031 | 7032
  - Row 32: Total | 7041 | 7042
  - Row 33: Not stated | 7051 | 7052
  - Row 35: Total females | 7061 | 7062
  - Row 37: PERSONS
  - Row 39: Count of selected long-term health conditions:
  - Row 40: None of the selected conditions | 7071 | 7072

**Notes (from bottom of template sheet):**
  - Three or more conditions
  - This table is based on place of usual residence.

---

## G21: G21: G21 TYPE OF LONG-TERM HEALTH CONDITION(a) BY SELECTED PERSON CHARACTERISTICS

**Potential Headers:** (None detected in typical range)

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 2: Australia (AUS) 7688094.9 sq kms
  - Row 4: G21 TYPE OF LONG-TERM HEALTH CONDITION(a) BY SELECTED PERSON CHARACTERISTICS
  - Row 5: Count of responses and persons
  - Row 14: Arthritis | Asthma
  - Row 16: Country of birth
  - Row 17: Australia(e) | 7141 | 7142
  - Row 18: Born overseas
  - Row 19: Other Oceania and Antarctica(f) | 7155 | 7156
  - Row 20: United Kingdom, Channel Islands and Isle of Man(g) | 7169 | 7170
  - Row 21: Other North-West Europe(h) | 7183 | 7184
  - Row 22: Southern and Eastern Europe | 7197 | 7198
  - Row 23: North Africa and the Middle East | 7211 | 7212
  - Row 24: South-East Asia | 7225 | 7226
  - Row 25: North-East Asia | 7239 | 7240
  - Row 26: Southern and Central Asia | 7253 | 7254
  - Row 27: Americas | 7267 | 7268
  - Row 28: Sub-Saharan Africa | 7281 | 7282
  - Row 29: Total overseas born | 7295 | 7296
  - Row 30: Country of birth not stated(i) | 7309 | 7310
  - Row 31: Total | 7323 | 7324

**Notes:** (None detected)

---

## G22: G22: G22 AUSTRALIAN DEFENCE FORCE SERVICE(a) BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 9: years, years, years, years, years, years, years, and over, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 10: MALES
  - Row 12: Has served in the Australian Defence Force:
  - Row 13: Currently serving in the regular service only | 7673 | 7674
  - Row 14: Currently serving in the reserve service only | 7682 | 7683
  - Row 15: Previously served (and not currently serving)(b) | 7691 | 7692
  - Row 16: Total ever served | 7700 | 7701
  - Row 17: Has never served | 7709 | 7710
  - Row 18: Not stated | 7718 | 7719
  - Row 20: Total males | 7727 | 7728
  - Row 22: FEMALES
  - Row 24: Has served in the Australian Defence Force:
  - Row 25: Currently serving in the regular service only | 7736 | 7737
  - Row 26: Currently serving in the reserve service only | 7745 | 7746
  - Row 27: Previously served (and not currently serving)(b) | 7754 | 7755
  - Row 28: Total ever served | 7763 | 7764
  - Row 29: Has never served | 7772 | 7773
  - Row 30: Not stated | 7781 | 7782
  - Row 32: Total females | 7790 | 7791
  - Row 34: PERSONS
  - Row 36: Has served in the Australian Defence Force:
  - Row 37: Currently serving in the regular service only | 7799 | 7800
  - Row 38: Currently serving in the reserve service only | 7808 | 7809
  - Row 39: Previously served (and not currently serving)(b) | 7817 | 7818

**Notes (from bottom of template sheet):**
  - This table is based on place of usual residence.
  - (a) Includes Royal Australian Navy, Australian Army, Royal Australian Air Force, Second Australian Imperial Force, National Service and NORFORCE. Excludes service for non-Australian Defence forces.
  - (b) Includes previous service in the regular service and/or reserves service.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G23: G23: G23 VOLUNTARY WORK FOR AN ORGANISATION OR GROUP BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Volunteer, Not a volunteer, work not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 9: MALES
  - Row 11: 15-19 years | 7862 | 7863
  - Row 12: 20-24 years | 7866 | 7867
  - Row 13: 25-34 years | 7870 | 7871
  - Row 14: 35-44 years | 7874 | 7875
  - Row 15: 45-54 years | 7878 | 7879
  - Row 16: 55-64 years | 7882 | 7883
  - Row 17: 65-74 years | 7886 | 7887
  - Row 18: 75-84 years | 7890 | 7891
  - Row 19: 85 years and over | 7894 | 7895
  - Row 21: Total | 7898 | 7899
  - Row 23: FEMALES
  - Row 25: 15-19 years | 7902 | 7903
  - Row 26: 20-24 years | 7906 | 7907
  - Row 27: 25-34 years | 7910 | 7911
  - Row 28: 35-44 years | 7914 | 7915
  - Row 29: 45-54 years | 7918 | 7919
  - Row 30: 55-64 years | 7922 | 7923
  - Row 31: 65-74 years | 7926 | 7927
  - Row 32: 75-84 years | 7930 | 7931
  - Row 33: 85 years and over | 7934 | 7935
  - Row 35: Total | 7938 | 7939
  - Row 37: PERSONS

**Notes:** (None detected)

---

## G24: G24: G24 UNPAID DOMESTIC WORK: NUMBER OF HOURS BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: Less than, 5 to 14, 15 to 29, 30 hours, Did no unpaid, Unpaid domestic
  - Row 10: 5 hours, hours, hours, or more, domestic work, work not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: 15-19 years | 7982 | 7983
  - Row 14: 20-24 years | 7989 | 7990
  - Row 15: 25-34 years | 7996 | 7997
  - Row 16: 35-44 years | 8003 | 8004
  - Row 17: 45-54 years | 8010 | 8011
  - Row 18: 55-64 years | 8017 | 8018
  - Row 19: 65-74 years | 8024 | 8025
  - Row 20: 75-84 years | 8031 | 8032
  - Row 21: 85 years and over | 8038 | 8039
  - Row 23: Total | 8045 | 8046
  - Row 25: FEMALES
  - Row 27: 15-19 years | 8052 | 8053
  - Row 28: 20-24 years | 8059 | 8060
  - Row 29: 25-34 years | 8066 | 8067
  - Row 30: 35-44 years | 8073 | 8074
  - Row 31: 45-54 years | 8080 | 8081
  - Row 32: 55-64 years | 8087 | 8088
  - Row 33: 65-74 years | 8094 | 8095
  - Row 34: 75-84 years | 8101 | 8102
  - Row 35: 85 years and over | 8108 | 8109
  - Row 37: Total | 8115 | 8116
  - Row 39: PERSONS

**Notes:** (None detected)

---

## G25: G25: G25 UNPAID ASSISTANCE TO A PERSON WITH A DISABILITY, HEALTH CONDITION OR DUE TO OLD AGE, BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Provided, No unpaid, Unpaid
  - Row 9: unpaid assistance, assistance provided, assistance not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 10: MALES
  - Row 12: 15-19 years | 8192 | 8193
  - Row 13: 20-24 years | 8196 | 8197
  - Row 14: 25-34 years | 8200 | 8201
  - Row 15: 35-44 years | 8204 | 8205
  - Row 16: 45-54 years | 8208 | 8209
  - Row 17: 55-64 years | 8212 | 8213
  - Row 18: 65-74 years | 8216 | 8217
  - Row 19: 75-84 years | 8220 | 8221
  - Row 20: 85 years and over | 8224 | 8225
  - Row 22: Total | 8228 | 8229
  - Row 24: FEMALES
  - Row 26: 15-19 years | 8232 | 8233
  - Row 27: 20-24 years | 8236 | 8237
  - Row 28: 25-34 years | 8240 | 8241
  - Row 29: 35-44 years | 8244 | 8245
  - Row 30: 45-54 years | 8248 | 8249
  - Row 31: 55-64 years | 8252 | 8253
  - Row 32: 65-74 years | 8256 | 8257
  - Row 33: 75-84 years | 8260 | 8261
  - Row 34: 85 years and over | 8264 | 8265
  - Row 36: Total | 8268 | 8269
  - Row 38: PERSONS

**Notes:** (None detected)

---

## G26: G26: G26 UNPAID CHILD CARE BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: Own child/, Did not, Unpaid
  - Row 10: Own child/, Other child/, children and, provide, child care
  - Row 11: children only, children only, other child/children, Total, child care, not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 12: MALES
  - Row 14: 15-19 years | 8312 | 8313
  - Row 15: 20-24 years | 8319 | 8320
  - Row 16: 25-34 years | 8326 | 8327
  - Row 17: 35-44 years | 8333 | 8334
  - Row 18: 45-54 years | 8340 | 8341
  - Row 19: 55-64 years | 8347 | 8348
  - Row 20: 65-74 years | 8354 | 8355
  - Row 21: 75-84 years | 8361 | 8362
  - Row 22: 85 years and over | 8368 | 8369
  - Row 24: Total | 8375 | 8376
  - Row 26: FEMALES
  - Row 28: 15-19 years | 8382 | 8383
  - Row 29: 20-24 years | 8389 | 8390
  - Row 30: 25-34 years | 8396 | 8397
  - Row 31: 35-44 years | 8403 | 8404
  - Row 32: 45-54 years | 8410 | 8411
  - Row 33: 55-64 years | 8417 | 8418
  - Row 34: 65-74 years | 8424 | 8425
  - Row 35: 75-84 years | 8431 | 8432
  - Row 36: 85 years and over | 8438 | 8439
  - Row 38: Total | 8445 | 8446
  - Row 40: PERSONS

**Notes:** (None detected)

---

## G27: G27: G27 RELATIONSHIP IN HOUSEHOLD BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 0-14, 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 10: years, years, years, years, years, years, years, years, and over, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Partner in a registered marriage(b) | .. | 8523
  - Row 14: Partner in de facto marriage(c) | .. | 8533
  - Row 15: Lone parent | .. | 8543
  - Row 16: Child under 15 | 8552 | ..
  - Row 17: Dependent student (aged 15-24 years) | .. | 8563
  - Row 18: Non-dependent child | .. | 8573
  - Row 19: Other related individual | .. | 8583
  - Row 20: Unrelated individual living in family household | .. | 8593
  - Row 21: Group household member | .. | 8603
  - Row 22: Lone person | .. | 8613
  - Row 23: Visitor (from within Australia)(d) | 8622 | 8623
  - Row 25: Total | 8632 | 8633
  - Row 27: FEMALES
  - Row 29: Partner in a registered marriage(b) | .. | 8643
  - Row 30: Partner in de facto marriage(c) | .. | 8653
  - Row 31: Lone parent | .. | 8663
  - Row 32: Child under 15 | 8672 | ..
  - Row 33: Dependent student (aged 15-24 years) | .. | 8683
  - Row 34: Non-dependent child | .. | 8693
  - Row 35: Other related individual | .. | 8703
  - Row 36: Unrelated individual living in family household | .. | 8713
  - Row 37: Group household member | .. | 8723
  - Row 38: Lone person | .. | 8733
  - Row 39: Visitor (from within Australia)(d) | 8742 | 8743

**Notes (from bottom of template sheet):**
  - Partner in a registered marriage(b)
  - Partner in de facto marriage(c)
  - Dependent student (aged 15-24 years)

---

## G28: G28: G28 NUMBER OF CHILDREN EVER BORN

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: No, One, Two, Three, Four, Five, Six or more
  - Row 10: children, child, children, children, children, children, children, Not stated, Total
  - Row 13: 15-19 years, 8882, 8883, 8884, 8885, 8886, 8887, 8888, 8889, 8890
  - Row 14: 20-24 years, 8891, 8892, 8893, 8894, 8895, 8896, 8897, 8898, 8899
  - Row 15: 25-29 years, 8900, 8901, 8902, 8903, 8904, 8905, 8906, 8907, 8908
  - Row 16: 30-34 years, 8909, 8910, 8911, 8912, 8913, 8914, 8915, 8916, 8917
  - Row 17: 35-39 years, 8918, 8919, 8920, 8921, 8922, 8923, 8924, 8925, 8926
  - Row 18: 40-44 years, 8927, 8928, 8929, 8930, 8931, 8932, 8933, 8934, 8935
  - Row 19: 45-49 years, 8936, 8937, 8938, 8939, 8940, 8941, 8942, 8943, 8944
  - Row 20: 50-54 years, 8945, 8946, 8947, 8948, 8949, 8950, 8951, 8952, 8953

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: 55-59 years | 8954 | 8955
  - Row 22: 60-64 years | 8963 | 8964
  - Row 23: 65-69 years | 8972 | 8973
  - Row 24: 70-74 years | 8981 | 8982
  - Row 25: 75-79 years | 8990 | 8991
  - Row 26: 80-84 years | 8999 | 9000
  - Row 27: 85 years and over | 9008 | 9009
  - Row 29: Total | 9017 | 9018
  - Row 31: This table is based on place of usual residence.
  - Row 33: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of usual residence.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G29: G29: G29 FAMILY COMPOSITION

**Potential Headers (from Template Rows ~5-20):**
  - Row 11: Couple family with no children, 9026, 9027
  - Row 15: dependent students and non-dependent children, 9028, 9029
  - Row 16: dependent students and no non-dependent children, 9030, 9031
  - Row 19: Total, 9036, 9037

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: no children under 15 and:
  - Row 22: with dependent students and non-dependent children | 9038 | 9039
  - Row 23: with dependent students and no non-dependent children | 9040 | 9041
  - Row 24: no dependent students and with non-dependent children | 9042 | 9043
  - Row 25: Total | 9044 | 9045
  - Row 27: Total | 9046 | 9047
  - Row 29: One parent family with:
  - Row 30: children under 15 and:
  - Row 31: dependent students and non-dependent children | 9048 | 9049
  - Row 32: dependent students and no non-dependent children | 9050 | 9051
  - Row 33: no dependent students and with non-dependent children | 9052 | 9053
  - Row 34: no dependent students and no non-dependent children | 9054 | 9055
  - Row 35: Total | 9056 | 9057
  - Row 37: no children under 15 and:
  - Row 38: with dependent students and non-dependent children | 9058 | 9059
  - Row 39: with dependent students and no non-dependent children | 9060 | 9061
  - Row 40: no dependent students and with non-dependent children | 9062 | 9063
  - Row 41: Total | 9064 | 9065
  - Row 43: Total | 9066 | 9067
  - Row 45: Other family | 9068 | 9069
  - Row 47: Total | 9070 | 9071
  - Row 49: This table is based on place of enumeration.

**Notes (from bottom of template sheet):**
  - This table is based on place of enumeration.
  - (a) Includes both same-sex couple families and opposite sex couple families.

---

## G30: G30: G30 FAMILY COMPOSITION AND COUNTRY OF BIRTH OF PARENTS BY AGE OF DEPENDENT CHILDREN(a)

**Potential Headers (from Template Rows ~5-20):**
  - Row 10: 0-4, 5-9, 10-12, 13-14, 15-17, 18-20, 21-24
  - Row 11: years, years, years, years, years, years, years, Total
  - Row 14: Both parents born overseas, 9072, 9073, 9074, 9075, 9076, 9077, 9078, 9079
  - Row 15: Father only born overseas, 9080, 9081, 9082, 9083, 9084, 9085, 9086, 9087
  - Row 16: Mother only born overseas, 9088, 9089, 9090, 9091, 9092, 9093, 9094, 9095
  - Row 17: Both parents born in Australia, 9096, 9097, 9098, 9099, 9100, 9101, 9102, 9103
  - Row 18: Birthplace not stated(c), 9104, 9105, 9106, 9107, 9108, 9109, 9110, 9111
  - Row 19: Total, 9112, 9113, 9114, 9115, 9116, 9117, 9118, 9119

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: One parent family:
  - Row 22: Both parents born overseas | 9120 | 9121
  - Row 23: Father only born overseas | 9128 | 9129
  - Row 24: Mother only born overseas | 9136 | 9137
  - Row 25: Both parents born in Australia | 9144 | 9145
  - Row 26: Birthplace not stated(c) | 9152 | 9153
  - Row 27: Total | 9160 | 9161
  - Row 29: Total | 9168 | 9169
  - Row 31: This table is based on place of enumeration.
  - Row 32: (a) Comprises children aged under 15 years and dependent students aged 15-24 years who were present at their usual residence on Census Night.
  - Row 33: (b) Includes both same-sex couple families and opposite sex couple families.
  - Row 34: (c) Includes birthplace for either or both parents not stated.
  - Row 36: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of enumeration.
  - (a) Comprises children aged under 15 years and dependent students aged 15-24 years who were present at their usual residence on Census Night.
  - (b) Includes both same-sex couple families and opposite sex couple families.
  - (c) Includes birthplace for either or both parents not stated.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G31: G31: G31 FAMILY BLENDING(a)

**Potential Headers:** (None detected in typical range)

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 2: Australia (AUS) 7688094.9 sq kms | Find out more:
  - Row 3: Family blending
  - Row 4: G31 FAMILY BLENDING(a)
  - Row 5: Count of couple families with children(b) (excludes overseas visitors)
  - Row 9: Families
  - Row 11: Intact family with no other children present | 9176
  - Row 12: Step family with no other children present | 9177
  - Row 13: Blended family with no other children present | 9178
  - Row 15: Intact family with other children present | 9179
  - Row 16: Step family with other children present | 9180
  - Row 17: Blended family with other children present | 9181
  - Row 19: Other couple family with other children only | 9182
  - Row 21: Total | 9183
  - Row 23: This table is based on place of enumeration.
  - Row 24: (a) Excludes 'Couple families with no children', 'One parent families' and 'Other families'.
  - Row 25: Excludes families in: 'Non-family/Non-classifiable households', 'Non-private dwellings' and 'Migratory, off-shore and
  - Row 26: shipping' SA1s.
  - Row 27: (b) Includes both same-sex couple families and opposite sex couple families.
  - Row 29: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These
  - Row 30: adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of enumeration.
  - (a) Excludes 'Couple families with no children', 'One parent families' and 'Other families'.
  - Excludes families in: 'Non-family/Non-classifiable households', 'Non-private dwellings' and 'Migratory, off-shore and
  - shipping' SA1s.
  - (b) Includes both same-sex couple families and opposite sex couple families.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These
  - adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G32: G32: G32 TOTAL FAMILY INCOME (WEEKLY) BY FAMILY COMPOSITION

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Couple family, Couple family, One
  - Row 9: with no children, with children, parent family, Other family, Total
  - Row 11: Negative/Nil income, 9184, 9185, 9186, 9187, 9188
  - Row 12: $1-$149, 9189, 9190, 9191, 9192, 9193
  - Row 13: $150-$299, 9194, 9195, 9196, 9197, 9198
  - Row 14: $300-$399, 9199, 9200, 9201, 9202, 9203
  - Row 15: $400-$499, 9204, 9205, 9206, 9207, 9208
  - Row 16: $500-$649, 9209, 9210, 9211, 9212, 9213
  - Row 17: $650-$799, 9214, 9215, 9216, 9217, 9218
  - Row 18: $800-$999, 9219, 9220, 9221, 9222, 9223
  - Row 19: $1,000-$1,249, 9224, 9225, 9226, 9227, 9228
  - Row 20: $1,250-$1,499, 9229, 9230, 9231, 9232, 9233

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: $1,500-$1,749 | 9234 | 9235
  - Row 22: $1,750-$1,999 | 9239 | 9240
  - Row 23: $2,000-$2,499 | 9244 | 9245
  - Row 24: $2,500-$2,999 | 9249 | 9250
  - Row 25: $3,000-$3,499 | 9254 | 9255
  - Row 26: $3,500-$3,999 | 9259 | 9260
  - Row 27: $4,000 or more | 9264 | 9265
  - Row 28: Partial income stated(c) | 9269 | 9270
  - Row 29: All incomes not stated(d) | 9274 | 9275
  - Row 31: Total | 9279 | 9280
  - Row 33: This table is based on place of enumeration.
  - Row 34: (a) Includes both same-sex couple families and opposite sex couple families.
  - Row 35: (b) Excludes 'Lone person', 'Group', 'Visitors only' and 'Other non-classifiable' households.
  - Row 36: (c) Comprises families where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent on Census Night.
  - Row 37: (d) Comprises families where no members present stated an income.
  - Row 39: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of enumeration.
  - (a) Includes both same-sex couple families and opposite sex couple families.
  - (b) Excludes 'Lone person', 'Group', 'Visitors only' and 'Other non-classifiable' households.
  - (c) Comprises families where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent on Census Night.
  - (d) Comprises families where no members present stated an income.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G33: G33: G33 TOTAL HOUSEHOLD INCOME (WEEKLY) BY HOUSEHOLD COMPOSITION

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: households, households(b), Total
  - Row 11: Negative/Nil income, 9284, 9285, 9286
  - Row 12: $1-$149, 9287, 9288, 9289
  - Row 13: $150-$299, 9290, 9291, 9292
  - Row 14: $300-$399, 9293, 9294, 9295
  - Row 15: $400-$499, 9296, 9297, 9298
  - Row 16: $500-$649, 9299, 9300, 9301
  - Row 17: $650-$799, 9302, 9303, 9304
  - Row 18: $800-$999, 9305, 9306, 9307
  - Row 19: $1,000-$1,249, 9308, 9309, 9310
  - Row 20: $1,250-$1,499, 9311, 9312, 9313

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: $1,500-$1,749 | 9314 | 9315
  - Row 22: $1,750-$1,999 | 9317 | 9318
  - Row 23: $2,000-$2,499 | 9320 | 9321
  - Row 24: $2,500-$2,999 | 9323 | 9324
  - Row 25: $3,000-$3,499 | 9326 | 9327
  - Row 26: $3,500-$3,999 | 9329 | 9330
  - Row 27: $4,000 or more | 9332 | 9333
  - Row 28: Partial income stated(c) | 9335 | 9336
  - Row 29: All incomes not stated(d) | 9338 | 9339
  - Row 31: Total | 9341 | 9342
  - Row 33: This table is based on place of enumeration.
  - Row 34: (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 35: (b) Comprises 'Lone person' and 'Group households'.
  - Row 36: (c) Comprises households where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent on Census Night.
  - Row 37: (d) Comprises households where no members present stated an income.
  - Row 39: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of enumeration.
  - (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - (b) Comprises 'Lone person' and 'Group households'.
  - (c) Comprises households where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent on Census Night.
  - (d) Comprises households where no members present stated an income.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G34: G34: G34 NUMBER OF MOTOR VEHICLES(a) BY DWELLINGS

**Potential Headers:** (None detected in typical range)

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 2: Australia (AUS) 7688094.9 sq kms | Find out more:
  - Row 3: Number of motor vehicles
  - Row 4: G34 NUMBER OF MOTOR VEHICLES(a) BY DWELLINGS | Dwelling
  - Row 5: Count of occupied private dwellings(b)
  - Row 8: Dwellings
  - Row 10: Number of motor vehicles per dwelling:
  - Row 11: No motor vehicles | 9344
  - Row 12: One motor vehicle | 9345
  - Row 13: Two motor vehicles | 9346
  - Row 14: Three motor vehicles | 9347
  - Row 15: Four or more motor vehicles | 9348
  - Row 16: Total | 9349
  - Row 18: Number of motor vehicles not stated | 9350
  - Row 20: Total | 9351
  - Row 22: This table is based on place of enumeration.
  - Row 23: (a) Excludes motorbikes, motor scooters and heavy vehicles.
  - Row 24: (b) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 26: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - Number of motor vehicles not stated
  - This table is based on place of enumeration.
  - (a) Excludes motorbikes, motor scooters and heavy vehicles.
  - (b) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G35: G35: G35 HOUSEHOLD COMPOSITION BY NUMBER OF PERSONS USUALLY RESIDENT(a)

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: households, households(c), Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: Number of persons usually resident:
  - Row 12: One | .. | 9353
  - Row 13: Two | 9355 | 9356
  - Row 14: Three | 9358 | 9359
  - Row 15: Four | 9361 | 9362
  - Row 16: Five | 9364 | 9365
  - Row 17: Six or more | 9367 | 9368
  - Row 19: Total | 9370 | 9371
  - Row 21: This table is based on place of enumeration.
  - Row 22: (a) Includes up to three residents who were temporarily absent on Census Night.
  - Row 23: (b) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 24: (c) Comprises 'Lone person' and 'Group households'.
  - Row 25: ..  Not applicable
  - Row 27: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of enumeration.
  - (a) Includes up to three residents who were temporarily absent on Census Night.
  - (b) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - (c) Comprises 'Lone person' and 'Group households'.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G36: G36: G36 DWELLING STRUCTURE

**Potential Headers (from Template Rows ~5-20):**
  - Row 12: Separate house, 9373, 9374
  - Row 15: One storey, 9375, 9376
  - Row 16: Two or more storeys, 9377, 9378
  - Row 17: Total, 9379, 9380
  - Row 20: In a one or two storey block, 9381, 9382

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: In a three storey block | 9383 | 9384
  - Row 22: In a four to eight storey block | 9385 | 9386
  - Row 23: In a nine or more storey block | 9387 | 9388
  - Row 24: Attached to a house | 9389 | 9390
  - Row 25: Total | 9391 | 9392
  - Row 27: Other dwelling:
  - Row 28: Caravan | 9393 | 9394
  - Row 29: Cabin, houseboat | 9395 | 9396
  - Row 30: Improvised home, tent, sleepers out | 9397 | 9398
  - Row 31: House or flat attached to a shop, office, etc. | 9399 | 9400
  - Row 32: Total | 9401 | 9402
  - Row 34: Dwelling structure not stated | 9403 | 9404
  - Row 36: Total occupied private dwellings | 9405 | 9406
  - Row 38: Unoccupied private dwellings | 9407 | ..
  - Row 40: Total private dwellings | 9409 | 9410
  - Row 42: This table is based on place of enumeration.
  - Row 43: (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 44: (b) Count of all persons enumerated in the dwelling on Census Night, including visitors from within Australia. Excludes usual residents who were temporarily
  - Row 45: absent on Census Night. Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 46: ..  Not applicable
  - Row 48: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - Total private dwellings
  - This table is based on place of enumeration.
  - (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - (b) Count of all persons enumerated in the dwelling on Census Night, including visitors from within Australia. Excludes usual residents who were temporarily
  - absent on Census Night. Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G37: G37: G37 TENURE AND LANDLORD TYPE BY DWELLING STRUCTURE

**Potential Headers (from Template Rows ~5-20):**
  - Row 11: Separate, terrace house,, Flat, Other
  - Row 12: house, townhouse etc., or apartment, dwelling, Not stated, Total
  - Row 14: Owned outright, 9411, 9412, 9413, 9414, 9415, 9416
  - Row 16: Owned with a mortgage(b), 9417, 9418, 9419, 9420, 9421, 9422
  - Row 19: Real estate agent, 9423, 9424, 9425, 9426, 9427, 9428
  - Row 20: State or territory housing authority, 9429, 9430, 9431, 9432, 9433, 9434

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Community housing provider | 9435 | 9436
  - Row 22: Person not in same household(d) | 9441 | 9442
  - Row 23: Other landlord type(e) | 9447 | 9448
  - Row 24: Landlord type not stated | 9453 | 9454
  - Row 25: Total | 9459 | 9460
  - Row 27: Other tenure type(f) | 9465 | 9466
  - Row 29: Tenure type not stated | 9471 | 9472
  - Row 31: Total | 9477 | 9478
  - Row 33: This table is based on place of enumeration.
  - Row 34: (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 35: (b) Includes dwellings being 'Purchased under a shared equity scheme'.
  - Row 36: (c) Excludes dwellings being 'Occupied rent-free'. In the 2016 Census, occupied rent-free was included as being rented.
  - Row 37: (d) Comprises dwellings being rented from a parent/other relative or other person.
  - Row 38: (e) Comprises dwellings being rented through a 'Owner/manager of a Residential park (including caravan parks and manufactured home estates)', 'Employer - Government (includes Defence Housing Australia)' and
  - Row 39: 'Employer - other employer'.
  - Row 40: (f) Includes dwellings being 'Occupied under a life tenure scheme' and 'Occupied rent-free'.
  - Row 42: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - (b) Includes dwellings being 'Purchased under a shared equity scheme'.
  - (c) Excludes dwellings being 'Occupied rent-free'. In the 2016 Census, occupied rent-free was included as being rented.
  - (d) Comprises dwellings being rented from a parent/other relative or other person.
  - (e) Comprises dwellings being rented through a 'Owner/manager of a Residential park (including caravan parks and manufactured home estates)', 'Employer - Government (includes Defence Housing Australia)' and
  - 'Employer - other employer'.
  - (f) Includes dwellings being 'Occupied under a life tenure scheme' and 'Occupied rent-free'.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G38: G38: G38 MORTGAGE REPAYMENT (MONTHLY) BY DWELLING STRUCTURE

**Potential Headers (from Template Rows ~5-20):**
  - Row 11: Separate, terrace house,, Flat, Other
  - Row 12: house, townhouse etc., or apartment, dwelling, Not stated, Total
  - Row 14: $0-$299, 9483, 9484, 9485, 9486, 9487, 9488
  - Row 15: $300-$449, 9489, 9490, 9491, 9492, 9493, 9494
  - Row 16: $450-$599, 9495, 9496, 9497, 9498, 9499, 9500
  - Row 17: $600-$799, 9501, 9502, 9503, 9504, 9505, 9506
  - Row 18: $800-$999, 9507, 9508, 9509, 9510, 9511, 9512
  - Row 19: $1,000-$1,399, 9513, 9514, 9515, 9516, 9517, 9518
  - Row 20: $1,400-$1,799, 9519, 9520, 9521, 9522, 9523, 9524

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: $1,800-$2,399 | 9525 | 9526
  - Row 22: $2,400-$2,999 | 9531 | 9532
  - Row 23: $3,000-$3,999 | 9537 | 9538
  - Row 24: $4,000 and over | 9543 | 9544
  - Row 25: Mortgage repayment not stated | 9549 | 9550
  - Row 27: Total | 9555 | 9556
  - Row 29: This table is based on place of enumeration.
  - Row 30: (a) Includes dwellings being purchased under a shared equity scheme. Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 32: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - Mortgage repayment not stated
  - This table is based on place of enumeration.
  - (a) Includes dwellings being purchased under a shared equity scheme. Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G39: G39: G39 MORTGAGE REPAYMENT (MONTHLY) BY FAMILY COMPOSITION

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: Children, No children, Children, No children, Other
  - Row 10: No children, under 15(b), under 15(b), under 15(b), under 15(b), family, Total
  - Row 12: $0–$149, 9561, 9562, 9563, 9564.0, 9564, 9565, 9566, 9567
  - Row 13: $150–$299, 9568, 9569, 9570, 9571.0, 9571, 9572, 9573, 9574
  - Row 14: $300–$449, 9575, 9576, 9577, 9578.0, 9578, 9579, 9580, 9581
  - Row 15: $450–$599, 9582, 9583, 9584, 9585.0, 9585, 9586, 9587, 9588
  - Row 16: $600–$799, 9589, 9590, 9591, 9592.0, 9592, 9593, 9594, 9595
  - Row 17: $800–$999, 9596, 9597, 9598, 9599.0, 9599, 9600, 9601, 9602
  - Row 18: $1,000–$1,199, 9603, 9604, 9605, 9606.0, 9606, 9607, 9608, 9609
  - Row 19: $1,200–$1,399, 9610, 9611, 9612, 9613.0, 9613, 9614, 9615, 9616
  - Row 20: $1,400–$1,599, 9617, 9618, 9619, 9620.0, 9620, 9621, 9622, 9623

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: $1,600–$1,799 | 9624 | 9625
  - Row 22: $1,800–$1,999 | 9631 | 9632
  - Row 23: $2,000–$2,199 | 9638 | 9639
  - Row 24: $2,200–$2,399 | 9645 | 9646
  - Row 25: $2,400–$2,599 | 9652 | 9653
  - Row 26: $2,600–$2,999 | 9659 | 9660
  - Row 27: $3,000–$3,999 | 9666 | 9667
  - Row 28: $4,000–$4,999 | 9673 | 9674
  - Row 29: $5,000 and over | 9680 | 9681
  - Row 31: Mortgage repayment not stated | 9687 | 9688
  - Row 33: Total | 9694 | 9695
  - Row 35: This table is based on place of enumeration.
  - Row 36: (a) Includes dwellings being purchased under a shared equity scheme. Excludes 'Lone person', 'Group', 'Visitors only' and 'Other non-classifiable' households.
  - Row 37: (b) Includes families that also have dependent students and non-dependent children.
  - Row 39: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - Mortgage repayment not stated
  - This table is based on place of enumeration.
  - (a) Includes dwellings being purchased under a shared equity scheme. Excludes 'Lone person', 'Group', 'Visitors only' and 'Other non-classifiable' households.
  - (b) Includes families that also have dependent students and non-dependent children.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G40: G40: G40 RENT (WEEKLY) BY LANDLORD TYPE

**Potential Headers (from Template Rows ~5-20):**
  - Row 10: State or territory, Community, Person not in, Other
  - Row 11: Real estate agent, housing authority, housing provider, same household(c), landlord type(d), Not stated, Total
  - Row 13: $1-$74, 9701, 9702, 9703, 9704, 9705, 9706, 9707
  - Row 14: $75-$99, 9708, 9709, 9710, 9711, 9712, 9713, 9714
  - Row 15: $100-$149, 9715, 9716, 9717, 9718, 9719, 9720, 9721
  - Row 16: $150-$199, 9722, 9723, 9724, 9725, 9726, 9727, 9728
  - Row 17: $200-$224, 9729, 9730, 9731, 9732, 9733, 9734, 9735
  - Row 18: $225-$274, 9736, 9737, 9738, 9739, 9740, 9741, 9742
  - Row 19: $275-$349, 9743, 9744, 9745, 9746, 9747, 9748, 9749
  - Row 20: $350-$449, 9750, 9751, 9752, 9753, 9754, 9755, 9756

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: $450-$549 | 9757 | 9758
  - Row 22: $550-$649 | 9764 | 9765
  - Row 23: $650-$749 | 9771 | 9772
  - Row 24: $750-$849 | 9778 | 9779
  - Row 25: $850-$949 | 9785 | 9786
  - Row 26: $950 and over | 9792 | 9793
  - Row 27: Rent not stated | 9799 | 9800
  - Row 29: Total | 9806 | 9807
  - Row 31: This table is based on place of enumeration.
  - Row 32: (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 33: (b) Excludes dwellings being 'Occupied rent-free'. In the 2016 Census, occupied rent-free was included as being rented.
  - Row 34: (c) Comprises dwellings being rented from a parent/other relative and other person.
  - Row 35: (d) Comprises dwellings being rented through a 'Owner/manager of a residential park (including caravan parks and manufactured home estates)', 'Employer - Government (includes Defence Housing Australia)' and
  - Row 36: 'Employer - other employer'.
  - Row 38: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of enumeration.
  - (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - (b) Excludes dwellings being 'Occupied rent-free'. In the 2016 Census, occupied rent-free was included as being rented.
  - (c) Comprises dwellings being rented from a parent/other relative and other person.
  - (d) Comprises dwellings being rented through a 'Owner/manager of a residential park (including caravan parks and manufactured home estates)', 'Employer - Government (includes Defence Housing Australia)' and
  - 'Employer - other employer'.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G41: G41: G41 DWELLING STRUCTURE BY NUMBER OF BEDROOMS

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: studio apartments, One, Two, Three, Four, Five, bedrooms
  - Row 10: or bedsitters), bedroom, bedrooms, bedrooms, bedrooms, bedrooms, or more, Not stated, Total
  - Row 12: Separate house, 9813, 9814, 9815, 9816, 9817, 9818, 9819, 9820, 9821
  - Row 15: One storey, 9822, 9823, 9824, 9825, 9826, 9827, 9828, 9829, 9830
  - Row 16: Two or more storeys, 9831, 9832, 9833, 9834, 9835, 9836, 9837, 9838, 9839
  - Row 17: Total, 9840, 9841, 9842, 9843, 9844, 9845, 9846, 9847, 9848
  - Row 20: In a one or two storey block, 9849, 9850, 9851, 9852, 9853, 9854, 9855, 9856, 9857

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: In a three storey block | 9858 | 9859
  - Row 22: In a four to eight storey block | 9867 | 9868
  - Row 23: In a nine or more storey block | 9876 | 9877
  - Row 24: Attached to a house | 9885 | 9886
  - Row 25: Total | 9894 | 9895
  - Row 27: Other dwelling | 9903 | 9904
  - Row 29: Dwelling structure not stated | 9912 | 9913
  - Row 31: Total | 9921 | 9922
  - Row 33: This table is based on place of enumeration.
  - Row 34: (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 36: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - Dwelling structure not stated
  - This table is based on place of enumeration.
  - (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G42: G42: G42 DWELLING STRUCTURE BY HOUSEHOLD COMPOSITION AND FAMILY COMPOSITION

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: Couple family, Couple family, One parent, Other, Lone person, Group
  - Row 10: with no children, with children, family, family, Total, households, households, Total
  - Row 12: Separate house, 9930, 9931, 9932, 9933, 9934, 9935, 9936, 9937
  - Row 15: One storey, 9938, 9939, 9940, 9941, 9942, 9943, 9944, 9945
  - Row 16: Two or more storeys, 9946, 9947, 9948, 9949, 9950, 9951, 9952, 9953
  - Row 17: Total, 9954, 9955, 9956, 9957, 9958, 9959, 9960, 9961
  - Row 20: In a one or two storey block, 9962, 9963, 9964, 9965, 9966, 9967, 9968, 9969

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: In a three storey block | 9970 | 9971
  - Row 22: In a four to eight storey block | 9978 | 9979
  - Row 23: In a nine or more storey block | 9986 | 9987
  - Row 24: Attached to a house | 9994 | 9995
  - Row 25: Total | 10002 | 10003
  - Row 27: Other dwelling:
  - Row 28: Caravan | 10010 | 10011
  - Row 29: Cabin, houseboat | 10018 | 10019
  - Row 30: Improvised home, tent, sleepers out | 10026 | 10027
  - Row 31: House or flat attached to a shop, office etc. | 10034 | 10035
  - Row 32: Total | 10042 | 10043
  - Row 34: Dwelling structure not stated | 10050 | 10051
  - Row 36: Total | 10058 | 10059
  - Row 38: This table is based on place of enumeration.
  - Row 39: (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - Row 40: (b) In multiple family households, only the family composition of the primary family is included.
  - Row 42: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - Dwelling structure not stated
  - This table is based on place of enumeration.
  - (a) Excludes 'Visitors only' and 'Other non-classifiable' households.
  - (b) In multiple family households, only the family composition of the primary family is included.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G43: G43: G43 SELECTED LABOUR FORCE, EDUCATION AND MIGRATION CHARACTERISTICS BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 10: Males, Females, Persons
  - Row 12: Persons aged 15 years and over, 10066, 10067, 10068
  - Row 15: Employed, worked full-time(b), 10069, 10070, 10071
  - Row 16: Employed, worked part-time(c), 10072, 10073, 10074
  - Row 17: Employed, away from work(d), 10075, 10076, 10077
  - Row 18: Unemployed, looking for work(e), 10078, 10079, 10080
  - Row 19: Total labour force, 10081, 10082, 10083

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Not in the labour force | 10084 | 10085
  - Row 23: % Unemployment(f) | 10087 | 10088
  - Row 24: % Labour force participation(g) | 10090 | 10091
  - Row 25: % Employment to population(h) | 10093 | 10094
  - Row 27: Highest non-school qualifications(a):
  - Row 28: Postgraduate Degree Level | 10096 | 10097
  - Row 29: Graduate Diploma and Graduate Certificate Level | 10099 | 10100
  - Row 30: Bachelor Degree Level | 10102 | 10103
  - Row 31: Advanced Diploma and Diploma Level | 10105 | 10106
  - Row 32: Certificate Level:
  - Row 33: Certificate III & IV Level(i) | 10108 | 10109
  - Row 34: Certificate I & II Level(j) | 10111 | 10112
  - Row 35: Certificate Level, nfd | 10114 | 10115
  - Row 36: Total Certificate Level | 10117 | 10118
  - Row 38: Migration:
  - Row 39: Lived at same address 1 year ago(k) | 10120 | 10121
  - Row 40: Lived at different address 1 year ago(k) | 10123 | 10124
  - Row 42: Lived at same address 5 years ago(l) | 10126 | 10127
  - Row 43: Lived at different address 5 years ago(l) | 10129 | 10130
  - Row 45: This table is based on place of usual residence.
  - Row 46: (a) Applicable to persons aged 15 years and over.
  - Row 47: (b) 'Employed, worked full-time' is defined as having worked 35 hours or more in all jobs during the week prior to Census Night.
  - Row 48: (c) 'Employed, worked part-time' is defined as having worked less than 35 hours in all jobs during the week prior to Census Night.
  - Row 49: (d) Comprises employed persons who did not work any hours in the week prior to Census Night and who did not state their number of hours worked.

**Notes (from bottom of template sheet):**
  - Lived at same address 5 years ago(l)
  - Lived at different address 5 years ago(l)
  - This table is based on place of usual residence.
  - (a) Applicable to persons aged 15 years and over.
  - (b) 'Employed, worked full-time' is defined as having worked 35 hours or more in all jobs during the week prior to Census Night.
  - (c) 'Employed, worked part-time' is defined as having worked less than 35 hours in all jobs during the week prior to Census Night.
  - (d) Comprises employed persons who did not work any hours in the week prior to Census Night and who did not state their number of hours worked.
  - (e) 'Unemployed' comprises of unemployed persons looking for full and part-time work.

---

## G44: G44: G44 PLACE OF USUAL RESIDENCE 1 YEAR AGO BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Males, Females, Persons
  - Row 11: Same usual address 1 year ago as in 2021, 10132, 10133, 10134
  - Row 14: Same Statistical Area Level 2 (SA2), 10135, 10136, 10137
  - Row 16: New South Wales, 10138, 10139, 10140
  - Row 17: Victoria, 10141, 10142, 10143
  - Row 18: Queensland, 10144, 10145, 10146
  - Row 19: South Australia, 10147, 10148, 10149
  - Row 20: Western Australia, 10150, 10151, 10152

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Tasmania | 10153 | 10154
  - Row 22: Northern Territory | 10156 | 10157
  - Row 23: Australian Capital Territory | 10159 | 10160
  - Row 24: Other Territories | 10162 | 10163
  - Row 25: Total | 10165 | 10166
  - Row 26: Overseas | 10168 | 10169
  - Row 27: Not stated(a) | 10171 | 10172
  - Row 28: Total | 10174 | 10175
  - Row 30: Not stated(b) | 10177 | 10178
  - Row 32: Total | 10180 | 10181
  - Row 34: This table is based on place of usual residence.
  - Row 35: (a) Includes persons who stated that they lived at a different address 1 year ago but did not state that address.
  - Row 36: (b) Includes persons who did not state whether they were usually resident at a different address 1 year ago.
  - Row 38: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of usual residence.
  - (a) Includes persons who stated that they lived at a different address 1 year ago but did not state that address.
  - (b) Includes persons who did not state whether they were usually resident at a different address 1 year ago.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G45: G45: G45 PLACE OF USUAL RESIDENCE 5 YEARS AGO BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Males, Females, Persons
  - Row 11: Same usual address 5 years ago as in 2021, 10183, 10184, 10185
  - Row 14: Same Statistical Area Level 2 (SA2), 10186, 10187, 10188
  - Row 16: New South Wales, 10189, 10190, 10191
  - Row 17: Victoria, 10192, 10193, 10194
  - Row 18: Queensland, 10195, 10196, 10197
  - Row 19: South Australia, 10198, 10199, 10200
  - Row 20: Western Australia, 10201, 10202, 10203

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Tasmania | 10204 | 10205
  - Row 22: Northern Territory | 10207 | 10208
  - Row 23: Australian Capital Territory | 10210 | 10211
  - Row 24: Other Territories | 10213 | 10214
  - Row 25: Total | 10216 | 10217
  - Row 26: Overseas | 10219 | 10220
  - Row 27: Not stated(a) | 10222 | 10223
  - Row 28: Total | 10225 | 10226
  - Row 30: Not stated(b) | 10228 | 10229
  - Row 32: Total | 10231 | 10232
  - Row 34: This table is based on place of usual residence.
  - Row 35: (a) Includes persons who stated that they lived at a different address 5 years ago but did not state that address.
  - Row 36: (b) Includes persons who did not state whether they were usually resident at a different address 5 years ago.
  - Row 38: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of usual residence.
  - (a) Includes persons who stated that they lived at a different address 5 years ago but did not state that address.
  - (b) Includes persons who did not state whether they were usually resident at a different address 5 years ago.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G46: G46: G46 LABOUR FORCE STATUS BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 15-19, 20-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 10: years, years, years, years, years, years, years, years, and over, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Employed, worked:
  - Row 14: Full-time(a) | 10234 | 10235
  - Row 15: Part-time(b) | 10244 | 10245
  - Row 16: Employed, away from work(c) | 10254 | 10255
  - Row 17: Hours worked not stated | 10264 | 10265
  - Row 18: Total employed | 10274 | 10275
  - Row 20: Unemployed, looking for:
  - Row 21: Full-time work | 10284 | 10285
  - Row 22: Part-time work | 10294 | 10295
  - Row 23: Total unemployed | 10304 | 10305
  - Row 25: Total labour force | 10314 | 10315
  - Row 27: Not in the labour force | 10324 | 10325
  - Row 28: Labour force status not stated | 10334 | 10335
  - Row 30: Total | 10344 | 10345
  - Row 32: FEMALES
  - Row 34: Employed, worked:
  - Row 35: Full-time(a) | 10354 | 10355
  - Row 36: Part-time(b) | 10364 | 10365
  - Row 37: Employed, away from work(c) | 10374 | 10375
  - Row 38: Hours worked not stated | 10384 | 10385
  - Row 39: Total employed | 10394 | 10395

**Notes (from bottom of template sheet):**
  - Not in the labour force
  - Labour force status not stated

---

## G47: Labour Force Status by Sex of Parents by Age of Dependent Children for Couple Families

**Error processing this table:** Sheet not found in template.

---

## G48: G48: G48 LABOUR FORCE STATUS BY SEX OF PARENT BY AGE OF DEPENDENT CHILDREN(a) FOR ONE PARENT FAMILIES

**Potential Headers (from Template Rows ~5-20):**
  - Row 12: 0-4, 5-9, 10-12, 13-14, 15-17, 18-20, 21-24
  - Row 13: years, years, years, years, years, years, years, Total
  - Row 17: Full-time(b), 11746, 11747, 11748, 11749, 11750, 11751, 11752, 11753
  - Row 18: Part-time(c), 11754, 11755, 11756, 11757, 11758, 11759, 11760, 11761
  - Row 19: Employed, away from work(d), 11762, 11763, 11764, 11765, 11766, 11767, 11768, 11769
  - Row 20: Hours worked not stated, 11770, 11771, 11772, 11773, 11774, 11775, 11776, 11777

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Total employed | 11778 | 11779
  - Row 23: Unemployed, looking for:
  - Row 24: Full-time work | 11786 | 11787
  - Row 25: Part-time work | 11794 | 11795
  - Row 26: Total unemployed | 11802 | 11803
  - Row 28: Total labour force | 11810 | 11811
  - Row 30: Not in the labour force | 11818 | 11819
  - Row 31: Labour force status not stated | 11826 | 11827
  - Row 33: Total | 11834 | 11835
  - Row 35: FEMALE LONE PARENT
  - Row 37: Employed, worked:
  - Row 38: Full-time(b) | 11842 | 11843
  - Row 39: Part-time(c) | 11850 | 11851
  - Row 40: Employed, away from work(d) | 11858 | 11859
  - Row 41: Hours worked not stated | 11866 | 11867
  - Row 42: Total employed | 11874 | 11875
  - Row 44: Unemployed, looking for:
  - Row 45: Full-time work | 11882 | 11883
  - Row 46: Part-time work | 11890 | 11891
  - Row 47: Total unemployed | 11898 | 11899
  - Row 49: Total labour force | 11906 | 11907

**Notes (from bottom of template sheet):**
  - Unemployed, looking for:

---

## G49: G49: G49 HIGHEST NON-SCHOOL QUALIFICATION: LEVEL OF EDUCATION(a) BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 10: years, years, years, years, years, years, years, and over, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Postgraduate Degree Level | 12034 | 12035
  - Row 15: Graduate Diploma and Graduate Certificate Level | 12043 | 12044
  - Row 17: Bachelor Degree Level | 12052 | 12053
  - Row 19: Advanced Diploma and Diploma Level | 12061 | 12062
  - Row 21: Certificate Level:
  - Row 22: Certificate III & IV Level(c) | 12070 | 12071
  - Row 23: Certificate I & II Level(d) | 12079 | 12080
  - Row 24: Certificate Level, nfd | 12088 | 12089
  - Row 25: Total | 12097 | 12098
  - Row 27: Level of education inadequately described | 12106 | 12107
  - Row 29: Level of education not stated | 12115 | 12116
  - Row 31: Total | 12124 | 12125
  - Row 33: FEMALES
  - Row 35: Postgraduate Degree Level | 12133 | 12134
  - Row 37: Graduate Diploma and Graduate Certificate Level | 12142 | 12143
  - Row 39: Bachelor Degree Level | 12151 | 12152

**Notes (from bottom of template sheet):**
  - Certificate III & IV Level(c)
  - Certificate I & II Level(d)
  - Certificate Level, nfd
  - Level of education inadequately described

---

## G50: G50: G50 HIGHEST NON-SCHOOL QUALIFICATION: FIELD OF STUDY BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 10: years, years, years, years, years, years, years, and over, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Natural and Physical Sciences | 12331 | 12332
  - Row 14: Information Technology | 12340 | 12341
  - Row 15: Engineering and Related Technologies | 12349 | 12350
  - Row 16: Architecture and Building | 12358 | 12359
  - Row 17: Agriculture, Environmental and Related Studies | 12367 | 12368
  - Row 18: Health | 12376 | 12377
  - Row 19: Education | 12385 | 12386
  - Row 20: Management and Commerce | 12394 | 12395
  - Row 21: Society and Culture | 12403 | 12404
  - Row 22: Creative Arts | 12412 | 12413
  - Row 23: Food, Hospitality and Personal Services | 12421 | 12422
  - Row 24: Mixed Field Programmes | 12430 | 12431
  - Row 25: Field of study inadequately described | 12439 | 12440
  - Row 26: Field of study not stated | 12448 | 12449
  - Row 28: Total | 12457 | 12458
  - Row 30: FEMALES
  - Row 32: Natural and Physical Sciences | 12466 | 12467
  - Row 33: Information Technology | 12475 | 12476
  - Row 34: Engineering and Related Technologies | 12484 | 12485
  - Row 35: Architecture and Building | 12493 | 12494
  - Row 36: Agriculture, Environmental and Related Studies | 12502 | 12503
  - Row 37: Health | 12511 | 12512
  - Row 38: Education | 12520 | 12521
  - Row 39: Management and Commerce | 12529 | 12530
  - Row 40: Society and Culture | 12538 | 12539

**Notes (from bottom of template sheet):**
  - Food, Hospitality and Personal Services
  - Mixed Field Programmes
  - Field of study inadequately described
  - Field of study not stated

---

## G51: G51: G51 HIGHEST NON-SCHOOL QUALIFICATION: FIELD OF STUDY BY OCCUPATION BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: Technicians, Community, Clerical and, Machinery, Inadequately
  - Row 10: and trades, and personal, administrative, Sales, operators, described/
  - Row 11: Managers, Professionals, workers, service workers, workers, workers, and drivers, Labourers, Not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 12: MALES
  - Row 14: Natural and Physical Sciences | 12736 | 12737
  - Row 15: Information Technology | 12746 | 12747
  - Row 16: Engineering and Related Technologies | 12756 | 12757
  - Row 17: Architecture and Building | 12766 | 12767
  - Row 18: Agriculture, Environmental and Related Studies | 12776 | 12777
  - Row 19: Health | 12786 | 12787
  - Row 20: Education | 12796 | 12797
  - Row 21: Management and Commerce | 12806 | 12807
  - Row 22: Society and Culture | 12816 | 12817
  - Row 23: Creative Arts | 12826 | 12827
  - Row 24: Food, Hospitality and Personal Services | 12836 | 12837
  - Row 25: Mixed Field Programmes | 12846 | 12847
  - Row 26: Field of study inadequately described | 12856 | 12857
  - Row 27: Field of study not stated | 12866 | 12867
  - Row 29: Total | 12876 | 12877
  - Row 31: FEMALES
  - Row 33: Natural and Physical Sciences | 12886 | 12887
  - Row 34: Information Technology | 12896 | 12897
  - Row 35: Engineering and Related Technologies | 12906 | 12907
  - Row 36: Architecture and Building | 12916 | 12917
  - Row 37: Agriculture, Environmental and Related Studies | 12926 | 12927
  - Row 38: Health | 12936 | 12937
  - Row 39: Education | 12946 | 12947
  - Row 40: Management and Commerce | 12956 | 12957
  - Row 41: Society and Culture | 12966 | 12967

**Notes (from bottom of template sheet):**
  - Food, Hospitality and Personal Services
  - Mixed Field Programmes
  - Field of study inadequately described
  - Field of study not stated

---

## G52: G52: G52 HIGHEST NON-SCHOOL QUALIFICATION: LEVEL OF EDUCATION(a) BY OCCUPATION BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: Technicians, Community, Clerical and, Machinery, Inadequately
  - Row 10: and trades, and personal, administrative, Sales, operators, described/
  - Row 11: Managers, Professionals, workers, service workers, workers, workers, and drivers, Labourers, Not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 12: MALES
  - Row 14: Postgraduate Degree Level:
  - Row 15: Postgraduate Degree Level, nfd | 13186 | 13187
  - Row 16: Doctoral Degree Level | 13196 | 13197
  - Row 17: Master Degree Level | 13206 | 13207
  - Row 19: Graduate Diploma and Graduate Certificate Level:
  - Row 20: Graduate Diploma and Graduate Certificate Level, nfd | 13216 | 13217
  - Row 21: Graduate Diploma Level | 13226 | 13227
  - Row 22: Graduate Certificate Level | 13236 | 13237
  - Row 24: Bachelor Degree Level:
  - Row 25: Bachelor Degree Level | 13246 | 13247
  - Row 27: Advanced Diploma and Diploma Level:
  - Row 28: Advanced Diploma and Diploma Level, nfd | 13256 | 13257
  - Row 29: Advanced Diploma and Associate Degree Level | 13266 | 13267
  - Row 30: Diploma Level | 13276 | 13277
  - Row 32: Certificate Level:
  - Row 33: Certificate III & IV Level(c) | 13286 | 13287
  - Row 34: Certificate I & II Level(d) | 13296 | 13297
  - Row 35: Certificate Level, nfd | 13306 | 13307
  - Row 37: Level of education inadequately described | 13316 | 13317
  - Row 38: Level of education not stated | 13326 | 13327
  - Row 40: Total | 13336 | 13337

**Notes (from bottom of template sheet):**
  - Postgraduate Degree Level:
  - Postgraduate Degree Level, nfd
  - Doctoral Degree Level
  - Graduate Diploma and Graduate Certificate Level:
  - Graduate Diploma and Graduate Certificate Level, nfd

---

## G53: G53: G53 HIGHEST NON-SCHOOL QUALIFICATION: LEVEL OF EDUCATION(a) BY INDUSTRY OF EMPLOYMENT BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: Graduate, Bachelor, and, Inadequately
  - Row 10: Postgraduate, Certificate, Degree, Diploma, Certificate, described/
  - Row 11: Degree Level, Level, Level, Level, Level, Not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 12: MALES
  - Row 14: Agriculture, Forestry and Fishing | 13666 | 13667
  - Row 15: Mining | 13673 | 13674
  - Row 16: Manufacturing | 13680 | 13681
  - Row 17: Electricity, Gas, Water and Waste Services | 13687 | 13688
  - Row 18: Construction | 13694 | 13695
  - Row 19: Wholesale Trade | 13701 | 13702
  - Row 20: Retail Trade | 13708 | 13709
  - Row 21: Accommodation and Food Services | 13715 | 13716
  - Row 22: Transport, Postal and Warehousing | 13722 | 13723
  - Row 23: Information Media and Telecommunications | 13729 | 13730
  - Row 24: Financial and Insurance Services | 13736 | 13737
  - Row 25: Rental, Hiring and Real Estate Services | 13743 | 13744
  - Row 26: Professional, Scientific and Technical Services | 13750 | 13751
  - Row 27: Administrative and Support Services | 13757 | 13758
  - Row 28: Public Administration and Safety | 13764 | 13765
  - Row 29: Education and Training | 13771 | 13772
  - Row 30: Health Care and Social Assistance | 13778 | 13779
  - Row 31: Arts and Recreation Services | 13785 | 13786
  - Row 32: Other Services | 13792 | 13793
  - Row 33: Inadequately described/Not stated | 13799 | 13800
  - Row 35: Total | 13806 | 13807
  - Row 37: FEMALES
  - Row 39: Agriculture, Forestry and Fishing | 13813 | 13814
  - Row 40: Mining | 13820 | 13821
  - Row 41: Manufacturing | 13827 | 13828

**Notes (from bottom of template sheet):**
  - Electricity, Gas, Water and Waste Services
  - Accommodation and Food Services
  - Transport, Postal and Warehousing
  - Information Media and Telecommunications
  - Financial and Insurance Services
  - Rental, Hiring and Real Estate Services

---

## G54: G54: G54 INDUSTRY OF EMPLOYMENT BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 15-19, 20-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85 years
  - Row 10: years, years, years, years, years, years, years, years, and over, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Agriculture, Forestry and Fishing | 14107 | 14108
  - Row 14: Mining | 14117 | 14118
  - Row 15: Manufacturing | 14127 | 14128
  - Row 16: Electricity, Gas, Water and Waste Services | 14137 | 14138
  - Row 17: Construction | 14147 | 14148
  - Row 18: Wholesale Trade | 14157 | 14158
  - Row 19: Retail Trade | 14167 | 14168
  - Row 20: Accommodation and Food Services | 14177 | 14178
  - Row 21: Transport, Postal and Warehousing | 14187 | 14188
  - Row 22: Information Media and Telecommunications | 14197 | 14198
  - Row 23: Financial and Insurance Services | 14207 | 14208
  - Row 24: Rental, Hiring and Real Estate Services | 14217 | 14218
  - Row 25: Professional, Scientific and Technical Services | 14227 | 14228
  - Row 26: Administrative and Support Services | 14237 | 14238
  - Row 27: Public Administration and Safety | 14247 | 14248
  - Row 28: Education and Training | 14257 | 14258
  - Row 29: Health Care and Social Assistance | 14267 | 14268
  - Row 30: Arts and Recreation Services | 14277 | 14278
  - Row 31: Other Services | 14287 | 14288
  - Row 32: Inadequately described/Not stated | 14297 | 14298
  - Row 34: Total | 14307 | 14308
  - Row 36: FEMALES
  - Row 38: Agriculture, Forestry and Fishing | 14317 | 14318
  - Row 39: Mining | 14327 | 14328
  - Row 40: Manufacturing | 14337 | 14338

**Notes (from bottom of template sheet):**
  - Accommodation and Food Services
  - Transport, Postal and Warehousing
  - Information Media and Telecommunications
  - Financial and Insurance Services
  - Rental, Hiring and Real Estate Services
  - Professional, Scientific and Technical Services

---

## G55: G55: G55 INDUSTRY OF EMPLOYMENT BY HOURS WORKED BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 1-19, 20-29, 30-34, 35-39, 40-44, 45-49, 50 hours
  - Row 10: None(a), hours, hours, hours, hours, hours, hours, and over, Not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Agriculture, Forestry and Fishing | 14737 | 14738
  - Row 14: Mining | 14747 | 14748
  - Row 15: Manufacturing | 14757 | 14758
  - Row 16: Electricity, Gas, Water and Waste Services | 14767 | 14768
  - Row 17: Construction | 14777 | 14778
  - Row 18: Wholesale Trade | 14787 | 14788
  - Row 19: Retail Trade | 14797 | 14798
  - Row 20: Accommodation and Food Services | 14807 | 14808
  - Row 21: Transport, Postal and Warehousing | 14817 | 14818
  - Row 22: Information Media and Telecommunications | 14827 | 14828
  - Row 23: Financial and Insurance Services | 14837 | 14838
  - Row 24: Rental, Hiring and Real Estate Services | 14847 | 14848
  - Row 25: Professional, Scientific and Technical Services | 14857 | 14858
  - Row 26: Administrative and Support Services | 14867 | 14868
  - Row 27: Public Administration and Safety | 14877 | 14878
  - Row 28: Education and Training | 14887 | 14888
  - Row 29: Health Care and Social Assistance | 14897 | 14898
  - Row 30: Arts and Recreation Services | 14907 | 14908
  - Row 31: Other Services | 14917 | 14918
  - Row 32: Inadequately described/Not stated | 14927 | 14928
  - Row 34: Total | 14937 | 14938
  - Row 36: FEMALES
  - Row 38: Agriculture, Forestry and Fishing | 14947 | 14948
  - Row 39: Mining | 14957 | 14958
  - Row 40: Manufacturing | 14967 | 14968

**Notes (from bottom of template sheet):**
  - Accommodation and Food Services
  - Transport, Postal and Warehousing
  - Information Media and Telecommunications
  - Financial and Insurance Services
  - Rental, Hiring and Real Estate Services
  - Professional, Scientific and Technical Services

---

## G56: G56: G56 INDUSTRY OF EMPLOYMENT BY OCCUPATION

**Potential Headers (from Template Rows ~5-20):**
  - Row 10: Technicians, and personal, Clerical and, Machinery, Inadequately
  - Row 11: and trades, service, administrative, Sales, operators, described/
  - Row 12: Managers, Professionals, workers, workers, workers, workers, and drivers, Labourers, Not stated, Total
  - Row 14: Agriculture, Forestry and Fishing, 15367, 15368, 15369, 15370, 15371, 15372, 15373, 15374, 15375, 15376
  - Row 15: Mining, 15377, 15378, 15379, 15380, 15381, 15382, 15383, 15384, 15385, 15386
  - Row 16: Manufacturing, 15387, 15388, 15389, 15390, 15391, 15392, 15393, 15394, 15395, 15396
  - Row 17: Electricity, Gas, Water and Waste Services, 15397, 15398, 15399, 15400, 15401, 15402, 15403, 15404, 15405, 15406
  - Row 18: Construction, 15407, 15408, 15409, 15410, 15411, 15412, 15413, 15414, 15415, 15416
  - Row 19: Wholesale Trade, 15417, 15418, 15419, 15420, 15421, 15422, 15423, 15424, 15425, 15426
  - Row 20: Retail Trade, 15427, 15428, 15429, 15430, 15431, 15432, 15433, 15434, 15435, 15436

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Accommodation and Food Services | 15437 | 15438
  - Row 22: Transport, Postal and Warehousing | 15447 | 15448
  - Row 23: Information Media and Telecommunications | 15457 | 15458
  - Row 24: Financial and Insurance Services | 15467 | 15468
  - Row 25: Rental, Hiring and Real Estate Services | 15477 | 15478
  - Row 26: Professional, Scientific and Technical Services | 15487 | 15488
  - Row 27: Administrative and Support Services | 15497 | 15498
  - Row 28: Public Administration and Safety | 15507 | 15508
  - Row 29: Education and Training | 15517 | 15518
  - Row 30: Health Care and Social Assistance | 15527 | 15528
  - Row 31: Arts and Recreation Services | 15537 | 15538
  - Row 32: Other Services | 15547 | 15548
  - Row 33: Inadequately described/Not stated | 15557 | 15558
  - Row 35: Total | 15567 | 15568
  - Row 37: This table is based on place of usual residence.
  - Row 39: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - Arts and Recreation Services
  - Inadequately described/Not stated
  - This table is based on place of usual residence.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G57: G57: G57 TOTAL FAMILY INCOME (WEEKLY) BY LABOUR FORCE STATUS OF PARTNERS FOR COUPLE FAMILIES WITH NO CHILDREN

**Potential Headers (from Template Rows ~5-20):**
  - Row 11: Both employed, One employed full-time(a), One employed part-time(b), from work(c),, Labour
  - Row 12: Worked, Worked, Both away, Other, Other away, Other not, Other away, Other not, other not, Both not, force status
  - Row 13: full-time(a), part-time(b), from work(c), part-time(b), from work(c), working(d), from work(c), working(d), working(d), working(d), not stated(e), Total
  - Row 15: Negative/Nil income, 15577, 15578, 15579, 15580, 15581, 15582, 15583, 15584, 15585, 15586, 15587, 15588
  - Row 16: $1-$149, 15589, 15590, 15591, 15592, 15593, 15594, 15595, 15596, 15597, 15598, 15599, 15600
  - Row 17: $150-$299, 15601, 15602, 15603, 15604, 15605, 15606, 15607, 15608, 15609, 15610, 15611, 15612
  - Row 18: $300-$399, 15613, 15614, 15615, 15616, 15617, 15618, 15619, 15620, 15621, 15622, 15623, 15624
  - Row 19: $400-$499, 15625, 15626, 15627, 15628, 15629, 15630, 15631, 15632, 15633, 15634, 15635, 15636
  - Row 20: $500-$649, 15637, 15638, 15639, 15640, 15641, 15642, 15643, 15644, 15645, 15646, 15647, 15648

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: $650-$799 | 15649 | 15650
  - Row 22: $800-$999 | 15661 | 15662
  - Row 23: $1,000-$1,249 | 15673 | 15674
  - Row 24: $1,250-$1,499 | 15685 | 15686
  - Row 25: $1,500-$1,749 | 15697 | 15698
  - Row 26: $1,750-$1,999 | 15709 | 15710
  - Row 27: $2,000-$2,499 | 15721 | 15722
  - Row 28: $2,500-$2,999 | 15733 | 15734
  - Row 29: $3,000-$3,499 | 15745 | 15746
  - Row 30: $3,500-$3,999 | 15757 | 15758
  - Row 31: $4,000 or more | 15769 | 15770
  - Row 32: Partial income stated(f) | 15781 | 15782
  - Row 33: All incomes not stated(g) | 15793 | 15794
  - Row 35: Total | 15805 | 15806
  - Row 37: This table is based on place of enumeration.
  - Row 38: (a) 'Employed, worked full-time' is defined as having worked 35 hours or more in all jobs during the week prior to Census Night.
  - Row 39: (b) 'Employed, worked part-time' is defined as having worked less than 35 hours in all jobs during the week prior to Census Night.
  - Row 40: (c) Comprises employed persons who did not work any hours in the week prior to Census Night and who did not state their number of hours worked.
  - Row 41: (d) Comprises people who are unemployed and people not in the labour force.
  - Row 42: (e) Includes families where one or both partners did not state their labour force status or a partner was temporarily absent on Census Night.
  - Row 43: (f) Includes families where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent.
  - Row 44: (g) Includes families where no members present stated an income.
  - Row 46: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - (a) 'Employed, worked full-time' is defined as having worked 35 hours or more in all jobs during the week prior to Census Night.
  - (b) 'Employed, worked part-time' is defined as having worked less than 35 hours in all jobs during the week prior to Census Night.
  - (c) Comprises employed persons who did not work any hours in the week prior to Census Night and who did not state their number of hours worked.
  - (d) Comprises people who are unemployed and people not in the labour force.
  - (e) Includes families where one or both partners did not state their labour force status or a partner was temporarily absent on Census Night.
  - (f) Includes families where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent.
  - (g) Includes families where no members present stated an income.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G58: G58: G58 TOTAL FAMILY INCOME (WEEKLY) BY LABOUR FORCE STATUS OF PARENTS/PARTNERS FOR COUPLE FAMILIES WITH CHILDREN

**Potential Headers (from Template Rows ~5-20):**
  - Row 11: Both employed, One employed full-time(a), One employed part-time(b), from work(c),, Labour
  - Row 12: Worked, Worked, Both away, Other, Other away, Other not, Other away, Other not, other not, Both not, force status
  - Row 13: full-time(a), part-time(b), from work(c), part-time(b), from work(c), working(d), from work(c), working(d), working(d), working(d), not stated(e), Total
  - Row 15: Negative/Nil income, 15817, 15818, 15819, 15820, 15821, 15822, 15823, 15824, 15825, 15826, 15827, 15828
  - Row 16: $1-$149, 15829, 15830, 15831, 15832, 15833, 15834, 15835, 15836, 15837, 15838, 15839, 15840
  - Row 17: $150-$299, 15841, 15842, 15843, 15844, 15845, 15846, 15847, 15848, 15849, 15850, 15851, 15852
  - Row 18: $300-$399, 15853, 15854, 15855, 15856, 15857, 15858, 15859, 15860, 15861, 15862, 15863, 15864
  - Row 19: $400-$499, 15865, 15866, 15867, 15868, 15869, 15870, 15871, 15872, 15873, 15874, 15875, 15876
  - Row 20: $500-$649, 15877, 15878, 15879, 15880, 15881, 15882, 15883, 15884, 15885, 15886, 15887, 15888

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: $650-$799 | 15889 | 15890
  - Row 22: $800-$999 | 15901 | 15902
  - Row 23: $1,000-$1,249 | 15913 | 15914
  - Row 24: $1,250-$1,499 | 15925 | 15926
  - Row 25: $1,500-$1,749 | 15937 | 15938
  - Row 26: $1,750-$1,999 | 15949 | 15950
  - Row 27: $2,000-$2,499 | 15961 | 15962
  - Row 28: $2,500-$2,999 | 15973 | 15974
  - Row 29: $3,000-$3,499 | 15985 | 15986
  - Row 30: $3,500-$3,999 | 15997 | 15998
  - Row 31: $4,000 or more | 16009 | 16010
  - Row 32: Partial income stated(f) | 16021 | 16022
  - Row 33: All incomes not stated(g) | 16033 | 16034
  - Row 35: Total | 16045 | 16046
  - Row 37: This table is based on place of enumeration.
  - Row 38: (a) 'Employed, worked full-time' is defined as having worked 35 hours or more in all jobs during the week prior to Census Night.
  - Row 39: (b) 'Employed, worked part-time' is defined as having worked less than 35 hours in all jobs during the week prior to Census Night.
  - Row 40: (c) Comprises employed persons who did not work any hours in the week prior to Census Night and who did not state their number of hours worked.
  - Row 41: (d) Comprises people who are unemployed and people not in the labour force.
  - Row 42: (e) Includes families where one or both partners did not state their labour force status or a partner was temporarily absent on Census Night.
  - Row 43: (f) Includes families where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent.
  - Row 44: (g) Includes families where no members present stated an income.
  - Row 46: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - (a) 'Employed, worked full-time' is defined as having worked 35 hours or more in all jobs during the week prior to Census Night.
  - (b) 'Employed, worked part-time' is defined as having worked less than 35 hours in all jobs during the week prior to Census Night.
  - (c) Comprises employed persons who did not work any hours in the week prior to Census Night and who did not state their number of hours worked.
  - (d) Comprises people who are unemployed and people not in the labour force.
  - (e) Includes families where one or both partners did not state their labour force status or a partner was temporarily absent on Census Night.
  - (f) Includes families where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent.
  - (g) Includes families where no members present stated an income.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G59: G59: G59 TOTAL FAMILY INCOME (WEEKLY) BY LABOUR FORCE STATUS OF PARENT FOR ONE PARENT FAMILIES

**Potential Headers (from Template Rows ~5-20):**
  - Row 10: Employed, Unemployed, looking for, Total, Labour
  - Row 11: Worked, Worked, Away, Full-time, Part-time, labour, Not in the, force status
  - Row 12: full-time(a), part-time(b), from work(c), Total, work, work, Total, force, labour force, not stated, Total
  - Row 14: Negative/Nil income, 16057, 16058, 16059, 16060, 16061, 16062, 16063, 16064, 16065, 16066, 16067
  - Row 15: $1-$149, 16068, 16069, 16070, 16071, 16072, 16073, 16074, 16075, 16076, 16077, 16078
  - Row 16: $150-$299, 16079, 16080, 16081, 16082, 16083, 16084, 16085, 16086, 16087, 16088, 16089
  - Row 17: $300-$399, 16090, 16091, 16092, 16093, 16094, 16095, 16096, 16097, 16098, 16099, 16100
  - Row 18: $400-$499, 16101, 16102, 16103, 16104, 16105, 16106, 16107, 16108, 16109, 16110, 16111
  - Row 19: $500-$649, 16112, 16113, 16114, 16115, 16116, 16117, 16118, 16119, 16120, 16121, 16122
  - Row 20: $650-$799, 16123, 16124, 16125, 16126, 16127, 16128, 16129, 16130, 16131, 16132, 16133

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: $800-$999 | 16134 | 16135
  - Row 22: $1,000-$1,249 | 16145 | 16146
  - Row 23: $1,250-$1,499 | 16156 | 16157
  - Row 24: $1,500-$1,749 | 16167 | 16168
  - Row 25: $1,750-$1,999 | 16178 | 16179
  - Row 26: $2,000-$2,499 | 16189 | 16190
  - Row 27: $2,500-$2,999 | 16200 | 16201
  - Row 28: $3,000-$3,499 | 16211 | 16212
  - Row 29: $3,500-$3,999 | 16222 | 16223
  - Row 30: $4,000 or more | 16233 | 16234
  - Row 31: Partial income stated(d) | 16244 | 16245
  - Row 32: All incomes not stated(e) | 16255 | 16256
  - Row 34: Total | 16266 | 16267
  - Row 36: This table is based on place of enumeration.
  - Row 37: (a) 'Employed, worked full-time' is defined as having worked 35 hours or more in all jobs during the week prior to Census Night.
  - Row 38: (b) 'Employed, worked part-time' is defined as having worked less than 35 hours in all jobs during the week prior to Census Night.
  - Row 39: (c) Comprises employed persons who did not work any hours in the week prior to Census Night and who did not state their number of hours worked.
  - Row 40: (d) Includes families where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent.
  - Row 41: (e) Includes families where no members present stated an income.
  - Row 43: Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

**Notes (from bottom of template sheet):**
  - This table is based on place of enumeration.
  - (a) 'Employed, worked full-time' is defined as having worked 35 hours or more in all jobs during the week prior to Census Night.
  - (b) 'Employed, worked part-time' is defined as having worked less than 35 hours in all jobs during the week prior to Census Night.
  - (c) Comprises employed persons who did not work any hours in the week prior to Census Night and who did not state their number of hours worked.
  - (d) Includes families where at least one, but not all, member(s) aged 15 years and over did not state an income and/or was temporarily absent.
  - (e) Includes families where no members present stated an income.
  - Please note that there are small random adjustments made to all cell values to protect the confidentiality of data. These adjustments may cause the sum of rows or columns to differ by small amounts from table totals.

---

## G60: G60: G60 OCCUPATION BY AGE BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 10: Technicians, and personal, Clerical and, Machinery, Inadequately
  - Row 11: and trades, service, administrative, Sales, operators, described/
  - Row 12: Managers, Professionals, workers, workers, workers, workers, and drivers, Labourers, Not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 13: MALES
  - Row 15: 15-19 years | 16277 | 16278
  - Row 16: 20-24 years | 16287 | 16288
  - Row 17: 25-34 years | 16297 | 16298
  - Row 18: 35-44 years | 16307 | 16308
  - Row 19: 45-54 years | 16317 | 16318
  - Row 20: 55-64 years | 16327 | 16328
  - Row 21: 65-74 years | 16337 | 16338
  - Row 22: 75-84 years | 16347 | 16348
  - Row 23: 85 years and over | 16357 | 16358
  - Row 25: Total | 16367 | 16368
  - Row 27: FEMALES
  - Row 29: 15-19 years | 16377 | 16378
  - Row 30: 20-24 years | 16387 | 16388
  - Row 31: 25-34 years | 16397 | 16398
  - Row 32: 35-44 years | 16407 | 16408
  - Row 33: 45-54 years | 16417 | 16418
  - Row 34: 55-64 years | 16427 | 16428
  - Row 35: 65-74 years | 16437 | 16438
  - Row 36: 75-84 years | 16447 | 16448
  - Row 37: 85 years and over | 16457 | 16458
  - Row 39: Total | 16467 | 16468
  - Row 41: PERSONS

**Notes:** (None detected)

---

## G61: G61: G61 OCCUPATION BY HOURS WORKED BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 9: 1-19, 20-29, 30-34, 35-39, 40-44, 45-49, 50 hours
  - Row 10: None(a), hours, hours, hours, hours, hours, hours, and over, Not stated, Total

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 11: MALES
  - Row 13: Managers | 16577 | 16578
  - Row 14: Professionals | 16587 | 16588
  - Row 15: Technicians and trades workers | 16597 | 16598
  - Row 16: Community and personal service workers | 16607 | 16608
  - Row 17: Clerical and administrative workers | 16617 | 16618
  - Row 18: Sales workers | 16627 | 16628
  - Row 19: Machinery operators and drivers | 16637 | 16638
  - Row 20: Labourers | 16647 | 16648
  - Row 21: Inadequately described/Not stated | 16657 | 16658
  - Row 23: Total | 16667 | 16668
  - Row 25: FEMALES
  - Row 27: Managers | 16677 | 16678
  - Row 28: Professionals | 16687 | 16688
  - Row 29: Technicians and trades workers | 16697 | 16698
  - Row 30: Community and personal service workers | 16707 | 16708
  - Row 31: Clerical and administrative workers | 16717 | 16718
  - Row 32: Sales workers | 16727 | 16728
  - Row 33: Machinery operators and drivers | 16737 | 16738
  - Row 34: Labourers | 16747 | 16748
  - Row 35: Inadequately described/Not stated | 16757 | 16758
  - Row 37: Total | 16767 | 16768
  - Row 39: PERSONS

**Notes (from bottom of template sheet):**
  - Technicians and trades workers
  - Community and personal service workers
  - Clerical and administrative workers
  - Machinery operators and drivers
  - Inadequately described/Not stated

---

## G62: G62: G62 METHOD OF TRAVEL TO WORK(a) BY SEX

**Potential Headers (from Template Rows ~5-20):**
  - Row 8: Males, Females, Persons
  - Row 11: Train, 16877, 16878, 16879
  - Row 12: Bus, 16880, 16881, 16882
  - Row 13: Ferry, 16883, 16884, 16885
  - Row 14: Tram/light rail, 16886, 16887, 16888
  - Row 15: Taxi/ride-share service, 16889, 16890, 16891
  - Row 16: Car, as driver, 16892, 16893, 16894
  - Row 17: Car, as passenger, 16895, 16896, 16897
  - Row 18: Truck, 16898, 16899, 16900
  - Row 19: Motorbike/scooter, 16901, 16902, 16903
  - Row 20: Bicycle, 16904, 16905, 16906

**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**
  - Row 21: Other | 16907 | 16908
  - Row 22: Walked only(b) | 16910 | 16911
  - Row 23: Total one method | 16913 | 16914
  - Row 25: Two methods:
  - Row 26: Train and:
  - Row 27: Bus | 16916 | 16917
  - Row 28: Ferry | 16919 | 16920
  - Row 29: Tram/light rail | 16922 | 16923
  - Row 30: Car, as driver | 16925 | 16926
  - Row 31: Car, as passenger | 16928 | 16929
  - Row 32: Other(c) | 16931 | 16932
  - Row 33: Total | 16934 | 16935
  - Row 34: Bus and:
  - Row 35: Ferry | 16937 | 16938
  - Row 36: Tram/light rail | 16940 | 16941
  - Row 37: Car, as driver | 16943 | 16944
  - Row 38: Car, as passenger | 16946 | 16947
  - Row 39: Other(c) | 16949 | 16950
  - Row 40: Total | 16952 | 16953
  - Row 41: Other two methods | 16955 | 16956
  - Row 42: Total two methods | 16958 | 16959
  - Row 44: Three methods:
  - Row 45: Train and two other methods | 16961 | 16962
  - Row 46: Bus and two other methods (excludes train) | 16964 | 16965
  - Row 47: Other three methods | 16967 | 16968
  - Row 48: Total three methods | 16970 | 16971
  - Row 50: Worked at home | 16973 | 16974

**Notes (from bottom of template sheet):**
  - Train and two other methods
  - Bus and two other methods (excludes train)

---

