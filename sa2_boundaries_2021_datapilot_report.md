# DataPilot Analysis Report

Analysis Target: sa2_boundaries_2021.parquet
Report Generated: 2025-06-18 23:55:02 (UTC)
DataPilot Version: v1.0.0 (TypeScript Edition)

---

## Section 1: Overview
This section provides a detailed snapshot of the dataset properties, how it was processed, and the context of this analysis run.

**1.1. Input Data File Details:**
    * Original Filename: `sa2_boundaries_2021.parquet`
    * Full Resolved Path: `/Users/[user]/AHGD/data/processed/sa2_boundaries_2021.parquet`
    * File Size (on disk): 65.80 MB
    * MIME Type (detected/inferred): `application/octet-stream`
    * File Last Modified (OS Timestamp): 2025-06-17 10:52:21 (UTC)
    * File Hash (SHA256): `990083972e975ee165be9f7039b8f7c7d52ef6f10ec642427e1b9cedf57f202f`
    **1.5. Compression & Storage Efficiency:**
    * Current File Size: 65.80 MB
    * Estimated Compressed Size (gzip): 49.37 MB (25% reduction)
    * Estimated Compressed Size (parquet): 32.90 MB (50% reduction)
    * Column Entropy Analysis:
        * High Entropy (poor compression): PAR1����L�&   ��0	   101021007.  8.  910.  2 6. 61131013.  4.  5.  64101���4102�u����.�  2�5153u5154u6154uuuu201102�	[ 3[	[	[	[	.[  3�	�	�	.� ��	.� �	�210�����521055�5.�  55.�  5.�  5� 3!� 5u 6u	u 6Q�	u 3h 6�	2 �	�	�	�2106-	�3107������4107���� 4!+ 8�	�	� 4� 8�	�	�	�	�	�	�2109�� 5� 9�	�	�	�	� 5N�	� 5!8 9�10�	� 3aЉ�	����60111�|	�	��b	�	�	�60211�o	u216.  6��' 66' -	�311�� 6��	�311��	�-	-60411��	��- 7!E�.	-	u	u 5��	-	u 7!E�U 7��U	'311�U	�311�U�U	�'�411�U 7-	�'�-'�H 5.  5�o 8!8�b	�	�	-8 8!�b	�	�	�	-�o 8��o 8ñU	u411�H	u416Q� 6Q�80511�b	��511�b'�b 9!�b	-	�	-�	� 9!+�b	�	� 9!�b	m@	-�	�311�b		�10��b)�	�	�10��b	�	�	�)10��b)�311�b10!ԭb 2�b�4120���-10112�b�	-	-�b�	�	�	�F 2�H-	�212�H�212M�	�H110312�H��5��5�312�
    * Analysis Method: Sample-based analysis (1024KB sample)
    **1.6. File Health Check:**
    * Overall Health Score: ⚠️ 70/100
    * ✅ Byte Order Mark (BOM): Not detected
    * ⚠️ Line endings: mixed
    * ❌ Null bytes: Detected
    * ✅ Valid UTF-8 encoding: Throughout
    * ℹ️ File size: Normal size
    * Recommendations:
        * Standardise line endings for consistent processing
        * File contains null bytes - verify data integrity

**1.2. Data Ingestion & Parsing Parameters:**
    * Data Source Type: Local File System
    * Parsing Engine Utilized: DataPilot Advanced CSV Parser v1.0.0
    * Time Taken for Parsing & Initial Load: 3.114 seconds
    * Detected Character Encoding: `utf8`
        * Encoding Detection Method: Statistical Character Pattern Analysis
        * Encoding Confidence: Low (50%)
    * Detected Delimiter Character: `;` (Semicolon)
        * Delimiter Detection Method: Character Frequency Analysis with Field Consistency Scoring
        * Delimiter Confidence: Low (45%)
    * Detected Line Ending Format: `LF (Unix-style)`
    * Detected Quoting Character: `"`
        * Empty Lines Encountered: 4
    * Header Row Processing:
        * Header Presence: Not Detected
        * Header Row Number(s): N/A
        * Column Names Derived From: Generated column indices (Col_0, Col_1, etc.)
    * Byte Order Mark (BOM): Not Detected
    * Initial Row/Line Scan Limit for Detection: First 1048576 bytes or 1000 lines

**1.3. Dataset Structural Dimensions & Initial Profile:**
    * Total Rows Read (including header, if any): 212,574
    * Total Rows of Data (excluding header): 212,574
    * Total Columns Detected: 160
    * Total Data Cells (Data Rows × Columns): 34,011,840
    * List of Column Names (160) and Original Index:
        1.  (Index 0) `Col_0`
        2.  (Index 1) `Col_1`
        3.  (Index 2) `Col_2`
        4.  (Index 3) `Col_3`
        5.  (Index 4) `Col_4`
        6.  (Index 5) `Col_5`
        7.  (Index 6) `Col_6`
        8.  (Index 7) `Col_7`
        9.  (Index 8) `Col_8`
        10.  (Index 9) `Col_9`
        11.  (Index 10) `Col_10`
        12.  (Index 11) `Col_11`
        13.  (Index 12) `Col_12`
        14.  (Index 13) `Col_13`
        15.  (Index 14) `Col_14`
        16.  (Index 15) `Col_15`
        17.  (Index 16) `Col_16`
        18.  (Index 17) `Col_17`
        19.  (Index 18) `Col_18`
        20.  (Index 19) `Col_19`
        21.  (Index 20) `Col_20`
        22.  (Index 21) `Col_21`
        23.  (Index 22) `Col_22`
        24.  (Index 23) `Col_23`
        25.  (Index 24) `Col_24`
        26.  (Index 25) `Col_25`
        27.  (Index 26) `Col_26`
        28.  (Index 27) `Col_27`
        29.  (Index 28) `Col_28`
        30.  (Index 29) `Col_29`
        31.  (Index 30) `Col_30`
        32.  (Index 31) `Col_31`
        33.  (Index 32) `Col_32`
        34.  (Index 33) `Col_33`
        35.  (Index 34) `Col_34`
        36.  (Index 35) `Col_35`
        37.  (Index 36) `Col_36`
        38.  (Index 37) `Col_37`
        39.  (Index 38) `Col_38`
        40.  (Index 39) `Col_39`
        41.  (Index 40) `Col_40`
        42.  (Index 41) `Col_41`
        43.  (Index 42) `Col_42`
        44.  (Index 43) `Col_43`
        45.  (Index 44) `Col_44`
        46.  (Index 45) `Col_45`
        47.  (Index 46) `Col_46`
        48.  (Index 47) `Col_47`
        49.  (Index 48) `Col_48`
        50.  (Index 49) `Col_49`
        51.  (Index 50) `Col_50`
        52.  (Index 51) `Col_51`
        53.  (Index 52) `Col_52`
        54.  (Index 53) `Col_53`
        55.  (Index 54) `Col_54`
        56.  (Index 55) `Col_55`
        57.  (Index 56) `Col_56`
        58.  (Index 57) `Col_57`
        59.  (Index 58) `Col_58`
        60.  (Index 59) `Col_59`
        61.  (Index 60) `Col_60`
        62.  (Index 61) `Col_61`
        63.  (Index 62) `Col_62`
        64.  (Index 63) `Col_63`
        65.  (Index 64) `Col_64`
        66.  (Index 65) `Col_65`
        67.  (Index 66) `Col_66`
        68.  (Index 67) `Col_67`
        69.  (Index 68) `Col_68`
        70.  (Index 69) `Col_69`
        71.  (Index 70) `Col_70`
        72.  (Index 71) `Col_71`
        73.  (Index 72) `Col_72`
        74.  (Index 73) `Col_73`
        75.  (Index 74) `Col_74`
        76.  (Index 75) `Col_75`
        77.  (Index 76) `Col_76`
        78.  (Index 77) `Col_77`
        79.  (Index 78) `Col_78`
        80.  (Index 79) `Col_79`
        81.  (Index 80) `Col_80`
        82.  (Index 81) `Col_81`
        83.  (Index 82) `Col_82`
        84.  (Index 83) `Col_83`
        85.  (Index 84) `Col_84`
        86.  (Index 85) `Col_85`
        87.  (Index 86) `Col_86`
        88.  (Index 87) `Col_87`
        89.  (Index 88) `Col_88`
        90.  (Index 89) `Col_89`
        91.  (Index 90) `Col_90`
        92.  (Index 91) `Col_91`
        93.  (Index 92) `Col_92`
        94.  (Index 93) `Col_93`
        95.  (Index 94) `Col_94`
        96.  (Index 95) `Col_95`
        97.  (Index 96) `Col_96`
        98.  (Index 97) `Col_97`
        99.  (Index 98) `Col_98`
        100.  (Index 99) `Col_99`
        101.  (Index 100) `Col_100`
        102.  (Index 101) `Col_101`
        103.  (Index 102) `Col_102`
        104.  (Index 103) `Col_103`
        105.  (Index 104) `Col_104`
        106.  (Index 105) `Col_105`
        107.  (Index 106) `Col_106`
        108.  (Index 107) `Col_107`
        109.  (Index 108) `Col_108`
        110.  (Index 109) `Col_109`
        111.  (Index 110) `Col_110`
        112.  (Index 111) `Col_111`
        113.  (Index 112) `Col_112`
        114.  (Index 113) `Col_113`
        115.  (Index 114) `Col_114`
        116.  (Index 115) `Col_115`
        117.  (Index 116) `Col_116`
        118.  (Index 117) `Col_117`
        119.  (Index 118) `Col_118`
        120.  (Index 119) `Col_119`
        121.  (Index 120) `Col_120`
        122.  (Index 121) `Col_121`
        123.  (Index 122) `Col_122`
        124.  (Index 123) `Col_123`
        125.  (Index 124) `Col_124`
        126.  (Index 125) `Col_125`
        127.  (Index 126) `Col_126`
        128.  (Index 127) `Col_127`
        129.  (Index 128) `Col_128`
        130.  (Index 129) `Col_129`
        131.  (Index 130) `Col_130`
        132.  (Index 131) `Col_131`
        133.  (Index 132) `Col_132`
        134.  (Index 133) `Col_133`
        135.  (Index 134) `Col_134`
        136.  (Index 135) `Col_135`
        137.  (Index 136) `Col_136`
        138.  (Index 137) `Col_137`
        139.  (Index 138) `Col_138`
        140.  (Index 139) `Col_139`
        141.  (Index 140) `Col_140`
        142.  (Index 141) `Col_141`
        143.  (Index 142) `Col_142`
        144.  (Index 143) `Col_143`
        145.  (Index 144) `Col_144`
        146.  (Index 145) `Col_145`
        147.  (Index 146) `Col_146`
        148.  (Index 147) `Col_147`
        149.  (Index 148) `Col_148`
        150.  (Index 149) `Col_149`
        151.  (Index 150) `Col_150`
        152.  (Index 151) `Col_151`
        153.  (Index 152) `Col_152`
        154.  (Index 153) `Col_153`
        155.  (Index 154) `Col_154`
        156.  (Index 155) `Col_155`
        157.  (Index 156) `Col_156`
        158.  (Index 157) `Col_157`
        159.  (Index 158) `Col_158`
        160.  (Index 159) `Col_159`
    * Estimated In-Memory Size (Post-Parsing & Initial Type Guessing): 190.4 MB
    * Average Row Length (bytes, approximate): 293 bytes
    * Dataset Sparsity (Initial Estimate): Dense dataset with minimal missing values (0.71% sparse cells via Statistical sampling of 10000 rows)
    **1.7. Quick Column Statistics:**
    * Numeric Columns: 0 (0.0%)
    * Text Columns: 160 (100.0%)
    * Columns with High Cardinality (>50% unique): 160
    * Columns with Low Cardinality (<10% unique): 0
    * Analysis Method: Sample-based analysis (1000 rows)

**1.4. Analysis Configuration & Execution Context:**
    * Full Command Executed: `datapilot overview /Users/massimoraso/AHGD/data/processed/sa2_boundaries_2021.parquet`
    * Analysis Mode Invoked: Comprehensive Deep Scan
    * Timestamp of Analysis Start: 2025-06-18 23:54:56 (UTC)
    * Global Dataset Sampling Strategy: Full dataset analysis (No record sampling applied for initial overview)
    * DataPilot Modules Activated for this Run: File I/O Manager, Advanced CSV Parser, Metadata Collector, Structural Analyzer, Report Generator
    * Processing Time for Section 1 Generation: 6.21 seconds
    * Host Environment Details:
        * Operating System: macOS (Unknown Version)
        * System Architecture: ARM64 (Apple Silicon/ARM 64-bit)
        * Execution Runtime: Node.js v23.6.1 (V8 12.9.202.28-node.12) on darwin
        * Available CPU Cores / Memory (at start of analysis): 8 cores / 8 GB

**1.8. Data Sample:**
    | PAR1���... |
    |---|
    | f��b@-�P�@... | �!� | ��I��b@[<� | ��b��b@���z... | �b����b@A�... | ������b@����... |
    | ���b@u�B� |
    | 8a@1�ՊC@���� |
    | c@�[�7 �8���<L� |
    |  | �;�z�@����� 4 | �5���@��1X� | �ЄH�@�\u#N | w�8+�@�i�� | �;^j�@��X�A |
    | ... | ... | ... | ... | ... | ... |

    * Note: Showing 5 of 424,027 rows
    * Preview Method: sample
    * Generation Time: 2856ms

---
### Analysis Warnings

**Structural Warnings:**
    * ⚠️ Wide dataset detected (160 columns) (Suggestion: Consider feature selection or dimensionality reduction)

---
### Performance Metrics

**Processing Performance:**
    * Total Analysis Time: 6.21 seconds
    * File analysis: 0.169s
    * Parsing: 3.128s
    * Structural analysis: 0.054s
    * Data preview: 2.859s