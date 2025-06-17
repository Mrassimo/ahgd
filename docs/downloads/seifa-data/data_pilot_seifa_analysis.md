# DataPilot Analysis Report

Analysis Target: seifa_2021_sa2_complete.csv
Report Generated: 2025-06-17 09:33:35 (UTC)
DataPilot Version: v1.0.0 (TypeScript Edition)

---

## Section 1: Overview
This section provides a detailed snapshot of the dataset properties, how it was processed, and the context of this analysis run.

**1.1. Input Data File Details:**
    * Original Filename: `seifa_2021_sa2_complete.csv`
    * Full Resolved Path: `/Users/[user]/AHGD/australian-health-analytics/docs/downloads/seifa-data/seifa_2021_sa2_complete.csv`
    * File Size (on disk): 0.127533 MB
    * MIME Type (detected/inferred): `text/csv`
    * File Last Modified (OS Timestamp): 2025-06-17 09:26:57 (UTC)
    * File Hash (SHA256): `18a69af98bc80cb12f662fe0ba04a60ba697c689e2cb5c3cd5bf2995336311ce`

**1.2. Data Ingestion & Parsing Parameters:**
    * Data Source Type: Local File System
    * Parsing Engine Utilized: DataPilot Advanced CSV Parser v1.0.0
    * Time Taken for Parsing & Initial Load: 0.023 seconds
    * Detected Character Encoding: `utf8`
        * Encoding Detection Method: Statistical Character Pattern Analysis
        * Encoding Confidence: High (95%)
    * Detected Delimiter Character: `,` (Comma)
        * Delimiter Detection Method: Character Frequency Analysis with Field Consistency Scoring
        * Delimiter Confidence: High (100%)
    * Detected Line Ending Format: `LF (Unix-style)`
    * Detected Quoting Character: `"`
        * Empty Lines Encountered: 1
    * Header Row Processing:
        * Header Presence: Detected
        * Header Row Number(s): 1
        * Column Names Derived From: First row interpreted as column headers
    * Byte Order Mark (BOM): Not Detected
    * Initial Row/Line Scan Limit for Detection: First 133728 bytes or 1000 lines

**1.3. Dataset Structural Dimensions & Initial Profile:**
    * Total Rows Read (including header, if any): 2,294
    * Total Rows of Data (excluding header): 2,293
    * Total Columns Detected: 11
    * Total Data Cells (Data Rows × Columns): 25,223
    * List of Column Names (11) and Original Index:
        1.  (Index 0) `sa2_code_2021`
        2.  (Index 1) `sa2_name_2021`
        3.  (Index 2) `irsd_score`
        4.  (Index 3) `irsd_decile`
        5.  (Index 4) `irsad_score`
        6.  (Index 5) `irsad_decile`
        7.  (Index 6) `ier_score`
        8.  (Index 7) `ier_decile`
        9.  (Index 8) `ieo_score`
        10.  (Index 9) `ieo_decile`
        11.  (Index 10) `usual_resident_population`
    * Estimated In-Memory Size (Post-Parsing & Initial Type Guessing): 1.08 MB
    * Average Row Length (bytes, approximate): 59 bytes
    * Dataset Sparsity (Initial Estimate): Dense dataset with minimal missing values (0.3% sparse cells via Full dataset analysis)

**1.4. Analysis Configuration & Execution Context:**
    * Full Command Executed: `datapilot overview /Users/massimoraso/AHGD/australian-health-analytics/docs/downloads/seifa-data/seifa_2021_sa2_complete.csv`
    * Analysis Mode Invoked: Comprehensive Deep Scan
    * Timestamp of Analysis Start: 2025-06-17 09:33:35 (UTC)
    * Global Dataset Sampling Strategy: Full dataset analysis (No record sampling applied for initial overview)
    * DataPilot Modules Activated for this Run: File I/O Manager, Advanced CSV Parser, Metadata Collector, Structural Analyzer, Report Generator
    * Processing Time for Section 1 Generation: 0.032 seconds
    * Host Environment Details:
        * Operating System: macOS (Unknown Version)
        * System Architecture: ARM64 (Apple Silicon/ARM 64-bit)
        * Execution Runtime: Node.js v23.6.1 (V8 12.9.202.28-node.12) on darwin
        * Available CPU Cores / Memory (at start of analysis): 8 cores / 8 GB

---
### Performance Metrics

**Processing Performance:**
    * Total Analysis Time: 0.033 seconds
    * File analysis: 0.005s
    * Parsing: 0.024s
    * Structural analysis: 0.003s