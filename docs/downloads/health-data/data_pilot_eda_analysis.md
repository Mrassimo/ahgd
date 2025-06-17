### **Section 3: Exploratory Data Analysis (EDA) Deep Dive** 📊🔬

This section provides a comprehensive statistical exploration of the dataset. The goal is to understand the data's underlying structure, identify patterns, detect anomalies, and extract key insights. Unless specified, all analyses are performed on the full dataset. Over 60 statistical tests and checks are considered in this module.

**3.1. EDA Methodology Overview:**
* **Approach:** Systematic univariate, bivariate, and multivariate analysis using streaming algorithms.
* **Column Type Classification:** Each column is analysed based on its inferred data type (Numerical, Categorical, Date/Time, Boolean, Text).
* **Statistical Significance:** Standard p-value thresholds (e.g., 0.05) are used where applicable, but effect sizes and practical significance are also considered.
* **Memory-Efficient Processing:** Streaming with online algorithms ensures scalability to large datasets.
* **Sampling Strategy:** Analysis performed on the complete dataset.

**3.2. Univariate Analysis (Per-Column In-Depth Profile):**

*This sub-section provides detailed statistical profiles for each column in the dataset, adapted based on detected data type.*

---
**Column: `sa2_code_2021`**
* **Detected Data Type:** numerical_integer
* **Inferred Semantic Type:** unknown
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 100
    * Missing Values: 0 (0%)
    * Unique Values: 100 (100% of total)

**3.2.A. Numerical Column Analysis:**

**Descriptive Statistics:**
* Minimum: 100001000
* Maximum: 100001099
* Range: 99
* Sum: 10000104950
* Mean (Arithmetic): 100001049.5
* Median (50th Percentile): 100001049
* Mode(s): [100001000 (Frequency: 1%), 100001001 (Frequency: 1%), 100001002 (Frequency: 1%), 100001003 (Frequency: 1%), 100001004 (Frequency: 1%)]
* Standard Deviation: 28.86607
* Variance: 833.25
* Coefficient of Variation (CV): 0%

**Quantile & Percentile Statistics:**
* 1st Percentile: 100001002
* 5th Percentile: 100001004
* 10th Percentile: 100001009
* 25th Percentile (Q1 - First Quartile): 100001024
* 75th Percentile (Q3 - Third Quartile): 100001073
* 90th Percentile: 100001087
* 95th Percentile: 100001092
* 99th Percentile: 100001096
* Interquartile Range (IQR = Q3 - Q1): 49
* Median Absolute Deviation (MAD): 25

**Distribution Shape & Normality Assessment:**
* Skewness: 0 (Approximately symmetric)
* Kurtosis (Excess): -1.2002 (Platykurtic (light tails))
* Histogram Analysis: Distribution spans 10 bins
* Normality Tests:
    * Shapiro-Wilk Test: W-statistic = 18.751875, p-value = NaN (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Jarque-Bera Test: JB-statistic = 6.0024, p-value = 0.05 (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Kolmogorov-Smirnov Test: D-statistic = 0.061037, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))

**Univariate Outlier Analysis:**
* Method 1: IQR Proximity Rule
    * Lower Fence (Q1 - 1.5 * IQR): 100000950.5
    * Upper Fence (Q3 + 1.5 * IQR): 100001146.5
    * Number of Outliers (Below Lower): 0 (0%)
    * Number of Outliers (Above Upper): 0 (0%)
    * Extreme Outliers (using 3.0 * IQR factor): 0 (0%)
* Method 2: Z-Score Method
    * Standard Deviations from Mean for Threshold: +/- 3
    * Number of Outliers (Z-score < -3): 0
    * Number of Outliers (Z-score > +3): 0
* Method 3: Modified Z-Score (using MAD)
    * Threshold: +/- 3.5
    * Number of Outliers: 0
* Summary of Outliers: Total 0 (0%). Min Outlier Value: 0, Max Outlier Value: 0.
* Potential Impact: Low outlier impact

**Specific Numerical Patterns & Characteristics:**
* Percentage of Zero Values: 0%
* Percentage of Negative Values: 0%
* Round Numbers Analysis: Moderate rounding detected
* Potential for Log Transformation: Good candidate for log transformation due to wide range

---
**Column: `sa2_name_2021`**
* **Detected Data Type:** text_general
* **Inferred Semantic Type:** category
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 100
    * Missing Values: 0 (0%)
    * Unique Values: 100 (100% of total)

**3.2.E. Text Column Analysis:**

**Length-Based Statistics (Characters):**
* Minimum Length: 11
* Maximum Length: 12
* Average Length: 11.9
* Median Length: 12
* Standard Deviation of Length: 0.3

**Word Count Statistics:**
* Minimum Word Count: 3
* Maximum Word Count: 3
* Average Word Count: 3

**Common Patterns:**
* Percentage of Empty Strings: 0%
* Percentage of Purely Numeric Text: 0%
* URLs Found: 0 (0%)
* Email Addresses Found: 0 (0%)

**Top 5 Most Frequent Words:** [test, area]

---
**Column: `seifa_risk_score`**
* **Detected Data Type:** numerical_float
* **Inferred Semantic Type:** rating
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 100
    * Missing Values: 0 (0%)
    * Unique Values: 76 (76% of total)

**3.2.A. Numerical Column Analysis:**

**Descriptive Statistics:**
* Minimum: 3.1500000000000004
* Maximum: 8.149999999999999
* Range: 4.999999999999998
* Sum: 549.9
* Mean (Arithmetic): 5.499
* Median (50th Percentile): 5.462914
* Mode(s): [5.15 (Frequency: 4%)]
* Standard Deviation: 1.134878
* Variance: 1.287949
* Coefficient of Variation (CV): 0.2064%

**Quantile & Percentile Statistics:**
* 1st Percentile: 3.307872
* 5th Percentile: 3.503671
* 10th Percentile: 3.839562
* 25th Percentile (Q1 - First Quartile): 4.779973
* 75th Percentile (Q3 - Third Quartile): 6.403662
* 90th Percentile: 7.176297
* 95th Percentile: 7.149807
* 99th Percentile: 7.511124
* Interquartile Range (IQR = Q3 - Q1): 1.623689
* Median Absolute Deviation (MAD): 0.762914

**Distribution Shape & Normality Assessment:**
* Skewness: -0.0225 (Approximately symmetric)
* Kurtosis (Excess): -0.446 (Mesokurtic (normal-like tails))
* Histogram Analysis: Distribution spans 10 bins
* Normality Tests:
    * Shapiro-Wilk Test: W-statistic = 16.074006, p-value = NaN (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Jarque-Bera Test: JB-statistic = 0.837285, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))
    * Kolmogorov-Smirnov Test: D-statistic = 0.051832, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))

**Univariate Outlier Analysis:**
* Method 1: IQR Proximity Rule
    * Lower Fence (Q1 - 1.5 * IQR): 2.34444
    * Upper Fence (Q3 + 1.5 * IQR): 8.839195
    * Number of Outliers (Below Lower): 0 (0%)
    * Number of Outliers (Above Upper): 0 (0%)
    * Extreme Outliers (using 3.0 * IQR factor): 0 (0%)
* Method 2: Z-Score Method
    * Standard Deviations from Mean for Threshold: +/- 3
    * Number of Outliers (Z-score < -3): 0
    * Number of Outliers (Z-score > +3): 0
* Method 3: Modified Z-Score (using MAD)
    * Threshold: +/- 3.5
    * Number of Outliers: 0
* Summary of Outliers: Total 0 (0%). Min Outlier Value: 0, Max Outlier Value: 0.
* Potential Impact: Low outlier impact

**Specific Numerical Patterns & Characteristics:**
* Percentage of Zero Values: 0%
* Percentage of Negative Values: 0%
* Round Numbers Analysis: No significant rounding detected
* Potential for Log Transformation: Log transformation may not be beneficial

---
**Column: `health_utilisation_risk`**
* **Detected Data Type:** numerical_float
* **Inferred Semantic Type:** unknown
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 99
    * Missing Values: 0 (0%)
    * Unique Values: 5 (5.05% of total)

**3.2.A. Numerical Column Analysis:**

**Descriptive Statistics:**
* Minimum: 0
* Maximum: 7
* Range: 7
* Sum: 266
* Mean (Arithmetic): 2.686869
* Median (50th Percentile): 2.998078
* Mode(s): [3 (Frequency: 42.42%)]
* Standard Deviation: 2.149214
* Variance: 4.61912
* Coefficient of Variation (CV): 0.7999%

**Quantile & Percentile Statistics:**
* 1st Percentile: 0
* 5th Percentile: 0.000023
* 10th Percentile: 0.000006
* 25th Percentile (Q1 - First Quartile): 0.141178
* 75th Percentile (Q3 - Third Quartile): 3.531696
* 90th Percentile: 6.180045
* 95th Percentile: 6.67619
* 99th Percentile: 6.979762
* Interquartile Range (IQR = Q3 - Q1): 3.390518
* Median Absolute Deviation (MAD): 1.001922

**Distribution Shape & Normality Assessment:**
* Skewness: 0.2377 (Approximately symmetric)
* Kurtosis (Excess): -0.723 (Platykurtic (light tails))
* Histogram Analysis: Distribution spans 10 bins
* Normality Tests:
    * Shapiro-Wilk Test: W-statistic = 13.13436, p-value = NaN (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Jarque-Bera Test: JB-statistic = 3.088696, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))
    * Kolmogorov-Smirnov Test: D-statistic = 0.234396, p-value = 0.01 (Data significantly deviates from normal distribution (p ≤ 0.05))

**Univariate Outlier Analysis:**
* Method 1: IQR Proximity Rule
    * Lower Fence (Q1 - 1.5 * IQR): -4.944599
    * Upper Fence (Q3 + 1.5 * IQR): 8.617473
    * Number of Outliers (Below Lower): 0 (0%)
    * Number of Outliers (Above Upper): 0 (0%)
    * Extreme Outliers (using 3.0 * IQR factor): 0 (0%)
* Method 2: Z-Score Method
    * Standard Deviations from Mean for Threshold: +/- 3
    * Number of Outliers (Z-score < -3): 0
    * Number of Outliers (Z-score > +3): 0
* Method 3: Modified Z-Score (using MAD)
    * Threshold: +/- 3.5
    * Number of Outliers: 0
* Summary of Outliers: Total 0 (0%). Min Outlier Value: 0, Max Outlier Value: 0.
* Potential Impact: Low outlier impact

**Specific Numerical Patterns & Characteristics:**
* Percentage of Zero Values: 31.31%
* Percentage of Negative Values: 0%
* Round Numbers Analysis: High proportion of round numbers suggests potential data rounding
* Potential for Log Transformation: Log transformation may not be beneficial

---
**Column: `total_prescriptions`**
* **Detected Data Type:** numerical_integer
* **Inferred Semantic Type:** unknown
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 98
    * Missing Values: 0 (0%)
    * Unique Values: 53 (54.08% of total)

**3.2.A. Numerical Column Analysis:**

**Descriptive Statistics:**
* Minimum: 14
* Maximum: 104
* Range: 90
* Sum: 4889
* Mean (Arithmetic): 49.887755
* Median (50th Percentile): 48.120157
* Mode(s): [52 (Frequency: 5.1%), 40 (Frequency: 5.1%)]
* Standard Deviation: 18.621297
* Variance: 346.752707
* Coefficient of Variation (CV): 0.3733%

**Quantile & Percentile Statistics:**
* 1st Percentile: 18.24758
* 5th Percentile: 23.363867
* 10th Percentile: 28.414245
* 25th Percentile (Q1 - First Quartile): 37.063779
* 75th Percentile (Q3 - Third Quartile): 57.516467
* 90th Percentile: 71.188722
* 95th Percentile: 83.108842
* 99th Percentile: 97.719094
* Interquartile Range (IQR = Q3 - Q1): 20.452688
* Median Absolute Deviation (MAD): 11.120157

**Distribution Shape & Normality Assessment:**
* Skewness: 0.7371 (Right-skewed (positive skew))
* Kurtosis (Excess): 0.5945 (Leptokurtic (heavy tails))
* Histogram Analysis: Distribution spans 10 bins
* Normality Tests:
    * Shapiro-Wilk Test: W-statistic = 14.071436, p-value = NaN (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Jarque-Bera Test: JB-statistic = 10.317193, p-value = 0.01 (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Kolmogorov-Smirnov Test: D-statistic = 0.097931, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))

**Univariate Outlier Analysis:**
* Method 1: IQR Proximity Rule
    * Lower Fence (Q1 - 1.5 * IQR): 6.384747
    * Upper Fence (Q3 + 1.5 * IQR): 88.195499
    * Number of Outliers (Below Lower): 0 (0%)
    * Number of Outliers (Above Upper): 6 (6.12%)
    * Extreme Outliers (using 3.0 * IQR factor): 0 (0%)
* Method 2: Z-Score Method
    * Standard Deviations from Mean for Threshold: +/- 3
    * Number of Outliers (Z-score < -3): 0
    * Number of Outliers (Z-score > +3): 0
* Method 3: Modified Z-Score (using MAD)
    * Threshold: +/- 3.5
    * Number of Outliers: 0
* Summary of Outliers: Total 5 (5.1%). Min Outlier Value: 92, Max Outlier Value: 104.
* Potential Impact: High outlier presence may affect analysis

**Specific Numerical Patterns & Characteristics:**
* Percentage of Zero Values: 0%
* Percentage of Negative Values: 0%
* Round Numbers Analysis: Moderate rounding detected
* Potential for Log Transformation: Log transformation may not be beneficial

---
**Column: `chronic_medication_rate`**
* **Detected Data Type:** numerical_float
* **Inferred Semantic Type:** percentage
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 98
    * Missing Values: 0 (0%)
    * Unique Values: 34 (34.69% of total)

**3.2.A. Numerical Column Analysis:**

**Descriptive Statistics:**
* Minimum: 0
* Maximum: 0.7142857142857143
* Range: 0.7142857142857143
* Sum: 27.18816
* Mean (Arithmetic): 0.27743
* Median (50th Percentile): 0.269462
* Mode(s): [0.25 (Frequency: 10.2%)]
* Standard Deviation: 0.132594
* Variance: 0.017581
* Coefficient of Variation (CV): 0.4779%

**Quantile & Percentile Statistics:**
* 1st Percentile: 0.051454
* 5th Percentile: 0.087943
* 10th Percentile: 0.126944
* 25th Percentile (Q1 - First Quartile): 0.167652
* 75th Percentile (Q3 - Third Quartile): 0.369573
* 90th Percentile: 0.460244
* 95th Percentile: 0.501005
* 99th Percentile: 0.527804
* Interquartile Range (IQR = Q3 - Q1): 0.201921
* Median Absolute Deviation (MAD): 0.102795

**Distribution Shape & Normality Assessment:**
* Skewness: 0.4446 (Approximately symmetric)
* Kurtosis (Excess): 0.0364 (Mesokurtic (normal-like tails))
* Histogram Analysis: Distribution spans 10 bins
* Normality Tests:
    * Shapiro-Wilk Test: W-statistic = 15.805815, p-value = NaN (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Jarque-Bera Test: JB-statistic = 3.234238, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))
    * Kolmogorov-Smirnov Test: D-statistic = 0.112145, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))

**Univariate Outlier Analysis:**
* Method 1: IQR Proximity Rule
    * Lower Fence (Q1 - 1.5 * IQR): -0.135229
    * Upper Fence (Q3 + 1.5 * IQR): 0.672455
    * Number of Outliers (Below Lower): 0 (0%)
    * Number of Outliers (Above Upper): 1 (1.02%)
    * Extreme Outliers (using 3.0 * IQR factor): 0 (0%)
* Method 2: Z-Score Method
    * Standard Deviations from Mean for Threshold: +/- 3
    * Number of Outliers (Z-score < -3): 0
    * Number of Outliers (Z-score > +3): 1
* Method 3: Modified Z-Score (using MAD)
    * Threshold: +/- 3.5
    * Number of Outliers: 0
* Summary of Outliers: Total 1 (1.02%). Min Outlier Value: 0.7142857142857143, Max Outlier Value: 0.7142857142857143.
* Potential Impact: Low outlier impact

**Specific Numerical Patterns & Characteristics:**
* Percentage of Zero Values: 2.04%
* Percentage of Negative Values: 0%
* Round Numbers Analysis: No significant rounding detected
* Potential for Log Transformation: Log transformation may not be beneficial

---
**Column: `state_name`**
* **Detected Data Type:** categorical
* **Inferred Semantic Type:** status
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 98
    * Missing Values: 0 (0%)
    * Unique Values: 3 (3.06% of total)

**3.2.B. Categorical Column Analysis:**

**Frequency & Proportionality:**
* Number of Unique Categories: 3
* Mode (Most Frequent Category): `VIC` (Frequency: 35, 35.71%)
* Second Most Frequent Category: `NSW` (Frequency: 33, 33.67%)
* Least Frequent Category: `QLD` (Frequency: 30, 30.61%)
* Frequency Distribution Table (Top 10):

| Category Label | Count | Percentage (%) | Cumulative % |
|----------------|-------|----------------|--------------|
| VIC | 35 | 35.71% | 35.71% |
| NSW | 33 | 33.67% | 69.38% |
| QLD | 30 | 30.61% | 99.99% |

**Diversity & Balance:**
* Shannon Entropy: 1.5821 (Range: 0 to 1.585)
* Gini Impurity: 0.6653
* Interpretation of Balance: Highly balanced distribution
* Major Category Dominance: Well distributed

**Category Label Analysis:**
* Minimum Label Length: 3 characters
* Maximum Label Length: 3 characters
* Average Label Length: 3 characters
* Empty String or Null-like Labels: 0 occurrences

**Potential Issues & Recommendations:**


* No significant issues detected.

---
**Column: `usual_resident_population`**
* **Detected Data Type:** numerical_integer
* **Inferred Semantic Type:** identifier
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 98
    * Missing Values: 0 (0%)
    * Unique Values: 97 (98.98% of total)

**3.2.A. Numerical Column Analysis:**

**Descriptive Statistics:**
* Minimum: 697
* Maximum: 4996
* Range: 4299
* Sum: 282952
* Mean (Arithmetic): 2887.265306
* Median (50th Percentile): 3127.911719
* Mode(s): [1995 (Frequency: 2.04%)]
* Standard Deviation: 1213.779202
* Variance: 1473259.950021
* Coefficient of Variation (CV): 0.4204%

**Quantile & Percentile Statistics:**
* 1st Percentile: 762.617401
* 5th Percentile: 1014.802004
* 10th Percentile: 1259.028693
* 25th Percentile (Q1 - First Quartile): 2030.45321
* 75th Percentile (Q3 - Third Quartile): 3891.98048
* 90th Percentile: 4406.982056
* 95th Percentile: 4732.059747
* 99th Percentile: 4960.093709
* Interquartile Range (IQR = Q3 - Q1): 1861.527271
* Median Absolute Deviation (MAD): 954.088281

**Distribution Shape & Normality Assessment:**
* Skewness: -0.0895 (Approximately symmetric)
* Kurtosis (Excess): -1.0784 (Platykurtic (light tails))
* Histogram Analysis: Distribution spans 10 bins
* Normality Tests:
    * Shapiro-Wilk Test: W-statistic = 18.337192, p-value = NaN (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Jarque-Bera Test: JB-statistic = 4.879556, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))
    * Kolmogorov-Smirnov Test: D-statistic = 0.072815, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))

**Univariate Outlier Analysis:**
* Method 1: IQR Proximity Rule
    * Lower Fence (Q1 - 1.5 * IQR): -761.837697
    * Upper Fence (Q3 + 1.5 * IQR): 6684.271387
    * Number of Outliers (Below Lower): 0 (0%)
    * Number of Outliers (Above Upper): 0 (0%)
    * Extreme Outliers (using 3.0 * IQR factor): 0 (0%)
* Method 2: Z-Score Method
    * Standard Deviations from Mean for Threshold: +/- 3
    * Number of Outliers (Z-score < -3): 0
    * Number of Outliers (Z-score > +3): 0
* Method 3: Modified Z-Score (using MAD)
    * Threshold: +/- 3.5
    * Number of Outliers: 0
* Summary of Outliers: Total 0 (0%). Min Outlier Value: 0, Max Outlier Value: 0.
* Potential Impact: Low outlier impact

**Specific Numerical Patterns & Characteristics:**
* Percentage of Zero Values: 0%
* Percentage of Negative Values: 0%
* Round Numbers Analysis: Moderate rounding detected
* Potential for Log Transformation: Good candidate for log transformation due to wide range

---
**Column: `composite_risk_score`**
* **Detected Data Type:** numerical_float
* **Inferred Semantic Type:** rating
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 98
    * Missing Values: 0 (0%)
    * Unique Values: 85 (86.73% of total)

**3.2.A. Numerical Column Analysis:**

**Descriptive Statistics:**
* Minimum: 1.9799999999999998
* Maximum: 7.33
* Range: 5.3500000000000005
* Sum: 429.29
* Mean (Arithmetic): 4.38051
* Median (50th Percentile): 4.26444
* Mode(s): [3.1799999999999997 (Frequency: 2.04%), 4.5600000000000005 (Frequency: 2.04%), 4.17 (Frequency: 2.04%), 3.39 (Frequency: 2.04%), 4.68 (Frequency: 2.04%)]
* Standard Deviation: 1.142952
* Variance: 1.30634
* Coefficient of Variation (CV): 0.2609%

**Quantile & Percentile Statistics:**
* 1st Percentile: 2.266652
* 5th Percentile: 2.739428
* 10th Percentile: 3.044688
* 25th Percentile (Q1 - First Quartile): 3.511684
* 75th Percentile (Q3 - Third Quartile): 5.216311
* 90th Percentile: 5.93138
* 95th Percentile: 6.149933
* 99th Percentile: 6.582947
* Interquartile Range (IQR = Q3 - Q1): 1.704626
* Median Absolute Deviation (MAD): 0.86556

**Distribution Shape & Normality Assessment:**
* Skewness: 0.1407 (Approximately symmetric)
* Kurtosis (Excess): -0.5245 (Platykurtic (light tails))
* Histogram Analysis: Distribution spans 10 bins
* Normality Tests:
    * Shapiro-Wilk Test: W-statistic = 16.460303, p-value = NaN (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Jarque-Bera Test: JB-statistic = 1.44683, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))
    * Kolmogorov-Smirnov Test: D-statistic = 0.06141, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))

**Univariate Outlier Analysis:**
* Method 1: IQR Proximity Rule
    * Lower Fence (Q1 - 1.5 * IQR): 0.954745
    * Upper Fence (Q3 + 1.5 * IQR): 7.77325
    * Number of Outliers (Below Lower): 0 (0%)
    * Number of Outliers (Above Upper): 0 (0%)
    * Extreme Outliers (using 3.0 * IQR factor): 0 (0%)
* Method 2: Z-Score Method
    * Standard Deviations from Mean for Threshold: +/- 3
    * Number of Outliers (Z-score < -3): 0
    * Number of Outliers (Z-score > +3): 0
* Method 3: Modified Z-Score (using MAD)
    * Threshold: +/- 3.5
    * Number of Outliers: 0
* Summary of Outliers: Total 0 (0%). Min Outlier Value: 0, Max Outlier Value: 0.
* Potential Impact: Low outlier impact

**Specific Numerical Patterns & Characteristics:**
* Percentage of Zero Values: 0%
* Percentage of Negative Values: 0%
* Round Numbers Analysis: No significant rounding detected
* Potential for Log Transformation: Log transformation may not be beneficial

---
**Column: `raw_risk_score`**
* **Detected Data Type:** numerical_float
* **Inferred Semantic Type:** rating
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 98
    * Missing Values: 0 (0%)
    * Unique Values: 85 (86.73% of total)

**3.2.A. Numerical Column Analysis:**

**Descriptive Statistics:**
* Minimum: 1.9799999999999998
* Maximum: 7.33
* Range: 5.3500000000000005
* Sum: 429.29
* Mean (Arithmetic): 4.38051
* Median (50th Percentile): 4.26444
* Mode(s): [3.1799999999999997 (Frequency: 2.04%), 4.5600000000000005 (Frequency: 2.04%), 4.17 (Frequency: 2.04%), 3.39 (Frequency: 2.04%), 4.68 (Frequency: 2.04%)]
* Standard Deviation: 1.142952
* Variance: 1.30634
* Coefficient of Variation (CV): 0.2609%

**Quantile & Percentile Statistics:**
* 1st Percentile: 2.266652
* 5th Percentile: 2.739428
* 10th Percentile: 3.044688
* 25th Percentile (Q1 - First Quartile): 3.511684
* 75th Percentile (Q3 - Third Quartile): 5.216311
* 90th Percentile: 5.93138
* 95th Percentile: 6.149933
* 99th Percentile: 6.582947
* Interquartile Range (IQR = Q3 - Q1): 1.704626
* Median Absolute Deviation (MAD): 0.86556

**Distribution Shape & Normality Assessment:**
* Skewness: 0.1407 (Approximately symmetric)
* Kurtosis (Excess): -0.5245 (Platykurtic (light tails))
* Histogram Analysis: Distribution spans 10 bins
* Normality Tests:
    * Shapiro-Wilk Test: W-statistic = 16.460303, p-value = NaN (Data significantly deviates from normal distribution (p ≤ 0.05))
    * Jarque-Bera Test: JB-statistic = 1.44683, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))
    * Kolmogorov-Smirnov Test: D-statistic = 0.06141, p-value = 0.2 (Data consistent with normal distribution (p > 0.05))

**Univariate Outlier Analysis:**
* Method 1: IQR Proximity Rule
    * Lower Fence (Q1 - 1.5 * IQR): 0.954745
    * Upper Fence (Q3 + 1.5 * IQR): 7.77325
    * Number of Outliers (Below Lower): 0 (0%)
    * Number of Outliers (Above Upper): 0 (0%)
    * Extreme Outliers (using 3.0 * IQR factor): 0 (0%)
* Method 2: Z-Score Method
    * Standard Deviations from Mean for Threshold: +/- 3
    * Number of Outliers (Z-score < -3): 0
    * Number of Outliers (Z-score > +3): 0
* Method 3: Modified Z-Score (using MAD)
    * Threshold: +/- 3.5
    * Number of Outliers: 0
* Summary of Outliers: Total 0 (0%). Min Outlier Value: 0, Max Outlier Value: 0.
* Potential Impact: Low outlier impact

**Specific Numerical Patterns & Characteristics:**
* Percentage of Zero Values: 0%
* Percentage of Negative Values: 0%
* Round Numbers Analysis: No significant rounding detected
* Potential for Log Transformation: Log transformation may not be beneficial

---
**Column: `risk_category`**
* **Detected Data Type:** categorical
* **Inferred Semantic Type:** category
* **Data Quality Flag:** Good
* **Quick Stats:**
    * Total Values (Count): 98
    * Missing Values: 0 (0%)
    * Unique Values: 3 (3.06% of total)

**3.2.B. Categorical Column Analysis:**

**Frequency & Proportionality:**
* Number of Unique Categories: 3
* Mode (Most Frequent Category): `Moderate Risk` (Frequency: 78, 79.59%)
* Second Most Frequent Category: `High Risk` (Frequency: 10, 10.2%)
* Least Frequent Category: `Low Risk` (Frequency: 10, 10.2%)
* Frequency Distribution Table (Top 10):

| Category Label | Count | Percentage (%) | Cumulative % |
|----------------|-------|----------------|--------------|
| Moderate Risk | 78 | 79.59% | 79.59% |
| High Risk | 10 | 10.2% | 89.79% |
| Low Risk | 10 | 10.2% | 99.99% |

**Diversity & Balance:**
* Shannon Entropy: 0.9341 (Range: 0 to 1.585)
* Gini Impurity: 0.3457
* Interpretation of Balance: Unbalanced distribution
* Major Category Dominance: Major category present

**Category Label Analysis:**
* Minimum Label Length: 8 characters
* Maximum Label Length: 13 characters
* Average Label Length: 12.1 characters
* Empty String or Null-like Labels: 0 occurrences

**Potential Issues & Recommendations:**


* No significant issues detected.

**3.3. Bivariate Analysis (Exploring Relationships Between Pairs of Variables):**

**Numerical vs. Numerical:**
    * **Correlation Matrix Summary (Pearson's r):**
        * Total Pairs Analysed: 26
        * Top 5 Strongest Positive Correlations:
        1. `health_utilisation_risk` vs `composite_risk_score`: r = 0.8108 (Correlation significantly different from zero (p ≤ 0.05)) - Very Strong positive correlation (Correlation significantly different from zero (p ≤ 0.05)).
        2. `health_utilisation_risk` vs `raw_risk_score`: r = 0.8108 (Correlation significantly different from zero (p ≤ 0.05)) - Very Strong positive correlation (Correlation significantly different from zero (p ≤ 0.05)).
        3. `seifa_risk_score` vs `composite_risk_score`: r = 0.67 (Correlation significantly different from zero (p ≤ 0.05)) - Strong positive correlation (Correlation significantly different from zero (p ≤ 0.05)).
        4. `seifa_risk_score` vs `raw_risk_score`: r = 0.67 (Correlation significantly different from zero (p ≤ 0.05)) - Strong positive correlation (Correlation significantly different from zero (p ≤ 0.05)).
        5. `health_utilisation_risk` vs `total_prescriptions`: r = 0.5293 (Correlation significantly different from zero (p ≤ 0.05)) - Moderate positive correlation (Correlation significantly different from zero (p ≤ 0.05)).
        * Top 5 Strongest Negative Correlations:
        1. `total_prescriptions` vs `usual_resident_population`: r = -0.1428 (Correlation not significantly different from zero (p > 0.05)) - Very Weak negative correlation (Correlation not significantly different from zero (p > 0.05)).
        2. `total_prescriptions` vs `chronic_medication_rate`: r = -0.1188 (Correlation not significantly different from zero (p > 0.05)) - Very Weak negative correlation (Correlation not significantly different from zero (p > 0.05)).
        3. `sa2_code_2021` vs `health_utilisation_risk`: r = -0.0804 (Correlation not significantly different from zero (p > 0.05)) - Very Weak negative correlation (Correlation not significantly different from zero (p > 0.05)).
        4. `sa2_code_2021` vs `total_prescriptions`: r = -0.0778 (Correlation not significantly different from zero (p > 0.05)) - Very Weak negative correlation (Correlation not significantly different from zero (p > 0.05)).
        5. `sa2_code_2021` vs `chronic_medication_rate`: r = -0.0708 (Correlation not significantly different from zero (p > 0.05)) - Very Weak negative correlation (Correlation not significantly different from zero (p > 0.05)).
        * Strong Correlations (|r| > 0.5): 5 pairs identified
    * **Scatter Plot Insights (Key Relationships):**
        * `sa2_code_2021` vs `seifa_risk_score`: "50 point sample shows linear relationship" (Recommended: Scatter plot with trend line)
        * `sa2_code_2021` vs `health_utilisation_risk`: "50 point sample shows linear relationship" (Recommended: Scatter plot with trend line)
        * `sa2_code_2021` vs `total_prescriptions`: "50 point sample shows linear relationship" (Recommended: Scatter plot with trend line)

**Numerical vs. Categorical:**
    * **Comparative Statistics (Mean/Median by Category):**
    * **`state_name` by `sa2_code_2021`:**
        | Category | Mean | Median | StdDev | Count |
        |----------|------|--------|--------|-------|
        | QLD | 100001040.6 | 100001040.6 | 29.0225 | 30 |
        | NSW | 100001055.6061 | 100001055.6061 | 32.3775 | 33 |
        | VIC | 100001050.5714 | 100001050.5714 | 22.9898 | 35 |
        * **Statistical Tests:** 
            * **ANOVA F-test:** F(2,95) = 2.276, p = 0.8917 (not significant (p ≥ 0.05)). No significant difference between group means.
            * **Kruskal-Wallis test:** H = 4.051, df = 2, p = 0.1319 (not significant (p ≥ 0.05)). No significant difference between group distributions.
        * **Summary:** NSW has highest mean (100001055.61), QLD has lowest (100001040.60)

    * **`risk_category` by `sa2_code_2021`:**
        | Category | Mean | Median | StdDev | Count |
        |----------|------|--------|--------|-------|
        | Moderate Risk | 100001047.4744 | 100001047.4744 | 27.5065 | 78 |
        | High Risk | 100001051.1 | 100001051.1 | 36.5088 | 10 |
        | Low Risk | 100001060.9 | 100001060.9 | 28.3177 | 10 |
        * **Statistical Tests:** 
            * **ANOVA F-test:** F(2,95) = 1.004, p = 0.6297 (not significant (p ≥ 0.05)). No significant difference between group means.
            * **Kruskal-Wallis test:** H = 2.042, df = 2, p = 0.3603 (not significant (p ≥ 0.05)). No significant difference between group distributions.
        * **Summary:** Low Risk has highest mean (100001060.90), Moderate Risk has lowest (100001047.47)

    * **`state_name` by `seifa_risk_score`:**
        | Category | Mean | Median | StdDev | Count |
        |----------|------|--------|--------|-------|
        | QLD | 5.6467 | 5.6467 | 1.2556 | 30 |
        | NSW | 5.3303 | 5.3303 | 1.0994 | 33 |
        | VIC | 5.6243 | 5.6243 | 0.9869 | 35 |
        * **Statistical Tests:** 
            * **ANOVA F-test:** F(2,95) = 0.823, p = 0.5577 (not significant (p ≥ 0.05)). No significant difference between group means.
            * **Kruskal-Wallis test:** H = 1.667, df = 2, p = 0.4345 (not significant (p ≥ 0.05)). No significant difference between group distributions.
        * **Summary:** QLD has highest mean (5.65), NSW has lowest (5.33)

**Categorical vs. Categorical:**
    * **Contingency Table Analysis:**
    * **`state_name` vs `risk_category`:**
        * **Contingency Table (Top 3x3):**
        |             | Moderate Risk | High Risk | Low Risk |
        |-------------|-------------|-------------|-------------|
        | QLD | 23 | 2 | 5 |
        | NSW | 24 | 5 | 4 |
        | VIC | 31 | 3 | 1 |
        * **Association Tests:**
            * Chi-Squared: χ² = 5.031298, df = 4, p-value = 1 (Chi-squared test assumptions violated: >20% of cells have expected frequency <5)
            * Cramer's V: 0 (Very weak association)
        * **Insights:** Most common combination: VIC & Moderate Risk (31 occurrences). Association strength: weak.

**3.4. Multivariate Analysis (Advanced Multi-Variable Interactions):**

**Analysis Overview:** Dataset well-suited for comprehensive multivariate analysis

**Variables Analysed:** sa2_code_2021, seifa_risk_score, health_utilisation_risk, total_prescriptions, chronic_medication_rate, usual_resident_population, composite_risk_score, raw_risk_score (8 numerical variables)

**3.4.A. Principal Component Analysis (PCA):**
* **Variance Explained:** 4 components explain 85% of variance, 5 explain 95%
* **Most Influential Variables:** total_prescriptions (loading: 0.862), chronic_medication_rate (loading: 0.706), health_utilisation_risk (loading: 0.652)
* **Recommendation:** Moderate dimensionality reduction: 5 components retain 90% of variance; Kaiser criterion suggests 3 meaningful components (eigenvalue > 1); Scree plot suggests 2 components based on elbow criterion; 1 variables show high importance, 2 show low importance - consider feature selection; Selected variables with importance >= 0.200

**3.4.B. Cluster Analysis:**
* **Optimal Clusters:** 2 clusters identified using elbow method
* **Cluster Quality:** Silhouette score = 0.239 (undefined)
* **Cluster Profiles:**
    * **Cluster 1:** Cluster characterized by moderately higher composite_risk_score and moderately higher raw_risk_score (54 members) (54 observations)
    * **Cluster 2:** Cluster characterized by moderately lower composite_risk_score and moderately lower raw_risk_score (44 members) (44 observations)
* **Recommendation:** Consider feature engineering or different clustering approach due to weak structure; Low variance explained - consider dimensionality reduction before clustering; Dataset may have limited natural clustering - verify with domain knowledge

**3.4.C. Multivariate Outlier Detection:**
* Not applicable: Outlier analysis failed: Covariance matrix is singular - cannot compute Mahalanobis distances

**3.4.D. Multivariate Normality Tests:**
* **Overall Assessment:** Multivariate normality not rejected (confidence: 51.9%)
* **Mardia's Test:** Mardia test failed due to matrix singularity or computational issues
* **Royston's Test:** Multivariate normality not rejected (p >= 0.05)

* **Recommendations:** Multivariate normal assumption satisfied - parametric methods appropriate

**3.4.E. Variable Relationship Analysis:**
* **Key Interactions:** composite_risk_score ↔ raw_risk_score (linear, strength: 1.000); health_utilisation_risk ↔ composite_risk_score (linear, strength: 0.811); health_utilisation_risk ↔ raw_risk_score (linear, strength: 0.811)
* **Correlated Groups:** 1 groups of highly correlated variables identified
* **Redundant Variables:** 1 variables with high correlation (r > 0.9)
* **Independent Variables:** 2 variables with low correlations
* **Dimensionality:** Reduction recommended - 1 effective dimensions detected

**3.4.F. Multivariate Insights & Recommendations:**
**Key Multivariate Findings:**
    1. 4 principal components explain 85% of variance
    2. 2 natural clusters identified (silhouette: 0.239)
    3. 3 strong variable relationships identified

**Data Quality Issues:**
    * Redundant variables detected

**Preprocessing Recommendations:**
    * Consider removing highly correlated variables

**Analysis Recommendations:**
    * Dimensionality reduction recommended based on correlation structure

**3.5. Specific Analysis Modules (Activated Based on Data Characteristics):**

    * **3.5.B. Text Analytics Deep Dive:**
        * **Detected Text Columns:** 1 columns identified
        * **Primary Text Column:** `sa2_name_2021`
        * **Advanced Analysis Available:** N-gram analysis, topic modelling, named entity recognition, sentiment analysis
        * **Sample Keywords:** [test, area]
        * **Recommendation:** Apply NLP preprocessing pipeline for deeper text insights if required for analysis goals.

**3.6. EDA Summary & Key Hypotheses/Insights:**
    * **Top Statistical Findings:**
    1. Streaming analysis processed 100 rows using only 0MB peak memory
    * **Data Quality Issues Uncovered:**
    * No major data quality issues identified during EDA.
    * **Hypotheses Generated for Further Testing:**
    * No specific hypotheses generated - consider domain knowledge for hypothesis formation.
    * **Recommendations for Data Preprocessing & Feature Engineering:**
    * Standard preprocessing steps recommended based on detected data types.
    * **Critical Warnings & Considerations:**




---

**Analysis Performance Summary:**
* **Processing Time:** 62ms (0.06 seconds)
* **Rows Analysed:** 100
* **Memory Efficiency:** Constant ~0MB usage
* **Analysis Method:** Streaming with online algorithms
* **Dataset Size:** 100 records across 11 columns