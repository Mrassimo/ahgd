# Australian Health and Geographic Data (AHGD) - R Example
# 
# This example demonstrates how to load and analyse the AHGD dataset using R

library(arrow)      # For Parquet files
library(readr)     # For CSV files
library(jsonlite)  # For JSON files
library(dplyr)     # For data manipulation
library(ggplot2)   # For visualisation

# Load dataset (Parquet recommended for performance)
load_ahgd_data <- function(format = "parquet") {
  if (format == "parquet") {
    data <- arrow::read_parquet("ahgd_data.parquet")
  } else if (format == "csv") {
    data <- readr::read_csv("ahgd_data.csv")
  } else if (format == "json") {
    json_data <- jsonlite::fromJSON("ahgd_data.json")
    data <- as.data.frame(json_data$data)
  } else {
    stop("Unsupported format. Use 'parquet', 'csv', or 'json'")
  }
  
  return(data)
}

# Basic analysis
analyse_ahgd <- function() {
  # Load data
  df <- load_ahgd_data("parquet")
  
  cat("Dataset dimensions:", dim(df), "\n")
  cat("Column names:", paste(names(df), collapse = ", "), "\n\n")
  
  # Summary statistics for numeric columns
  numeric_cols <- sapply(df, is.numeric)
  if (any(numeric_cols)) {
    cat("Summary statistics:\n")
    print(summary(df[, numeric_cols]))
  }
  
  # State-level health indicators
  if ("state_name" %in% names(df) && "life_expectancy_years" %in% names(df)) {
    state_summary <- df %>%
      group_by(state_name) %>%
      summarise(
        avg_life_expectancy = mean(life_expectancy_years, na.rm = TRUE),
        avg_smoking = mean(smoking_prevalence_percent, na.rm = TRUE),
        avg_obesity = mean(obesity_prevalence_percent, na.rm = TRUE),
        .groups = 'drop'
      )
    
    cat("\nHealth indicators by state:\n")
    print(state_summary)
  }
  
  return(df)
}

# Create visualisations
create_plots <- function(df) {
  # Life expectancy distribution
  p1 <- ggplot(df, aes(x = life_expectancy_years)) +
    geom_histogram(bins = 20, fill = "skyblue", alpha = 0.7) +
    labs(title = "Distribution of Life Expectancy",
         x = "Life Expectancy (Years)",
         y = "Count") +
    theme_minimal()
  
  # Smoking vs Life Expectancy
  if (all(c("smoking_prevalence_percent", "life_expectancy_years") %in% names(df))) {
    p2 <- ggplot(df, aes(x = smoking_prevalence_percent, y = life_expectancy_years)) +
      geom_point(alpha = 0.6, color = "darkblue") +
      geom_smooth(method = "lm", se = TRUE, color = "red") +
      labs(title = "Smoking Prevalence vs Life Expectancy",
           x = "Smoking Prevalence (%)",
           y = "Life Expectancy (Years)") +
      theme_minimal()
    
    # Save plots
    ggsave("life_expectancy_distribution.png", p1, width = 8, height = 6, dpi = 300)
    ggsave("smoking_vs_life_expectancy.png", p2, width = 8, height = 6, dpi = 300)
  }
}

# Run analysis
main <- function() {
  cat("Loading Australian Health and Geographic Data...\n")
  data <- analyse_ahgd()
  
  cat("Creating visualisations...\n")
  create_plots(data)
  
  cat("Analysis complete!\n")
}

# Execute if run directly
if (sys.nframe() == 0) {
  main()
}
