# ---------------------------------------
# Robust Yield Curve vs Commodity Futures Spread Dynamic Correlation Analysis
# Using rmgarch package for DCC-GARCH modeling with multiple rolling windows (512, 126, 256 days)
# Modified version with increased tolerance and maximum iterations
# ---------------------------------------

# Clear environment
rm(list = ls())

# Add locale setting for UTF-8 support
Sys.setlocale("LC_CTYPE", "en_US.UTF-8")

# Load required packages with error handling
required_packages <- c("rmgarch", "rugarch", "xts", "zoo", "ggplot2", "parallel")
for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Step 1: Load and preprocess data
cat("Loading and preprocessing data...\n")
data <- read.csv("merged_data.csv")
data$Date <- as.Date(data$Date)

# Print data summary
cat("Data summary:\n")
print(summary(data))

# Identify column names for spreads
if("X6M_Spread" %in% colnames(data)) {
  spread_col <- "X6M_Spread"
} else if("6M_Spread" %in% colnames(data)) {
  spread_col <- "6M_Spread"
} else {
  spread_col <- "6M_Spread"  # Default
  cat("Using '6M_Spread' as spread column\n")
}

if("T10Y2Y" %in% colnames(data)) {
  yield_col <- "T10Y2Y"
} else {
  yield_col <- "T10Y2Y"  # Default
  cat("Using 'T10Y2Y' as yield column\n")
}

# Handle missing values properly
data[[spread_col]] <- na.locf(data[[spread_col]], na.rm = FALSE)
data[[yield_col]] <- na.locf(data[[yield_col]], na.rm = FALSE)

# Create input matrix with proper NA handling
input_data <- data.frame(
  Date = data$Date,
  Oil_Spread = data[[spread_col]],
  Yield_Spread = data[[yield_col]]
)

# Remove rows with any NA values
clean_data <- na.omit(input_data)
cat("Removed", nrow(input_data) - nrow(clean_data), "rows with NA values\n")

# Extract the date vector and create xts objects for model
processed_dates <- clean_data$Date
xts_data <- xts(clean_data[, c("Oil_Spread", "Yield_Spread")], 
                order.by = processed_dates)

# Preprocess data for better GARCH fitting
# 1. Convert to returns (day-to-day changes)
returns_data <- diff(xts_data)
returns_data <- returns_data[-1,] # Remove first NA row
returns_dates <- index(returns_data)

# 2. Winsorize to handle extreme outliers (cap at 99th percentile)
winsorize <- function(x, prob = 0.01) {
  lower_bound <- quantile(x, prob, na.rm = TRUE)
  upper_bound <- quantile(x, 1 - prob, na.rm = TRUE)
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

for(i in 1:ncol(returns_data)) {
  returns_data[,i] <- winsorize(returns_data[,i])
}

cat("Preprocessed data for GARCH modeling. Used returns and winsorized outliers.\n")
cat("Final data dimensions:", nrow(returns_data), "rows x", ncol(returns_data), "columns\n")

# Verify no NA values remain
cat("NA values in processed data:", sum(is.na(returns_data)), "\n")

# Step 2: Implement rolling window DCC-GARCH analysis with different window sizes
cat("\nImplementing rolling window DCC-GARCH analysis with multiple window sizes...\n")

# Define window sizes (512, 126, and 256 trading days)
window_sizes <- c(512, 126, 256)

fit_dcc_garch <- function(window_data) {
  # GARCH specification with t-distribution
  uspec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
    distribution.model = "std"  # t-distribution handles outliers better
  )
  
  # Create multivariate GARCH specification
  mspec <- multispec(replicate(ncol(window_data), uspec))
  
  # DCC specification
  dcc_spec <- dccspec(
    uspec = mspec,
    dccOrder = c(1, 1),
    distribution = "mvt"  # multivariate t-distribution
  )
  
  # Fit control parameters - INCREASED TOLERANCE AND MAX ITERATIONS
  fit_control <- list(
    eval.se = FALSE,
    scale = TRUE,
    tol = 1e-2,       # Further increased tolerance
    maxit = 20000,     # Further increased iterations
    stationarity = FALSE  # Relax stationarity constraint
  )
  
  # Fit the model
  tryCatch({
    dcc_fit <- dccfit(
      dcc_spec,
      data = window_data,
      fit.control = fit_control,
      solver = "solnp"  # Try alternative solver
    )
    
    # Extract correlation
    dcc_cors <- rcor(dcc_fit)
    last_cor <- dcc_cors[1, 2, dim(dcc_cors)[3]]
    
    return(last_cor)
  }, error = function(e) {
    # Enhanced error reporting
    cat("Error in DCC fitting:", conditionMessage(e), "\n")
    return(NA)
  })
}

# Process each window size
for (window_size in window_sizes) {
  cat("\n*** Processing", window_size, "day window ***\n")
  
  # Create storage for rolling correlation results
  total_periods <- nrow(returns_data) - window_size + 1
  dates_rolling <- returns_dates[(window_size):length(returns_dates)]
  rolling_correlations <- numeric(total_periods)
  
  # Create a progress indicator
  progress_step <- ceiling(total_periods / 10)
  
  # Set up parallel processing if available
  num_cores <- min(detectCores() - 1, 4)  # Use up to 4 cores
  if(num_cores > 1) {
    cat("Using parallel processing with", num_cores, "cores\n")
    cl <- makeCluster(num_cores)
    clusterExport(cl, c("fit_dcc_garch", "window_size", "returns_data"))
    clusterEvalQ(cl, {
      library(rmgarch)
      library(rugarch)
    })
    
    # Create windows in parallel with better error reporting
    result_list <- parLapply(cl, 1:total_periods, function(i) {
      start_idx <- i
      end_idx <- i + window_size - 1
      
      if (end_idx <= nrow(returns_data)) {
        window_data <- returns_data[start_idx:end_idx, ]
        
        # Calculate DCC-GARCH correlation
        result <- fit_dcc_garch(window_data)
        if(i %% 100 == 0) {
          if(is.na(result)) {
            cat("Window", i, "produced NA correlation\n")
          } else {
            cat("Window", i, "correlation:", result, "\n")
          }
        }
        return(result)
      } else {
        return(NA)
      }
    })
    
    stopCluster(cl)
    
    # Extract results
    for(i in 1:length(result_list)) {
      rolling_correlations[i] <- result_list[[i]]
    }
  } else {
    # Non-parallel version
    for (i in 1:total_periods) {
      # Show progress
      if (i %% progress_step == 0 || i == 1 || i == total_periods) {
        cat(sprintf("Processing window %d of %d (%.1f%%)\n", 
                    i, total_periods, i/total_periods*100))
      }
      
      # Extract current window
      start_idx <- i
      end_idx <- i + window_size - 1
      
      if (end_idx <= nrow(returns_data)) {
        window_data <- returns_data[start_idx:end_idx, ]
        
        # Calculate DCC-GARCH correlation with better error handling
        rolling_correlations[i] <- fit_dcc_garch(window_data)
      }
    }
  }
  
  # Check if we have valid results
  valid_results <- sum(!is.na(rolling_correlations))
  cat("\nNumber of valid correlation results:", valid_results, "out of", total_periods, "\n")
  
  # Create results dataframe
  results_df <- data.frame(
    Date = dates_rolling,
    DCC_Correlation = rolling_correlations
  )
  
  # Handle any NA values by filling them with nearest non-NA value
  if(sum(is.na(results_df$DCC_Correlation)) > 0) {
    cat("Filling", sum(is.na(results_df$DCC_Correlation)), "NA values with nearest available values\n")
    results_df$DCC_Correlation <- na.approx(results_df$DCC_Correlation, na.rm = FALSE)
    
    # If there are still NAs at the beginning or end, use locf and nocb
    results_df$DCC_Correlation <- na.locf(results_df$DCC_Correlation, na.rm = FALSE)
    results_df$DCC_Correlation <- na.locf(results_df$DCC_Correlation, fromLast = TRUE, na.rm = FALSE)
  }
  
  # Save results to CSV
  output_file <- paste0("yield_spread_commodity_correlation_", window_size, "day.csv")
  write.csv(results_df, output_file, row.names = FALSE)
  cat("Results saved to '", output_file, "'\n")
  
  # Plot results
  cat("\nPlotting results...\n")
  
  # Create DCC-GARCH only plot
  p_dcc <- ggplot(results_df, aes(x = Date, y = DCC_Correlation)) +
    geom_line(color = "steelblue") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = paste0(window_size, "-Day Rolling Window DCC-GARCH Correlation"),
         subtitle = "Oil Spread vs Yield Curve Spread",
         y = "Correlation Coefficient",
         x = "Date") +
    theme_minimal()
  
  # Save the DCC-only plot
  plot_file <- paste0("yield_commodity_dcc_correlation_", window_size, "day.pdf")
  ggsave(plot_file, p_dcc, width = 10, height = 6)
  cat("DCC-only plot saved to '", plot_file, "'\n")
  
  # Calculate summary statistics
  cat("\nSummary statistics for", window_size, "-day rolling window DCC-GARCH correlations:\n")
  print(summary(results_df$DCC_Correlation))
  
  # Calculate proportion of positive vs negative correlations for DCC
  pos_corr <- sum(results_df$DCC_Correlation > 0, na.rm = TRUE)
  neg_corr <- sum(results_df$DCC_Correlation < 0, na.rm = TRUE)
  total_obs <- nrow(results_df)
  
  cat("\nCorrelation direction analysis (DCC-GARCH):\n")
  cat("- Positive correlations:", pos_corr, "(", 
      round(pos_corr/total_obs*100, 1), "%)\n")
  cat("- Negative correlations:", neg_corr, "(", 
      round(neg_corr/total_obs*100, 1), "%)\n")
  
  # Find periods of strongest positive and negative correlation for DCC
  if(any(!is.na(results_df$DCC_Correlation))) {
    max_corr_idx <- which.max(results_df$DCC_Correlation)
    min_corr_idx <- which.min(results_df$DCC_Correlation)
    
    cat("\nStrongest positive correlation (DCC-GARCH):", 
        round(results_df$DCC_Correlation[max_corr_idx], 4), 
        "on", format(results_df$Date[max_corr_idx], "%Y-%m-%d"), "\n")
    cat("Strongest negative correlation (DCC-GARCH):", 
        round(results_df$DCC_Correlation[min_corr_idx], 4), 
        "on", format(results_df$Date[min_corr_idx], "%Y-%m-%d"), "\n")
  } else {
    cat("\nNo valid correlation values found to determine max/min\n")
  }
}

# Step 3: Generate comparison plot of all window sizes
cat("\nGenerating comparison plot for all window sizes...\n")

# Load the results for each window size
results_list <- list()
for (window_size in window_sizes) {
  file_name <- paste0("yield_spread_commodity_correlation_", window_size, "day.csv")
  if (file.exists(file_name)) {
    df <- read.csv(file_name)
    df$Date <- as.Date(df$Date)
    df$Window <- paste0(window_size, " days")
    results_list[[as.character(window_size)]] <- df
  }
}

# Combine all results into one dataframe for plotting
if (length(results_list) > 0) {
  combined_results <- do.call(rbind, lapply(names(results_list), function(ws) {
    df <- results_list[[ws]]
    df$Window_Size <- ws
    return(df)
  }))
  
  # Plot comparison
  p_comparison <- ggplot(combined_results, aes(x = Date, y = DCC_Correlation, color = Window_Size)) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = "Comparison of Rolling Window DCC-GARCH Correlations",
         subtitle = "Oil Spread vs Yield Curve Spread",
         y = "Correlation Coefficient",
         x = "Date",
         color = "Window Size (Days)") +
    theme_minimal() +
    scale_color_brewer(palette = "Set1")
  
  # Save the comparison plot
  ggsave("yield_commodity_dcc_correlation_comparison.pdf", p_comparison, width = 12, height = 7)
  cat("Comparison plot saved to 'yield_commodity_dcc_correlation_comparison.pdf'\n")
}

cat("\nRolling window analysis complete for all window sizes!\n")