#!/usr/bin/env Rscript

# ==============================================================================
# DA-SMOTE-PENN Full Benchmark Pipeline for Publication
# 
# Features:
# - Standalone dataset acquisition & standardization (no manual CSV prep needed)
# - Stronger baselines: SMOTE, ADASYN, Borderline-SMOTE, SMOTE-Tomek,
#   SMOTE-ENN, LD-SMOTE, DA-SMOTE, PENN, DA-SMOTE-PENN
# - Multiple classifiers: kNN, logistic regression, SVM, random forest, XGBoost
# - Metrics: Sensitivity, Specificity, Precision, F1, G-mean,
#   Balanced Accuracy, MCC, AUPRC, AUROC
# - Ablation, sensitivity, significance tests, runtime logging, and LaTeX export
# ==============================================================================

options(stringsAsFactors = FALSE)
options(bitmapType = "cairo")
options(dplyr.summarise.inform = FALSE)

# ==============================================================================
# 1. PACKAGE INSTALLATION & LOADING
# ==============================================================================
install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
}

required_pkgs <- c(
  "modeldata", "ROSE", "mlbench", "data.table", "dplyr", "tidyr", 
  "purrr", "stringr", "tibble", "rsample", "pROC", "PRROC", 
  "FNN", "class", "e1071", "ranger", "xgboost", "coin"
)
install_if_missing(required_pkgs)

suppressPackageStartupMessages({
  library(modeldata)
  library(ROSE)
  library(mlbench)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(tibble)
  library(rsample)
  library(pROC)
  library(PRROC)
  library(FNN)
  library(class)
  library(e1071)
  library(ranger)
  library(xgboost)
  library(coin)
})

# ==============================================================================
# 2. DATASET ACQUISITION & STANDARDIZATION
# ==============================================================================
datasets <- list()

# A. modeldata: Real-world business datasets
data("attrition", package = "modeldata")
datasets$attrition <- attrition %>%
  mutate(Class = ifelse(Attrition == "Yes", "positive", "negative")) %>%
  select(-Attrition)

data("credit_data", package = "modeldata")
datasets$credit_data <- credit_data %>%
  na.omit() %>% 
  mutate(Class = ifelse(Status == "bad", "positive", "negative")) %>%
  select(-Status)

# B. KEEL benchmarks
cat("Downloading 'newthyroid1' and 'glass0' directly from source...\n")
download.file(
  "https://raw.githubusercontent.com/cran/imbalance/master/data/newthyroid1.rda", 
  destfile = "newthyroid1.rda", mode = "wb", quiet = TRUE
)
download.file(
  "https://raw.githubusercontent.com/cran/imbalance/master/data/glass0.rda", 
  destfile = "glass0.rda", mode = "wb", quiet = TRUE
)
load("newthyroid1.rda")
load("glass0.rda")
file.remove("newthyroid1.rda", "glass0.rda")

datasets$newthyroid1 <- newthyroid1 %>% mutate(Class = as.character(Class))
datasets$glass0 <- glass0 %>% mutate(Class = as.character(Class))

# C. mlbench: Custom-built UCI/KEEL imbalanced benchmarks
data("Vehicle", package = "mlbench")
datasets$vehicle2 <- Vehicle %>%
  mutate(Class = ifelse(Class == "saab", "positive", "negative"))

data("PimaIndiansDiabetes", package = "mlbench")
datasets$pima <- PimaIndiansDiabetes %>%
  na.omit() %>%
  mutate(Class = ifelse(diabetes == "pos", "positive", "negative")) %>%
  select(-diabetes)

# Note: 'hacide' and 'glass4' are intentionally excluded to prevent NaN/errors 
# as identified in previous HPC runs.

# ==============================================================================
# 3. EXPORT TO CSV & GENERATE MANIFEST
# ==============================================================================
dir.create("data", showWarnings = FALSE, recursive = TRUE)
manifest_rows <- list()

cat("\nExporting datasets to 'data/' directory...\n")
cat(sprintf("%-15s | %-5s | %-5s | %-5s | %-6s\n", "Dataset", "Total", "Pos", "Neg", "Ratio"))
cat("-------------------------------------------------------------------\n")

for (name in names(datasets)) {
  df <- datasets[[name]]
  df$Class <- factor(df$Class, levels = c("positive", "negative"))
  
  file_path <- file.path("data", paste0(name, ".csv"))
  write.csv(df, file_path, row.names = FALSE)
  
  n_pos <- sum(df$Class == "positive")
  n_neg <- sum(df$Class == "negative")
  cat(sprintf("%-15s | %-5d | %-5d | %-5d | 1:%.2f\n", 
              name, nrow(df), n_pos, n_neg, n_neg/n_pos))
  
  manifest_rows[[name]] <- tibble(
    dataset = name,
    file = paste0(name, ".csv"),
    target = "Class",
    positive_class = "positive"
  )
}

dataset_manifest <- bind_rows(manifest_rows)
cat("\nManifest generated successfully:\n")
print(dataset_manifest)

# ==============================================================================
# 4. PIPELINE CONFIGURATION
# ==============================================================================
config <- list(
  data_dir = "data",
  output_dir = "outputs",
  results_dir = file.path("outputs", "results"),
  table_dir = file.path("outputs", "tables"),
  figure_dir = file.path("outputs", "figures"),
  log_dir = file.path("outputs", "logs"),
  repo_url = "https://github.com/REPLACE-WITH-ANON-REPOSITORY/da-smote-penn",
  data_url = "https://github.com/REPLACE-WITH-ANON-REPOSITORY/da-smote-penn-data",
  master_seed = 20260828L,
  outer_folds = 5L,
  outer_repeats = 5L,
  inner_folds = 3L,
  tune_classifiers = TRUE,
  primary_metric = "AUPRC",
  decision_threshold = 0.5,
  target_ratio = 0.25,
  benchmark_beta = 0.20,
  benchmark_k_smote = 7L,
  benchmark_k_enn = 3L,
  benchmark_k_ld = 5L,
  alpha_grid = c(0.25, 0.50, 0.75, 1.00),
  beta_grid = c(0.00, 0.05, 0.10, 0.15, 0.20, 0.25),
  k_smote_grid = c(3L, 5L, 7L),
  k_enn_grid = c(3L, 5L, 7L),
  k_ld_grid = c(3L, 5L, 7L),
  simulation_ratios = list(c(5L, 995L), c(10L, 990L), c(100L, 900L), c(200L, 800L), c(500L, 500L)),
  n_sim_replications = 100L,
  simulation_n_test = 1000L,
  methods = c("baseline", "smote", "adasyn", "borderline_smote", "smote_tomek",
              "smote_enn", "da_smote", "penn", "ld_smote", "da_smote_penn"),
  classifiers = c("knn", "logit", "svm_rbf", "rf", "xgb")
)

for (sub in c("data", "outputs", "outputs/results", "outputs/tables", "outputs/figures", "outputs/logs")) {
  dir.create(file.path("outputs", sub), recursive = TRUE, showWarnings = FALSE)
}

set.seed(config$master_seed)
seed_ledger <- tibble(seed_id = 1:50000, seed_value = sample.int(1e9, 50000, replace = FALSE))
write.csv(seed_ledger, file.path(config$results_dir, "seeds_used.csv"), row.names = FALSE)

# ==============================================================================
# 5. HELPERS & RESAMPLING FUNCTIONS
# ==============================================================================
get_next_seed <- local({
  i <- 0L
  function() {
    i <<- i + 1L
    seed_ledger$seed_value[[i]]
  }
})

mode_binary <- function(x) as.integer(mean(x) >= 0.5)
safe_div <- function(a, b) ifelse(b == 0, NA_real_, a / b)

compute_metrics <- function(y_true, prob_pos, threshold = 0.5) {
  y_true <- as.integer(y_true)
  prob_pos <- pmin(pmax(prob_pos, 1e-12), 1 - 1e-12)
  y_pred <- as.integer(prob_pos >= threshold)
  
  TP <- sum(y_true == 1 & y_pred == 1)
  TN <- sum(y_true == 0 & y_pred == 0)
  FP <- sum(y_true == 0 & y_pred == 1)
  FN <- sum(y_true == 1 & y_pred == 0)
  
  sensitivity <- safe_div(TP, TP + FN)
  specificity <- safe_div(TN, TN + FP)
  precision   <- safe_div(TP, TP + FP)
  recall      <- sensitivity
  f1          <- ifelse(is.na(precision) || is.na(recall) || precision + recall == 0, NA_real_, 2 * precision * recall / (precision + recall))
  bal_acc     <- mean(c(sensitivity, specificity), na.rm = TRUE)
  gmean       <- sqrt(sensitivity * specificity)
  denom_mcc   <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  mcc         <- ifelse(denom_mcc == 0, NA_real_, (TP * TN - FP * FN) / denom_mcc)
  
  auroc <- tryCatch(as.numeric(pROC::auc(y_true, prob_pos, quiet = TRUE)), error = function(e) NA_real_)
  auprc <- tryCatch({
    PRROC::pr.curve(scores.class0 = prob_pos[y_true == 1],
                    scores.class1 = prob_pos[y_true == 0],
                    curve = FALSE)$auc.integral
  }, error = function(e) NA_real_)
  
  # Safeguard: warn if classifier is predicting the wrong class probabilities
  if (sum(y_true == 1) > 0 && sum(y_true == 0) > 0) {
    mean_prob_pos <- mean(prob_pos[y_true == 1])
    mean_prob_neg <- mean(prob_pos[y_true == 0])
    if (mean_prob_pos < mean_prob_neg) {
      warning("Classifier predicting wrong class! Mean prob for positive: ", 
              round(mean_prob_pos, 3), " vs negative: ", round(mean_prob_neg, 3))
    }
  }
  
  tibble(
    TP = TP, TN = TN, FP = FP, FN = FN,
    Sensitivity = sensitivity, Specificity = specificity, Precision = precision,
    F1 = f1, Gmean = gmean, BalancedAccuracy = bal_acc, MCC = mcc,
    AUPRC = auprc, AUROC = auroc
  )
}

make_design_matrices <- function(df_train, df_test, target_col, positive_class) {
  y_train <- as.integer(df_train[[target_col]] == positive_class)
  y_test  <- as.integer(df_test[[target_col]]  == positive_class)
  
  x_train_df <- df_train[, setdiff(names(df_train), target_col), drop = FALSE]
  x_test_df  <- df_test[,  setdiff(names(df_test),  target_col), drop = FALSE]
  
  for (nm in names(x_train_df)) {
    if (is.character(x_train_df[[nm]]) || is.factor(x_train_df[[nm]])) {
      x_train_df[[nm]] <- factor(x_train_df[[nm]])
      x_test_df[[nm]]  <- factor(x_test_df[[nm]], levels = levels(x_train_df[[nm]]))
    }
  }
  
  mm_train <- model.matrix(~ . - 1, data = x_train_df)
  mm_test  <- model.matrix(~ . - 1, data = x_test_df)
  
  common_cols <- intersect(colnames(mm_train), colnames(mm_test))
  mm_train <- mm_train[, common_cols, drop = FALSE]
  mm_test  <- mm_test[, common_cols, drop = FALSE]
  
  sds <- apply(mm_train, 2, sd)
  keep <- which(sds > 1e-8)
  mm_train <- mm_train[, keep, drop = FALSE]
  mm_test  <- mm_test[, keep, drop = FALSE]
  
  means <- colMeans(mm_train)
  sds <- apply(mm_train, 2, sd)
  mm_train <- scale(mm_train, center = means, scale = sds)
  mm_test  <- scale(mm_test,  center = means, scale = sds)
  
  list(x_train = as.matrix(mm_train), y_train = y_train,
       x_test = as.matrix(mm_test),   y_test = y_test,
       feature_names = colnames(mm_train))
}

compute_knn_excluding_self <- function(x, k) {
  kn <- FNN::get.knnx(data = x, query = x, k = k + 1)
  idx <- kn$nn.index[, -1, drop = FALSE]
  dist <- kn$nn.dist[, -1, drop = FALSE]
  list(index = idx, dist = dist)
}

allocate_counts <- function(weights, n_total) {
  if (n_total <= 0) return(integer(length(weights)))
  raw <- weights * n_total
  base <- floor(raw)
  rem_n <- n_total - sum(base)
  if (rem_n > 0) {
    ord <- order(raw - base, decreasing = TRUE)
    base[ord[seq_len(rem_n)]] <- base[ord[seq_len(rem_n)]] + 1L
  }
  as.integer(base)
}

resample_smote_like <- function(x, y, n_to_generate, k = 5, weights = NULL) {
  y <- as.integer(y)
  min_idx <- which(y == 1)
  if (length(min_idx) < 2 || n_to_generate <= 0) {
    return(list(x = x, y = y, synthetic = rep(FALSE, length(y)), origin = rep("original", length(y))))
  }
  
  x_min <- x[min_idx, , drop = FALSE]
  k_eff <- min(k, nrow(x_min) - 1)
  kn <- FNN::get.knn(x_min, k = k_eff)
  
  if (is.null(weights)) weights <- rep(1 / length(min_idx), length(min_idx))
  weights <- weights / sum(weights)
  counts <- allocate_counts(weights, n_to_generate)
  
  synth <- list(); row_id <- 1L
  for (i in seq_along(min_idx)) {
    if (counts[i] == 0) next
    for (j in seq_len(counts[i])) {
      nb_local <- sample(kn$nn.index[i, ], size = 1)
      x_i <- x_min[i, , drop = FALSE]
      x_z <- x_min[nb_local, , drop = FALSE]
      lam <- runif(1)
      synth[[row_id]] <- x_i + lam * (x_z - x_i)
      row_id <- row_id + 1L
    }
  }
  synth_mat <- do.call(rbind, synth)
  list(
    x = rbind(x, synth_mat),
    y = c(y, rep(1L, nrow(synth_mat))),
    synthetic = c(rep(FALSE, nrow(x)), rep(TRUE, nrow(synth_mat))),
    origin = c(rep("original", nrow(x)), rep("synthetic", nrow(synth_mat)))
  )
}

resample_smote <- function(x, y, target_ratio = 1, k = 5) {
  n_min <- sum(y == 1); n_maj <- sum(y == 0)
  n_to_generate <- max(0, ceiling(target_ratio * n_maj) - n_min)
  resample_smote_like(x, y, n_to_generate = n_to_generate, k = k)
}

resample_da_smote <- function(x, y, target_ratio = 1, k = 5) {
  min_idx <- which(y == 1)
  x_min <- x[min_idx, , drop = FALSE]
  if (nrow(x_min) < 2) return(resample_smote(x, y, target_ratio = target_ratio, k = k))
  k_eff <- min(k, nrow(x_min) - 1)
  kn <- FNN::get.knn(x_min, k = k_eff)
  sparsity <- rowMeans(kn$nn.dist)
  weights <- sparsity / sum(sparsity)
  n_min <- sum(y == 1); n_maj <- sum(y == 0)
  n_to_generate <- max(0, ceiling(target_ratio * n_maj) - n_min)
  resample_smote_like(x, y, n_to_generate = n_to_generate, k = k, weights = weights)
}

resample_adasyn <- function(x, y, target_ratio = 1, k = 5) {
  y <- as.integer(y)
  min_idx <- which(y == 1)
  if (length(min_idx) < 2) return(resample_smote(x, y, target_ratio = target_ratio, k = k))
  n_min <- sum(y == 1); n_maj <- sum(y == 0)
  n_to_generate <- max(0, ceiling(target_ratio * n_maj) - n_min)
  if (n_to_generate <= 0) return(list(x = x, y = y, synthetic = rep(FALSE, length(y)), origin = rep("original", length(y))))
  
  k_eff <- min(k + 1, nrow(x))
  kn_all <- FNN::get.knnx(data = x, query = x[min_idx, , drop = FALSE], k = k_eff)$nn.index
  ri <- numeric(length(min_idx))
  for (i in seq_along(min_idx)) {
    neigh <- kn_all[i, ]
    neigh <- neigh[neigh != min_idx[i]][1:k]
    ri[i] <- mean(y[neigh] == 0)
  }
  if (sum(ri) == 0) ri <- rep(1, length(ri))
  weights <- ri / sum(ri)
  resample_smote_like(x, y, n_to_generate = n_to_generate, k = k, weights = weights)
}

resample_borderline_smote <- function(x, y, target_ratio = 1, k = 5) {
  y <- as.integer(y)
  min_idx <- which(y == 1)
  if (length(min_idx) < 2) return(resample_smote(x, y, target_ratio = target_ratio, k = k))
  n_min <- sum(y == 1); n_maj <- sum(y == 0)
  n_to_generate <- max(0, ceiling(target_ratio * n_maj) - n_min)
  if (n_to_generate <= 0) return(list(x = x, y = y, synthetic = rep(FALSE, length(y)), origin = rep("original", length(y))))
  
  k_eff <- min(k + 1, nrow(x))
  kn_all <- FNN::get.knnx(data = x, query = x[min_idx, , drop = FALSE], k = k_eff)$nn.index
  danger <- logical(length(min_idx))
  for (i in seq_along(min_idx)) {
    neigh <- kn_all[i, ]
    neigh <- neigh[neigh != min_idx[i]][1:k]
    maj_count <- sum(y[neigh] == 0)
    danger[i] <- maj_count >= 1 && maj_count < k
  }
  weights <- if (sum(danger) == 0) rep(1, length(min_idx)) else as.numeric(danger)
  weights <- weights / sum(weights)
  resample_smote_like(x, y, n_to_generate = n_to_generate, k = k, weights = weights)
}

compute_tomek_pairs <- function(x, y) {
  if (nrow(x) < 2) return(matrix(integer(0), ncol = 2))
  kn1 <- FNN::get.knn(x, k = 2)$nn.index[, 2]
  pairs <- list(); t <- 1L
  for (i in seq_along(kn1)) {
    j <- kn1[i]
    if (!is.na(j) && kn1[j] == i && y[i] != y[j] && i < j) {
      pairs[[t]] <- c(i, j)
      t <- t + 1L
    }
  }
  if (length(pairs) == 0) matrix(integer(0), ncol = 2) else do.call(rbind, pairs)
}

apply_tomek_clean <- function(x, y, synthetic = rep(FALSE, length(y)), origin = rep("original", length(y))) {
  pairs <- compute_tomek_pairs(x, y)
  if (nrow(pairs) == 0) return(list(x = x, y = y, synthetic = synthetic, origin = origin))
  remove_idx <- integer(0)
  for (r in seq_len(nrow(pairs))) {
    pair <- pairs[r, ]
    cls <- y[pair]
    if (all(cls %in% c(0,1))) {
      maj <- pair[which(cls == 0)]
      remove_idx <- c(remove_idx, maj[1])
    }
  }
  keep <- setdiff(seq_len(nrow(x)), unique(remove_idx))
  list(x = x[keep, , drop = FALSE], y = y[keep], synthetic = synthetic[keep], origin = origin[keep])
}

apply_enn_clean <- function(x, y, k = 5, synthetic = rep(FALSE, length(y)), origin = rep("original", length(y))) {
  if (nrow(x) <= k) return(list(x = x, y = y, synthetic = synthetic, origin = origin))
  kn <- compute_knn_excluding_self(x, k)
  pred <- apply(kn$index, 1, function(idx) mode_binary(y[idx]))
  keep <- pred == y
  list(x = x[keep, , drop = FALSE], y = y[keep], synthetic = synthetic[keep], origin = origin[keep])
}

apply_penn_clean <- function(x, y, synthetic = rep(FALSE, length(y)), beta = 0.15, k = 5, origin = rep("original", length(y))) {
  if (nrow(x) <= k) {
    return(list(x = x, y = y, synthetic = synthetic, origin = origin))
  }
  
  y <- as.integer(y)
  kn <- compute_knn_excluding_self(x, k)
  
  disagreement <- numeric(nrow(x))
  for (i in seq_len(nrow(x))) {
    neighbor_idx <- kn$index[i, ]
    disagreement[i] <- mean(y[neighbor_idx] != y[i])
  }
  
  keep <- rep(TRUE, length(y))
  for (i in seq_along(y)) {
    if (y[i] == 0) {
      if (disagreement[i] > 0.5) { keep[i] <- FALSE }
    } else if (y[i] == 1) {
      if (synthetic[i]) {
        if (disagreement[i] > 0.5) { keep[i] <- FALSE }
      } else {
        if (disagreement[i] > 0.5 + beta) { keep[i] <- FALSE }
      }
    }
  }
  
  list(x = x[keep, , drop = FALSE], y = y[keep], synthetic = synthetic[keep], origin = origin[keep])
}

resample_ld_smote <- function(x, y, target_ratio = 1, k = 5) {
  y <- as.integer(y)
  min_idx <- which(y == 1)
  n_min <- sum(y == 1); n_maj <- sum(y == 0)
  n_to_generate <- max(0, ceiling(target_ratio * n_maj) - n_min)
  if (length(min_idx) < 3 || n_to_generate <= 0) return(resample_smote(x, y, target_ratio = target_ratio, k = k))
  
  x_min <- x[min_idx, , drop = FALSE]
  k_eff <- min(k, nrow(x_min) - 1)
  kn <- FNN::get.knn(x_min, k = k_eff)
  
  avg_dist <- rowMeans(kn$nn.dist)
  ad_med <- median(avg_dist)
  rho <- exp(-(avg_dist^2) / (2 * (ad_med^2 + 1e-12)))
  weights <- rho / sum(rho)
  counts <- allocate_counts(weights, n_to_generate)
  
  synth <- list(); row_id <- 1L
  nn_idx <- kn$nn.index
  for (i in seq_along(min_idx)) {
    if (counts[i] == 0) next
    neighs <- nn_idx[i, ]
    if (length(neighs) < 2) next
    for (j in seq_len(counts[i])) {
      pick <- sample(neighs, size = 2, replace = FALSE)
      u <- runif(1); v <- runif(1)
      if (u + v > 1) { u <- 1 - u; v <- 1 - v }
      x_i <- x_min[i, , drop = FALSE]
      x_a <- x_min[pick[1], , drop = FALSE]
      x_b <- x_min[pick[2], , drop = FALSE]
      synth[[row_id]] <- x_i + u * (x_a - x_i) + v * (x_b - x_i)
      row_id <- row_id + 1L
    }
  }
  synth <- Filter(Negate(is.null), synth)
  if (length(synth) == 0) return(resample_smote(x, y, target_ratio = target_ratio, k = k))
  synth_mat <- do.call(rbind, synth)
  list(
    x = rbind(x, synth_mat),
    y = c(y, rep(1L, nrow(synth_mat))),
    synthetic = c(rep(FALSE, nrow(x)), rep(TRUE, nrow(synth_mat))),
    origin = c(rep("original", nrow(x)), rep("synthetic", nrow(synth_mat)))
  )
}

apply_method <- function(method, x, y, params) {
  base <- list(x = x, y = y, synthetic = rep(FALSE, length(y)), origin = rep("original", length(y)))
  if (method == "baseline") return(base)
  if (method == "smote") return(resample_smote(x, y, target_ratio = params$target_ratio, k = params$k_smote))
  if (method == "adasyn") return(resample_adasyn(x, y, target_ratio = params$target_ratio, k = params$k_smote))
  if (method == "borderline_smote") return(resample_borderline_smote(x, y, target_ratio = params$target_ratio, k = params$k_smote))
  if (method == "smote_tomek") {
    rs <- resample_smote(x, y, target_ratio = params$target_ratio, k = params$k_smote)
    return(apply_tomek_clean(rs$x, rs$y, rs$synthetic, rs$origin))
  }
  if (method == "smote_enn") {
    rs <- resample_smote(x, y, target_ratio = params$target_ratio, k = params$k_smote)
    return(apply_enn_clean(rs$x, rs$y, k = params$k_enn, synthetic = rs$synthetic, origin = rs$origin))
  }
  if (method == "da_smote") return(resample_da_smote(x, y, target_ratio = params$target_ratio, k = params$k_smote))
  if (method == "penn") return(apply_penn_clean(x, y, beta = params$beta, k = params$k_enn))
  if (method == "ld_smote") return(resample_ld_smote(x, y, target_ratio = params$target_ratio, k = params$k_ld))
  if (method == "da_smote_penn") {
    rs <- resample_da_smote(x, y, target_ratio = params$target_ratio, k = params$k_smote)
    return(apply_penn_clean(rs$x, rs$y, synthetic = rs$synthetic, beta = params$beta, k = params$k_enn, origin = rs$origin))
  }
  stop("Unknown method: ", method)
}

classifier_grids <- list(
  knn = tibble(k = c(3L, 5L, 7L, 11L)),
  logit = tibble(dummy = 1L),
  svm_rbf = expand.grid(cost = c(0.1, 1, 10), gamma = c(0.01, 0.1, 1), stringsAsFactors = FALSE) %>% as_tibble(),
  rf = expand.grid(num.trees = c(500L), mtry_frac = c(0.2, 0.4, 0.6), min.node.size = c(1L, 5L), stringsAsFactors = FALSE) %>% as_tibble(),
  xgb = expand.grid(max_depth = c(3L, 6L), eta = c(0.03, 0.10), nrounds = c(150L, 300L), subsample = c(0.8), colsample_bytree = c(0.8), min_child_weight = c(1), stringsAsFactors = FALSE) %>% as_tibble()
)

train_predict_classifier <- function(classifier, x_train, y_train, x_test, params) {
  if (length(unique(y_train)) < 2) return(rep(0.5, nrow(x_test)))
  
  y_train_factor <- factor(y_train, levels = c(0,1), labels = c("neg","pos"))
  
  if (classifier == "knn") {
    k_val <- min(ifelse(is.null(params$k), 3, params$k), nrow(x_train))
    if (k_val < 1) k_val <- 1
    pred <- tryCatch(class::knn(train = x_train, test = x_test, cl = y_train_factor, k = k_val, prob = TRUE), error = function(e) NULL)
    if (is.null(pred)) return(rep(0.5, nrow(x_test)))
    win_prob <- attr(pred, "prob")
    if (is.null(win_prob)) win_prob <- rep(0.5, length(pred))
    return(as.numeric(ifelse(pred == "pos", as.numeric(win_prob), 1 - as.numeric(win_prob))))
  }
  
  if (classifier == "logit") {
    df_train <- as.data.frame(x_train); df_train$y <- y_train
    df_test <- as.data.frame(x_test)
    fit <- tryCatch(glm(y ~ ., data = df_train, family = binomial()), error = function(e) NULL)
    if (is.null(fit)) return(rep(0.5, nrow(x_test)))
    return(as.numeric(predict(fit, newdata = df_test, type = "response")))
  }
  
  if (classifier == "svm_rbf") {
    fit <- tryCatch(e1071::svm(x = x_train, y = y_train_factor, kernel = "radial", probability = TRUE, cost = params$cost, gamma = params$gamma, scale = FALSE), error = function(e) NULL)
    if (is.null(fit)) return(rep(0.5, nrow(x_test)))
    pred <- predict(fit, x_test, probability = TRUE)
    probs <- attr(pred, "probabilities")
    if (!is.null(probs) && "pos" %in% colnames(probs)) return(as.numeric(probs[, "pos"]))
    return(rep(0.5, nrow(x_test)))
  }
  
  if (classifier == "rf") {
    df_train <- as.data.frame(x_train); df_train$y <- y_train_factor
    df_test <- as.data.frame(x_test)
    mtry <- max(1, floor(params$mtry_frac * ncol(x_train)))
    fit <- tryCatch(ranger::ranger(y ~ ., data = df_train, probability = TRUE, num.trees = params$num.trees, mtry = mtry, min.node.size = params$min.node.size, seed = get_next_seed()), error = function(e) NULL)
    if (is.null(fit)) return(rep(0.5, nrow(x_test)))
    pred <- tryCatch(predict(fit, data = df_test), error = function(e) NULL)
    if (is.null(pred) || is.null(pred$predictions)) return(rep(0.5, nrow(x_test)))
    pred_mat <- pred$predictions
    if ("pos" %in% colnames(pred_mat)) return(as.numeric(pred_mat[, "pos"]))
    return(rep(0.5, nrow(x_test)))
  }
  
  if (classifier == "xgb") {
    y_train_num <- as.numeric(as.character(y_train))
    dtrain <- tryCatch(xgboost::xgb.DMatrix(data = as.matrix(x_train), label = y_train_num), error = function(e) NULL)
    if (is.null(dtrain)) return(rep(0.5, nrow(x_test)))
    fit <- tryCatch(xgboost::xgb.train(
      data = dtrain,
      params = list(objective = "binary:logistic", eval_metric = "logloss", base_score = 0.5,
                    max_depth = as.integer(params$max_depth), eta = as.numeric(params$eta),
                    subsample = as.numeric(params$subsample), colsample_bytree = as.numeric(params$colsample_bytree),
                    min_child_weight = as.numeric(params$min_child_weight), nthread = 1),
      nrounds = as.integer(params$nrounds), verbose = 0
    ), error = function(e) NULL)
    if (is.null(fit)) return(rep(0.5, nrow(x_test)))
    dtest <- tryCatch(xgboost::xgb.DMatrix(data = as.matrix(x_test)), error = function(e) NULL)
    if (is.null(dtest)) return(rep(0.5, nrow(x_test)))
    return(as.numeric(predict(fit, dtest)))
  }
  
  stop("Unknown classifier: ", classifier)
}

score_parameter_row <- function(classifier, param_row, x, y, v = 3, seed = 1) {
  df <- data.frame(y = as.factor(y))
  rownames(df) <- seq_len(nrow(df))
  set.seed(seed)
  v_adj <- min(v, sum(y == 1))
  if (v_adj < 2) v_adj <- 2
  folds <- tryCatch(rsample::vfold_cv(df, v = v_adj, strata = y), error = function(e) NULL)
  if (is.null(folds) || nrow(folds) == 0) return(NA_real_)
  
  scores <- numeric(nrow(folds))
  for (i in seq_len(nrow(folds))) {
    split <- folds$splits[[i]]
    tr_idx <- as.integer(rownames(rsample::analysis(split)))
    te_idx <- as.integer(rownames(rsample::assessment(split)))
    if (length(unique(y[tr_idx])) < 2 || length(unique(y[te_idx])) < 2) { scores[i] <- NA_real_; next }
    prob <- tryCatch(train_predict_classifier(classifier, x[tr_idx, , drop = FALSE], y[tr_idx], x[te_idx, , drop = FALSE], as.list(param_row)), error = function(e) NA_real_)
    if (is.numeric(prob) && length(prob) == length(te_idx)) {
      met <- tryCatch(compute_metrics(y[te_idx], prob), error = function(e) NULL)
      if (!is.null(met) && !is.na(met[[config$primary_metric]][1])) scores[i] <- met[[config$primary_metric]][1] else scores[i] <- NA_real_
    } else { scores[i] <- NA_real_ }
  }
  mean(scores, na.rm = TRUE)
}

tune_classifier_params <- function(classifier, x, y) {
  grid <- classifier_grids[[classifier]]
  if (!config$tune_classifiers || nrow(grid) == 1) return(as.list(grid[1,]))
  scores <- purrr::map_dbl(seq_len(nrow(grid)), function(i) {
    score_parameter_row(classifier, grid[i,], x, y, v = config$inner_folds, seed = get_next_seed())
  })
  scores[is.na(scores) | is.nan(scores)] <- -Inf
  best_idx <- which.max(scores)
  if (length(best_idx) == 0 || is.na(best_idx)) best_idx <- 1
  as.list(grid[best_idx, , drop = FALSE])
}

create_outer_folds <- function(y, v = 5, repeats = 5, seed = 1) {
  set.seed(seed)
  df <- data.frame(y = y)
  tryCatch(rsample::vfold_cv(df, v = v, repeats = repeats, strata = y),
           error = function(e) rsample::vfold_cv(df, v = v, repeats = repeats))
}

benchmark_method_params <- function() {
  list(
    target_ratio = config$target_ratio,
    beta = config$benchmark_beta,
    k_smote = config$benchmark_k_smote,
    k_enn = config$benchmark_k_enn,
    k_ld = config$benchmark_k_ld
  )
}

# ==============================================================================
# 6. BENCHMARK EXECUTION FUNCTIONS
# ==============================================================================
summarise_dataset_manifest <- function() {
  out <- purrr::pmap_dfr(dataset_manifest, function(dataset, file, target, positive_class) {
    path <- file.path(config$data_dir, file)
    if (!file.exists(path)) return(tibble(dataset = dataset, N = NA_integer_, minority_count = NA_integer_, majority_count = NA_integer_, predictors = NA_integer_, data_type = NA_character_))
    df <- fread(path, data.table = FALSE)
    y <- as.integer(df[[target]] == positive_class)
    tibble(dataset = dataset, N = nrow(df), minority_count = sum(y == 1), majority_count = sum(y == 0), predictors = ncol(df) - 1, data_type = ifelse(any(vapply(df, function(z) is.character(z) || is.factor(z), logical(1))), "mixed", "numeric"))
  })
  write.csv(out, file.path(config$results_dir, "dataset_summary.csv"), row.names = FALSE)
  out
}

run_realworld_benchmark <- function() {
  method_params <- benchmark_method_params()
  all_results <- list(); counter <- 1L; param_log <- list(); param_counter <- 1L
  
  for (r in seq_len(nrow(dataset_manifest))) {
    info <- dataset_manifest[r, ]
    path <- file.path(config$data_dir, info$file)
    if (!file.exists(path)) { message("Skipping missing file: ", path); next }
    raw_df <- fread(path, data.table = FALSE)
    raw_df[[info$target]] <- as.character(raw_df[[info$target]])
    y_all <- as.integer(raw_df[[info$target]] == info$positive_class)
    folds <- create_outer_folds(y_all, v = config$outer_folds, repeats = config$outer_repeats, seed = get_next_seed())
    
    for (s in seq_len(nrow(folds))) {
      tr_idx <- as.integer(rownames(rsample::analysis(folds$splits[[s]])))
      te_idx <- as.integer(rownames(rsample::assessment(folds$splits[[s]])))
      prep <- make_design_matrices(raw_df[tr_idx, , drop = FALSE], raw_df[te_idx, , drop = FALSE], info$target, info$positive_class)
      
      tuned_params <- lapply(config$classifiers, function(clf) tune_classifier_params(clf, prep$x_train, prep$y_train))
      names(tuned_params) <- config$classifiers
      
      for (clf in config$classifiers) {
        param_log[[param_counter]] <- tibble(dataset = info$dataset, split = s, classifier = clf, parameters = paste(names(tuned_params[[clf]]), unlist(tuned_params[[clf]]), sep = "=", collapse = "; "))
        param_counter <- param_counter + 1L
      }
      
      for (method in config$methods) {
        t0 <- Sys.time()
        rs <- apply_method(method, prep$x_train, prep$y_train, method_params)
        runtime <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        for (clf in config$classifiers) {
          prob <- train_predict_classifier(clf, rs$x, rs$y, prep$x_test, tuned_params[[clf]])
          met <- compute_metrics(prep$y_test, prob, threshold = config$decision_threshold)
          all_results[[counter]] <- tibble(dataset = info$dataset, split = s, method = method, classifier = clf, runtime_seconds = runtime, n_train_original = nrow(prep$x_train), n_train_final = nrow(rs$x)) %>% bind_cols(met)
          counter <- counter + 1L
        }
      }
    }
  }
  fold_results <- bind_rows(all_results)
  write.csv(fold_results, file.path(config$results_dir, "fold_level_metrics_realworld.csv"), row.names = FALSE)
  write.csv(bind_rows(param_log), file.path(config$results_dir, "selected_classifier_parameters.csv"), row.names = FALSE)
  fold_results
}

generate_simulation_dataset <- function(n_min, n_maj, seed = 1) {
  set.seed(seed)
  mu <- runif(1, 3.2, 4.2); sigma2 <- runif(1, 0.6, 1.0); rho <- runif(1, 0.1, 0.3)
  Sigma <- matrix(c(sigma2, rho * sigma2, rho * sigma2, sigma2), 2, 2)
  cholS <- chol(Sigma)
  rmvnorm2 <- function(n, mean_vec) { z <- matrix(rnorm(n * 2), ncol = 2); sweep(z %*% cholS, 2, mean_vec, `+`) }
  x_min <- rmvnorm2(n_min, c(0.75 * mu, mu))
  x_maj <- rmvnorm2(n_maj, c(mu, 0.75 * mu))
  x <- rbind(x_min, x_maj); y <- c(rep(1L, n_min), rep(0L, n_maj))
  idx <- sample(seq_along(y))
  list(x = x[idx, , drop = FALSE], y = y[idx])
}

run_simulation_benchmark <- function() {
  method_params <- benchmark_method_params()
  out <- list(); ctr <- 1L
  
  for (ratio in config$simulation_ratios) {
    n_min <- ratio[1]; n_maj <- ratio[2]; ratio_name <- paste0(n_min, ":", n_maj)
    for (rep_id in seq_len(config$n_sim_replications)) {
      dat <- generate_simulation_dataset(n_min, n_maj, seed = get_next_seed())
      df <- data.frame(y = dat$y)
      v_sim <- min(5, n_min); if (v_sim < 2) v_sim <- 2
      folds <- tryCatch(rsample::vfold_cv(df, v = v_sim, strata = y), error = function(e) NULL)
      if (is.null(folds)) next
      
      for (s in seq_len(nrow(folds))) {
        tr_idx <- as.integer(rownames(rsample::analysis(folds$splits[[s]])))
        te_idx <- as.integer(rownames(rsample::assessment(folds$splits[[s]])))
        if (length(unique(dat$y[tr_idx])) < 2 || length(unique(dat$y[te_idx])) < 2) next
        
        x_train <- scale(dat$x[tr_idx, , drop = FALSE])
        center <- attr(x_train, "scaled:center"); scalev <- attr(x_train, "scaled:scale")
        y_train <- dat$y[tr_idx]
        x_test <- scale(dat$x[te_idx, , drop = FALSE], center = center, scale = scalev)
        y_test <- dat$y[te_idx]
        
        tuned_params <- lapply(config$classifiers, function(clf) tune_classifier_params(clf, x_train, y_train))
        names(tuned_params) <- config$classifiers
        
        for (method in config$methods) {
          t0 <- Sys.time()
          rs <- apply_method(method, x_train, y_train, method_params)
          runtime <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
          for (clf in config$classifiers) {
            prob <- tryCatch(train_predict_classifier(clf, rs$x, rs$y, x_test, tuned_params[[clf]]), error = function(e) rep(0.5, length(y_test)))
            met <- compute_metrics(y_test, prob)
            out[[ctr]] <- tibble(imbalance_ratio = ratio_name, replication = rep_id, split = s, method = method, classifier = clf, runtime_seconds = runtime) %>% bind_cols(met)
            ctr <- ctr + 1L
          }
        }
      }
    }
  }
  sim_results <- bind_rows(out)
  write.csv(sim_results, file.path(config$results_dir, "fold_level_metrics_simulation.csv"), row.names = FALSE)
  sim_results
}

run_sensitivity_analysis <- function() {
  sens_out <- list(); ctr <- 1L
  for (r in seq_len(nrow(dataset_manifest))) {
    info <- dataset_manifest[r, ]
    path <- file.path(config$data_dir, info$file)
    if (!file.exists(path)) next
    raw_df <- fread(path, data.table = FALSE)
    y_all <- as.integer(raw_df[[info$target]] == info$positive_class)
    folds <- create_outer_folds(y_all, v = config$outer_folds, repeats = config$outer_repeats, seed = get_next_seed())
    
    for (s in seq_len(nrow(folds))) {
      tr_idx <- as.integer(rownames(rsample::analysis(folds$splits[[s]])))
      te_idx <- as.integer(rownames(rsample::assessment(folds$splits[[s]])))
      prep <- make_design_matrices(raw_df[tr_idx, , drop = FALSE], raw_df[te_idx, , drop = FALSE], info$target, info$positive_class)
      tuned_params <- lapply(config$classifiers, function(clf) tune_classifier_params(clf, prep$x_train, prep$y_train))
      names(tuned_params) <- config$classifiers
      
      for (alpha in config$alpha_grid) {
        for (beta in config$beta_grid) {
          for (k_s in config$k_smote_grid) {
            for (k_e in config$k_enn_grid) {
              rs <- apply_method(
                "da_smote_penn",
                prep$x_train,
                prep$y_train,
                list(target_ratio = alpha, beta = beta, k_smote = k_s, k_enn = k_e, k_ld = config$benchmark_k_ld)
              )
              
              for (clf in config$classifiers) {
                prob <- tryCatch(
                  train_predict_classifier(clf, rs$x, rs$y, prep$x_test, tuned_params[[clf]]),
                  error = function(e) rep(0.5, length(prep$y_test))
                )
                met <- compute_metrics(prep$y_test, prob, threshold = config$decision_threshold)
                sens_out[[ctr]] <- tibble(
                  dataset = info$dataset,
                  split = s,
                  classifier = clf,
                  alpha = alpha,
                  beta = beta,
                  k_smote = k_s,
                  k_enn = k_e,
                  primary_metric = met[[config$primary_metric]][1]
                )
                ctr <- ctr + 1L
              }
            }
          }
        }
      }
    }
  }
  sens <- bind_rows(sens_out)
  write.csv(sens, file.path(config$results_dir, "sensitivity_analysis.csv"), row.names = FALSE)
  sens
}

# ==============================================================================
# 7. REPORTING & SUMMARIZATION
# ==============================================================================
summarise_results <- function(real_results, sim_results, sensitivity_results) {
  real_summary <- real_results %>% group_by(dataset, method, classifier) %>%
    summarise(across(c(TP, TN, FP, FN, Sensitivity, Specificity, Precision, F1, Gmean, BalancedAccuracy, MCC, AUPRC, AUROC, runtime_seconds, n_train_original, n_train_final), list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE))), .groups = "drop")
  write.csv(real_summary, file.path(config$results_dir, "realworld_summary.csv"), row.names = FALSE)
  
  sim_summary <- sim_results %>% group_by(imbalance_ratio, method) %>%
    summarise(across(c(Sensitivity, Specificity, Precision, F1, Gmean, BalancedAccuracy, MCC, AUPRC, AUROC, runtime_seconds), list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE))), .groups = "drop")
  write.csv(sim_summary, file.path(config$results_dir, "simulation_summary.csv"), row.names = FALSE)
  
  ablation_summary <- real_results %>% filter(method %in% c("smote", "da_smote", "smote_enn", "penn", "da_smote_penn")) %>%
    group_by(method, classifier) %>% summarise(across(c(Gmean, BalancedAccuracy, MCC, AUPRC, AUROC), mean, na.rm = TRUE), .groups = "drop")
  write.csv(ablation_summary, file.path(config$results_dir, "ablation_summary.csv"), row.names = FALSE)
  
  sensitivity_summary <- sensitivity_results %>% group_by(alpha, beta, k_smote, k_enn) %>%
    summarise(primary_metric = mean(primary_metric, na.rm = TRUE), n_dataset_classifier_splits = dplyr::n(), .groups = "drop") %>% arrange(desc(primary_metric))
  write.csv(sensitivity_summary, file.path(config$results_dir, "sensitivity_summary.csv"), row.names = FALSE)
  
  dataset_summary <- summarise_dataset_manifest()
  invisible(list(real_summary = real_summary, sim_summary = sim_summary, ablation_summary = ablation_summary, sensitivity_summary = sensitivity_summary, dataset_summary = dataset_summary))
}

compute_average_ranks <- function(real_results, metric = "AUPRC") {
  ranked <- real_results %>% group_by(dataset, classifier, method) %>%
    summarise(metric_value = mean(.data[[metric]], na.rm = TRUE), .groups = "drop") %>%
    group_by(dataset, classifier) %>% mutate(rank = rank(-metric_value, ties.method = "average")) %>% ungroup()
  avg_ranks <- ranked %>% group_by(classifier, method) %>% summarise(avg_rank = mean(rank, na.rm = TRUE), .groups = "drop") %>% arrange(classifier, avg_rank)
  write.csv(ranked, file.path(config$results_dir, paste0("ranks_", metric, ".csv")), row.names = FALSE)
  write.csv(avg_ranks, file.path(config$results_dir, paste0("average_ranks_", metric, ".csv")), row.names = FALSE)
  avg_ranks
}

run_statistical_tests <- function(real_results, metric = "AUPRC", reference_method = "da_smote_penn") {
  ds <- real_results %>% group_by(dataset, classifier, method) %>% summarise(metric_value = mean(.data[[metric]], na.rm = TRUE), .groups = "drop")
  friedman_out <- list(); holm_out <- list(); fc <- 1L; hc <- 1L
  
  for (clf in unique(ds$classifier)) {
    wide <- ds %>% filter(classifier == clf) %>% select(dataset, method, metric_value) %>% tidyr::pivot_wider(names_from = method, values_from = metric_value)
    method_cols <- setdiff(names(wide), "dataset")
    mat <- as.matrix(wide[, method_cols, drop = FALSE])
    if (nrow(mat) < 2) next
    
    fr <- suppressWarnings(friedman.test(mat))
    friedman_out[[fc]] <- tibble(classifier = clf, metric = metric, statistic = as.numeric(fr$statistic), p_value = fr$p.value)
    fc <- fc + 1L
    
    if (!reference_method %in% names(wide)) next
    others <- setdiff(method_cols, reference_method)
    pair_rows <- list(); pc <- 1L
    for (m in others) {
      test_df <- wide %>% select(dataset, all_of(reference_method), all_of(m)) %>% tidyr::drop_na()
      if (nrow(test_df) < 2) next
      wt <- suppressWarnings(wilcox.test(test_df[[reference_method]], test_df[[m]], paired = TRUE, alternative = "two.sided", exact = FALSE))
      clean_m <- str_to_title(gsub("_", " ", m))
      pair_rows[[pc]] <- tibble(classifier = clf, metric = metric, comparison = paste("DA-SMOTE-PENN vs", clean_m), statistic = unname(wt$statistic), p_raw = wt$p.value)
      pc <- pc + 1L
    }
    if (length(pair_rows)) {
      pair_df <- bind_rows(pair_rows)
      pair_df$p_holm <- p.adjust(pair_df$p_raw, method = "holm")
      holm_out[[hc]] <- pair_df
      hc <- hc + 1L
    }
  }
  friedman_df <- bind_rows(friedman_out); holm_df <- bind_rows(holm_out)
  write.csv(friedman_df, file.path(config$results_dir, paste0("friedman_", metric, ".csv")), row.names = FALSE)
  write.csv(holm_df, file.path(config$results_dir, paste0("holm_posthoc_", metric, ".csv")), row.names = FALSE)
  invisible(list(friedman = friedman_df, holm = holm_df))
}

# ==============================================================================
# LATEX TABLE GENERATION HELPERS
# ==============================================================================
escape_latex <- function(x) {
  x <- as.character(x)
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub("&", "\\\\&", x)
  x <- gsub("%", "\\\\%", x)
  x <- gsub("_", "\\\\_", x)
  x <- gsub("#", "\\\\#", x)
  x <- gsub("\\$", "\\\\\\$", x)
  x <- gsub("\\{", "\\\\{", x)
  x <- gsub("\\}", "\\\\}", x)
  return(x)
}

safe_escape <- function(x) {
  x_char <- as.character(x)
  if (grepl("^\\\\", x_char)) return(x_char)
  escape_latex(x_char)
}

write_simple_latex_table <- function(df, path, caption = NULL, label = NULL) {
  cols <- names(df)
  cols_escaped <- sapply(cols, safe_escape)
  align <- paste0("l", paste(rep("c", length(cols) - 1), collapse = ""))
  lines <- c("\\begin{table*}[htb]", "\\centering")
  if (!is.null(caption)) lines <- c(lines, paste0("\\caption{", caption, "}"))
  if (!is.null(label)) lines <- c(lines, paste0("\\label{", label, "}"))
  lines <- c(lines, "\\small", paste0("\\begin{tabular}{", align, "}"), "\\toprule", paste(cols_escaped, collapse = " & "), "\\\\", "\\midrule")
  
  for (i in seq_len(nrow(df))) {
    vals <- vapply(df[i, ], function(z) { 
      if (is.numeric(z)) sprintf("%.3f", z) else safe_escape(z)
    }, character(1))
    lines <- c(lines, paste(vals, collapse = " & "), "\\\\")
  }
  lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table*}")
  writeLines(lines, path)
}

write_manuscript_support_tables <- function(real_results, sim_results, sensitivity_results, dataset_summary, stat_tests) {
  friedman_df <- stat_tests$friedman; holm_df <- stat_tests$holm
  
  # 1. Dataset Summary Table
  ds_tex <- dataset_summary %>% 
    mutate(Dataset = dataset, `N` = N, `Minority count` = minority_count, `Majority count` = majority_count, 
           `Predictors` = predictors, `Data type` = data_type, `Source` = "KEEL/UCI") %>% 
    select(Dataset, `N`, `Minority count`, `Majority count`, `Predictors`, `Data type`, `Source`)
  write_simple_latex_table(ds_tex, file.path(config$table_dir, "tab_datasets.tex"), 
                           caption = "Real-world datasets used in the strengthened benchmark.", label = "tab:datasets")
  
  # 2. Main Real-World Results Table
  main_rw <- real_results %>% 
    group_by(method, classifier) %>% 
    summarise(metric_val = mean(.data[[config$primary_metric]], na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = classifier, values_from = metric_val) %>%
    mutate(Method = dplyr::case_when(
      method == "baseline" ~ "Baseline", method == "smote" ~ "SMOTE", method == "adasyn" ~ "ADASYN",
      method == "borderline_smote" ~ "Borderline-SMOTE", method == "smote_tomek" ~ "SMOTE-Tomek",
      method == "smote_enn" ~ "SMOTE-ENN", method == "da_smote" ~ "DA-SMOTE", method == "penn" ~ "PENN",
      method == "ld_smote" ~ "LD-SMOTE", method == "da_smote_penn" ~ "\\textbf{DA-SMOTE-PENN}"
    )) %>%
    select(Method, any_of(c("knn", "logit", "svm_rbf", "rf", "xgb")))
  .ren <- c(knn = "k-NN", logit = "LogReg", svm_rbf = "RBF-SVM", rf = "RF", xgb = "XGB")
  .cls <- setdiff(names(main_rw), "Method")
  names(main_rw)[-1] <- unname(.ren[.cls])
  write_simple_latex_table(main_rw, file.path(config$table_dir, "tab_main_realworld.tex"), 
                           caption = paste0("Primary full-dimensional real-world benchmark (mean ", config$primary_metric, ")."), label = "tab:main_realworld")
  
  # 3. Dataset-level AUPRC Table
  dataset_auprc <- real_results %>%
    group_by(dataset, method) %>%
    summarise(mean_auprc = mean(AUPRC, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = method, values_from = mean_auprc) %>%
    mutate(dataset = str_to_title(gsub("_", " ", dataset)))
  
  col_rename <- c(
    "baseline" = "Base", "smote" = "SMOTE", "adasyn" = "ADASYN",
    "borderline_smote" = "B-SMOTE", "smote_tomek" = "S-Tomek",
    "smote_enn" = "S-ENN", "da_smote" = "DA-S", "penn" = "PENN",
    "ld_smote" = "LD-S", "da_smote_penn" = "\\textbf{DA-S-P}"
  )
  dataset_auprc <- dataset_auprc %>% rename_with(~ col_rename[.x], .cols = any_of(names(col_rename)))
  write_simple_latex_table(dataset_auprc, file.path(config$table_dir, "tab_datasets_auprc.tex"),
                           caption = "Mean AUPRC per dataset and resampling method (averaged across classifiers).",
                           label = "tab:datasets_auprc")
  
  # 4. Simulation Summary Table
  sim_summ <- sim_results %>% 
    group_by(imbalance_ratio, method) %>% 
    summarise(metric_val = mean(.data[[config$primary_metric]], na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = imbalance_ratio, values_from = metric_val) %>%
    mutate(Method = dplyr::case_when(
      method == "baseline" ~ "Baseline", method == "smote" ~ "SMOTE", method == "adasyn" ~ "ADASYN",
      method == "borderline_smote" ~ "Borderline-SMOTE", method == "smote_tomek" ~ "SMOTE-Tomek",
      method == "smote_enn" ~ "SMOTE-ENN", method == "da_smote" ~ "DA-SMOTE", method == "penn" ~ "PENN",
      method == "ld_smote" ~ "LD-SMOTE", method == "da_smote_penn" ~ "\\textbf{DA-SMOTE-PENN}"
    )) %>% 
    select(Method, any_of(c("5:995", "10:990", "100:900", "200:800", "500:500")))
  write_simple_latex_table(sim_summ, file.path(config$table_dir, "tab_simulation_summary.tex"), 
                           caption = paste0("Simulation summary across imbalance ratios (mean ", config$primary_metric, ")."), label = "tab:simulation_summary")
  
  # 5. Ablation Analysis Table
  ablation <- real_results %>% 
    filter(method %in% c("smote", "da_smote", "smote_enn", "penn", "da_smote_penn")) %>%
    group_by(method) %>% 
    summarise(metric_val = mean(.data[[config$primary_metric]], na.rm = TRUE), .groups = "drop") %>%
    mutate(
      Variant = dplyr::case_when(method == "smote" ~ "SMOTE", method == "da_smote" ~ "DA-SMOTE", method == "smote_enn" ~ "SMOTE-ENN", method == "penn" ~ "PENN", method == "da_smote_penn" ~ "\\textbf{DA-SMOTE-PENN}"),
      `Density-aware allocation` = dplyr::case_when(method == "smote" ~ "No", method == "da_smote" ~ "Yes", method == "smote_enn" ~ "No", method == "penn" ~ "No oversampling", method == "da_smote_penn" ~ "Yes"),
      `Protected editing` = dplyr::case_when(method == "smote" ~ "No", method == "da_smote" ~ "No", method == "smote_enn" ~ "No (standard ENN)", method == "penn" ~ "Yes", method == "da_smote_penn" ~ "Yes"),
      `Primary metric` = sprintf("%.3f", metric_val),
      Interpretation = dplyr::case_when(method == "smote" ~ "uniform oversampling baseline", method == "da_smote" ~ "effect of density awareness alone", method == "smote_enn" ~ "effect of post-SMOTE cleaning", method == "penn" ~ "cleaning-only ablation", method == "da_smote_penn" ~ "full proposed pipeline")
    ) %>% 
    select(Variant, `Density-aware allocation`, `Protected editing`, `Primary metric`, Interpretation)
  write_simple_latex_table(ablation, file.path(config$table_dir, "tab_ablation.tex"), 
                           caption = paste0("Ablation analysis of the proposed pipeline (mean ", config$primary_metric, ")."), label = "tab:ablation")
  
  # 6. Sensitivity Analysis Table
  sens_summ <- sensitivity_results %>%
    group_by(alpha, beta, k_smote, k_enn) %>%
    summarise(metric_val = mean(primary_metric, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(metric_val))
  canonical_metric <- sens_summ %>%
    filter(alpha == config$target_ratio,
           beta == config$benchmark_beta,
           k_smote == config$benchmark_k_smote,
           k_enn == config$benchmark_k_enn) %>%
    summarise(metric_val = mean(metric_val, na.rm = TRUE)) %>%
    pull(metric_val)
  if (!length(canonical_metric) || is.na(canonical_metric)) canonical_metric <- NA_real_
  
  sens_tex <- tibble(
    Parameter = c("Target ratio $\\alpha$", "Protection margin $\\beta$", "$k_{\\mathrm{SMOTE}}$", "$k_{\\mathrm{ENN}}$"),
    `Values examined` = c(
      paste(config$alpha_grid, collapse = ", "),
      paste(config$beta_grid, collapse = ", "),
      paste(config$k_smote_grid, collapse = ", "),
      paste(config$k_enn_grid, collapse = ", ")
    ),
    `Canonical benchmark value` = c(config$target_ratio, config$benchmark_beta, config$benchmark_k_smote, config$benchmark_k_enn),
    `Analysis scope` = c(
      "All datasets, all classifiers, repeated outer resampling",
      "All datasets, all classifiers, repeated outer resampling",
      "All datasets, all classifiers, repeated outer resampling",
      "All datasets, all classifiers, repeated outer resampling"
    ),
    `Selection note` = c(
      "Primary benchmark uses the canonical setting; full grid retained for robustness analysis",
      "Primary benchmark uses the canonical setting; full grid retained for robustness analysis",
      "Primary benchmark uses the canonical setting; full grid retained for robustness analysis",
      "Primary benchmark uses the canonical setting; full grid retained for robustness analysis"
    )
  )
  write_simple_latex_table(sens_tex, file.path(config$table_dir, "tab_sensitivity.tex"),
                           caption = paste0("Study-wide sensitivity-analysis design for DA-SMOTE-PENN (canonical-setting mean ", config$primary_metric, " = ", sprintf("%.3f", canonical_metric), ")."),
                           label = "tab:sensitivity")
  
  # 7. Statistical Tests Table
  add_stars <- function(p_val) {
    ifelse(p_val < 0.001, paste0(sprintf("%.3e", p_val), "***"),
           ifelse(p_val < 0.01, paste0(sprintf("%.3e", p_val), "**"),
                  ifelse(p_val < 0.05, paste0(sprintf("%.3e", p_val), "*"),
                         sprintf("%.3e", p_val))))
  }
  
  clean_method_name <- function(m) {
    m <- gsub("_", "-", m)
    m <- gsub("smote", "SMOTE", m, ignore.case = TRUE)
    m <- gsub("adasyn", "ADASYN", m, ignore.case = TRUE)
    m <- gsub("penn", "PENN", m, ignore.case = TRUE)
    m <- gsub("baseline", "Baseline", m, ignore.case = TRUE)
    return(m)
  }
  
  stats_tex <- holm_df %>% 
    mutate(
      clf_name = dplyr::case_when(
        classifier == "knn" ~ "k-NN",
        classifier == "logit" ~ "LogReg",
        classifier == "svm_rbf" ~ "RBF-SVM",
        classifier == "rf" ~ "RF",
        classifier == "xgb" ~ "XGB",
        TRUE ~ classifier
      ),
      `Comparison / test` = paste0("DA-SMOTE-PENN vs ", clean_method_name(gsub("DA-SMOTE-PENN vs ", "", comparison)), " [", clf_name, "]"),
      `Raw p-value` = sprintf("%.3e", p_raw),
      `Corrected p-value` = add_stars(p_holm)
    ) %>% 
    select(`Comparison / test`, Statistic = statistic, `Raw p-value`, `Corrected p-value`)
  
  friedman_rows <- friedman_df %>% 
    mutate(
      clf_name = dplyr::case_when(
        classifier == "knn" ~ "k-NN",
        classifier == "logit" ~ "LogReg",
        classifier == "svm_rbf" ~ "RBF-SVM",
        classifier == "rf" ~ "RF",
        classifier == "xgb" ~ "XGB",
        TRUE ~ classifier
      ),
      `Comparison / test` = paste0("Friedman test (", clf_name, ")"), 
      `Raw p-value` = add_stars(p_value),
      `Corrected p-value` = "N/A"
    ) %>% 
    select(`Comparison / test`, Statistic = statistic, `Raw p-value`, `Corrected p-value`)
  
  full_stats <- bind_rows(friedman_rows, stats_tex)
  
  lines <- c(
    "\\begin{table*}[htb]", "\\centering",
    paste0("\\caption{Nonparametric statistical comparison for ", config$primary_metric, ".}"),
    "\\label{tab:stats}", "\\small",
    "\\begin{tabular}{lccc}", "\\toprule",
    "Comparison / test & Statistic & Raw $p$-value & Corrected $p$-value \\\\", "\\midrule"
  )
  
  for (i in seq_len(nrow(full_stats))) {
    vals <- vapply(full_stats[i, ], function(z) escape_latex(z), character(1))
    lines <- c(lines, paste(vals, collapse = " & "), "\\\\")
  }
  
  lines <- c(lines, 
             "\\bottomrule", "\\end{tabular}",
             "\\begin{tablenotes}", "\\small",
             "\\item \\textit{Note:} * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$. Asterisks are applied to the raw $p$-value for the omnibus Friedman test, and to the Holm-corrected $p$-value for post-hoc pairwise comparisons.",
             "\\end{tablenotes}",
             "\\end{table*}"
  )
  writeLines(lines, file.path(config$table_dir, "tab_stats.tex"))
  
  # 8. Runtime Table
  runtime_tex <- real_results %>% 
    filter(method %in% c("smote", "smote_tomek", "smote_enn", "ld_smote", "da_smote_penn")) %>%
    group_by(method) %>% 
    summarise(runtime = mean(runtime_seconds, na.rm = TRUE), .groups = "drop") %>%
    mutate(
      Method = dplyr::case_when(method == "smote" ~ "SMOTE", method == "smote_tomek" ~ "SMOTE-Tomek", method == "smote_enn" ~ "SMOTE-ENN", method == "ld_smote" ~ "LD-SMOTE", method == "da_smote_penn" ~ "\\textbf{DA-SMOTE-PENN}"),
      `Relative runtime vs SMOTE-ENN` = sprintf("%.2f\\times", runtime / runtime[method == "smote_enn"]),
      `Practical note` = dplyr::case_when(method == "smote" ~ "Fastest baseline", method == "smote_tomek" ~ "Moderate overhead", method == "smote_enn" ~ "Standard hybrid baseline", method == "ld_smote" ~ "Higher due to density estimation", method == "da_smote_penn" ~ "Additional hybrid overhead")
    ) %>% 
    rename(`Mean training time per fold` = runtime) %>% 
    select(Method, `Mean training time per fold`, `Relative runtime vs SMOTE-ENN`, `Practical note`)
  write_simple_latex_table(runtime_tex, file.path(config$table_dir, "tab_runtime.tex"), 
                           caption = "Runtime and practical trade-off summary.", label = "tab:runtime")
  
  # 9. Appendix Metrics Table
  appendix_metrics <- real_results %>%
    group_by(method) %>%
    summarise(
      Sens = mean(Sensitivity, na.rm = TRUE),
      Spec = mean(Specificity, na.rm = TRUE),
      Prec = mean(Precision, na.rm = TRUE),
      F1 = mean(F1, na.rm = TRUE),
      Gmean = mean(Gmean, na.rm = TRUE),
      BalAcc = mean(BalancedAccuracy, na.rm = TRUE),
      MCC = mean(MCC, na.rm = TRUE),
      AUROC = mean(AUROC, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(Method = dplyr::case_when(
      method == "baseline" ~ "Baseline", method == "smote" ~ "SMOTE", method == "adasyn" ~ "ADASYN",
      method == "borderline_smote" ~ "Borderline-SMOTE", method == "smote_tomek" ~ "SMOTE-Tomek",
      method == "smote_enn" ~ "SMOTE-ENN", method == "da_smote" ~ "DA-SMOTE", method == "penn" ~ "PENN",
      method == "ld_smote" ~ "LD-SMOTE", method == "da_smote_penn" ~ "\\textbf{DA-SMOTE-PENN}"
    )) %>%
    select(Method, Sens, Spec, Prec, F1, Gmean, BalAcc, MCC, AUROC)
  write_simple_latex_table(appendix_metrics, file.path(config$table_dir, "tab_appendix_metrics.tex"),
                           caption = "Mean secondary performance metrics across all real-world datasets and classifiers.",
                           label = "tab:appendix_metrics")
  
  # 10. Average Ranks Table
  ranked_data <- real_results %>% 
    group_by(dataset, classifier, method) %>%
    summarise(metric_value = mean(.data[[config$primary_metric]], na.rm = TRUE), .groups = "drop") %>%
    group_by(dataset, classifier) %>% 
    mutate(rank = rank(-metric_value, ties.method = "average")) %>% 
    ungroup()
  
  avg_ranks_df <- ranked_data %>% 
    group_by(classifier, method) %>% 
    summarise(avg_rank = mean(rank, na.rm = TRUE), .groups = "drop") %>% 
    arrange(method)
  
  avg_ranks_wide <- avg_ranks_df %>%
    pivot_wider(names_from = classifier, values_from = avg_rank) %>%
    mutate(Method = dplyr::case_when(
      method == "baseline" ~ "Baseline",
      method == "smote" ~ "SMOTE",
      method == "adasyn" ~ "ADASYN",
      method == "borderline_smote" ~ "Borderline-SMOTE",
      method == "smote_tomek" ~ "SMOTE-Tomek",
      method == "smote_enn" ~ "SMOTE-ENN",
      method == "da_smote" ~ "DA-SMOTE",
      method == "penn" ~ "PENN",
      method == "ld_smote" ~ "LD-SMOTE",
      method == "da_smote_penn" ~ "\\textbf{DA-SMOTE-PENN}"
    )) %>%
    select(Method, any_of(c("knn", "logit", "svm_rbf", "rf", "xgb")))
  
  .ren_ranks <- c(knn = "k-NN", logit = "LogReg", svm_rbf = "RBF-SVM", rf = "RF", xgb = "XGB")
  .cls_ranks <- setdiff(names(avg_ranks_wide), "Method")
  names(avg_ranks_wide)[-1] <- unname(.ren_ranks[.cls_ranks])
  avg_ranks_wide <- avg_ranks_wide %>% mutate(across(-Method, ~ sprintf("%.2f", .x)))
  
  write_simple_latex_table(avg_ranks_wide, file.path(config$table_dir, "tab_avg_ranks.tex"),
                           caption = paste0("Average rank of resampling methods based on ", config$primary_metric, " across all datasets and classifiers (lower is better)."),
                           label = "tab:avg_ranks")
  
  # 11. Abstract Stats Snippet
  sim_5_995 <- sim_results %>% 
    filter(imbalance_ratio == "5:995", method == "da_smote_penn") %>% 
    summarise(val = mean(.data[[config$primary_metric]], na.rm = TRUE)) %>% pull(val)
  
  ablation_da_smote <- real_results %>% 
    filter(method == "da_smote") %>% 
    summarise(val = mean(.data[[config$primary_metric]], na.rm = TRUE)) %>% pull(val)
  
  ablation_full <- real_results %>% 
    filter(method == "da_smote_penn") %>% 
    summarise(val = mean(.data[[config$primary_metric]], na.rm = TRUE)) %>% pull(val)
  
  abstract_lines <- c(
    paste0("\\newcommand{\\AUPRCFiveNineNineFive}{", sprintf("%.3f", sim_5_995), "}"),
    paste0("\\newcommand{\\AUPRCDaSmote}{", sprintf("%.3f", ablation_da_smote), "}"),
    paste0("\\newcommand{\\AUPRCFullPipeline}{", sprintf("%.3f", ablation_full), "}")
  )
  writeLines(abstract_lines, file.path(config$table_dir, "abstract_stats.tex"))
}

write_reproducibility_files <- function() {
  sink(file.path(config$results_dir, "sessionInfo.txt"))
  print(sessionInfo())
  sink()
  
  sys_info <- c(
    paste0("R_Version: ", R.version.string),
    paste0("OS: ", Sys.info()["sysname"]),
    paste0("n_cores_detected: ", parallel::detectCores())
  )
  writeLines(sys_info, file.path(config$results_dir, "system_info.txt"))
  
  cfg <- c(
    paste0("repo_url: ", config$repo_url), 
    paste0("data_url: ", config$data_url), 
    paste0("master_seed: ", config$master_seed), 
    paste0("primary_metric: ", config$primary_metric), 
    paste0("decision_threshold: ", config$decision_threshold), 
    paste0("canonical_target_ratio: ", config$target_ratio), 
    paste0("canonical_beta: ", config$benchmark_beta), 
    paste0("canonical_k_smote: ", config$benchmark_k_smote), 
    paste0("canonical_k_enn: ", config$benchmark_k_enn), 
    paste0("canonical_k_ld: ", config$benchmark_k_ld), 
    paste0("alpha_grid: ", paste(config$alpha_grid, collapse = ",")), 
    paste0("beta_grid: ", paste(config$beta_grid, collapse = ",")), 
    paste0("k_smote_grid: ", paste(config$k_smote_grid, collapse = ",")), 
    paste0("k_enn_grid: ", paste(config$k_enn_grid, collapse = ",")), 
    paste0("xgb_fixed_defaults: subsample=0.8,colsample_bytree=0.8,min_child_weight=1"), 
    paste0("methods: ", paste(config$methods, collapse = ",")), 
    paste0("classifiers: ", paste(config$classifiers, collapse = ","))
  )
  writeLines(cfg, file.path(config$results_dir, "config_used.txt"))
}

# ==============================================================================
# 8. EXECUTION
# ==============================================================================
main <- function() {
  message("Writing reproducibility files...")
  write_reproducibility_files()
  message("Summarising datasets...")
  dataset_summary <- summarise_dataset_manifest()
  message("Running real-world benchmark...")
  real_results <- run_realworld_benchmark()
  message("Running simulation benchmark...")
  sim_results <- run_simulation_benchmark()
  message("Running sensitivity analysis...")
  sensitivity_results <- run_sensitivity_analysis()
  message("Summarising outputs...")
  summarise_results(real_results, sim_results, sensitivity_results)
  message("Computing average ranks...")
  compute_average_ranks(real_results, metric = config$primary_metric)
  message("Running statistical tests...")
  stat_tests <- run_statistical_tests(real_results, metric = config$primary_metric, reference_method = "da_smote_penn")
  message("Writing manuscript-support tables...")
  write_manuscript_support_tables(real_results, sim_results, sensitivity_results, dataset_summary, stat_tests)
  message("Done. Results and LaTeX tables are in: ", normalizePath(config$output_dir))
}

# ------------------------------------------------------------------------------
# Optional: Quick check function for rapid pipeline validation
# (Useful for verifying repository setup without running the full benchmark)
# ------------------------------------------------------------------------------
quick_main <- function() {
  message("=== RUNNING QUICK CHECK VERSION ===")
  
  # Shrink configuration for speed
  config$outer_folds <- 2
  config$outer_repeats <- 1
  config$inner_folds <- 2
  config$n_sim_replications <- 5
  config$simulation_ratios <- list(c(10L, 990L), c(500L, 500L))
  config$methods <- c("baseline", "smote", "smote_enn", "da_smote_penn")
  config$classifiers <- c("knn", "logit")
  config$alpha_grid <- 0.5
  config$beta_grid <- 0.2
  config$k_smote_grid <- 3L
  config$k_enn_grid <- 3L
  
  # Limit datasets to the smallest ones
  quick_manifest <- dataset_manifest[dataset_manifest$dataset %in% c("newthyroid1", "glass0"), ]
  original_manifest <- dataset_manifest
  assign("dataset_manifest", quick_manifest, envir = .GlobalEnv)
  
  message("Writing reproducibility files...")
  write_reproducibility_files()
  
  message("Summarising datasets...")
  dataset_summary <- summarise_dataset_manifest()
  
  message("Running quick real-world benchmark...")
  real_results <- run_realworld_benchmark()
  
  message("Running quick simulation benchmark...")
  sim_results <- run_simulation_benchmark()
  
  message("Running quick sensitivity analysis...")
  sensitivity_results <- run_sensitivity_analysis()
  
  message("Summarising outputs...")
  summarise_results(real_results, sim_results, sensitivity_results)
  
  message("Computing average ranks...")
  compute_average_ranks(real_results, metric = config$primary_metric)
  
  message("Running statistical tests...")
  stat_tests <- run_statistical_tests(real_results, metric = config$primary_metric, reference_method = "da_smote_penn")
  
  message("Writing manuscript-support tables...")
  write_manuscript_support_tables(real_results, sim_results, sensitivity_results, dataset_summary, stat_tests)
  
  # Restore original manifest for future full runs
  assign("dataset_manifest", original_manifest, envir = .GlobalEnv)
  
  message("=== QUICK CHECK COMPLETE ===")
  message("Check the outputs/tables/ directory for the generated .tex files.")
}

# Execute script
main()
