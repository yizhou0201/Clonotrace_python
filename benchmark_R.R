# R Benchmark Script for Clonotrace
# Benchmarks the original R implementations for comparison with Python

library(Matrix)
library(RANN)
library(dbscan)
library(expm)

set.seed(42)

# Source the R package functions
R_DIR <- "/Users/yizhouw/Desktop/packages/Clonotrace_yuntian/R"

# We need to load dependencies first
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(magrittr)
  library(igraph)
  library(RSpectra)
})

# Source the R files
source(file.path(R_DIR, "auxiliary.R"))
source(file.path(R_DIR, "pseudotime.R"))
source(file.path(R_DIR, "cluster.R"))
source(file.path(R_DIR, "clone_dis.R"))

# Helper: time a function and measure memory via pryr/lobstr or object sizes
bench <- function(func, ..., repeats = 3) {
  times <- numeric(repeats)
  result <- NULL
  for (i in seq_len(repeats)) {
    gc(verbose = FALSE, reset = TRUE)
    t0 <- proc.time()["elapsed"]
    result <- func(...)
    t1 <- proc.time()["elapsed"]
    times[i] <- t1 - t0
  }
  gc_info <- gc(verbose = FALSE)
  # gc matrix: row 2 = Vcells, col 6 = max used, col 7 = (Mb)
  peak_mem_mb <- gc_info[2, 7]  # max used Vcells in MB
  list(time = median(times), peak_mem_mb = peak_mem_mb, result = result)
}

results <- list()

# ---------------------------------------------------------------
# 1. dis_points_to_edges (5000 points, 50 edges, 3D)
# ---------------------------------------------------------------
cat("1. dis_points_to_edges ... ")
points <- data.frame(matrix(rnorm(5000 * 3), ncol = 3))
edges <- lapply(1:50, function(i) matrix(rnorm(6), nrow = 2, ncol = 3))

b <- bench(dis_points_to_edges, points, edges)
results$dis_points_to_edges <- list(time = b$time, peak_mem_mb = b$peak_mem_mb)
cat(sprintf("%.4fs  mem=%.1fMB\n", b$time, b$peak_mem_mb))

# ---------------------------------------------------------------
# 2. mat_sparsify (300x300)
# ---------------------------------------------------------------
cat("2. mat_sparsify ... ")
mat <- matrix(runif(300 * 300), nrow = 300)

b <- bench(mat_sparsify, mat, 0.9, 0.9)
results$mat_sparsify <- list(time = b$time, peak_mem_mb = b$peak_mem_mb)
cat(sprintf("%.4fs  mem=%.1fMB\n", b$time, b$peak_mem_mb))

# ---------------------------------------------------------------
# 3. embedding2knn (3000 pts, k=15)
# ---------------------------------------------------------------
cat("3. embedding2knn ... ")
emb <- matrix(rnorm(3000 * 20), ncol = 20)

b <- bench(embedding2knn, emb, k = 15, mode = "connectivity", repeats = 2)
results$embedding2knn <- list(time = b$time, peak_mem_mb = b$peak_mem_mb)
cat(sprintf("%.4fs  mem=%.1fMB\n", b$time, b$peak_mem_mb))

# ---------------------------------------------------------------
# 4. snn_from_dist (dbscan::sNN, equivalent to _snn_from_dist)
# ---------------------------------------------------------------
cat("4. snn_from_dist ... ")
set.seed(42)
x <- matrix(rnorm(500 * 10), ncol = 10)
dismat <- as.matrix(dist(x))

bench_snn <- function(dismat, k) {
  snn <- sNN(as.dist(dismat), k = k)
  snn$jaccard <- snn$shared / (2 * k - snn$shared)
  return(snn)
}
b <- bench(bench_snn, dismat, 15)
results$snn_from_dist <- list(time = b$time, peak_mem_mb = b$peak_mem_mb)
cat(sprintf("%.4fs  mem=%.1fMB\n", b$time, b$peak_mem_mb))

# ---------------------------------------------------------------
# 5. link2cluster (500 nodes)
# ---------------------------------------------------------------
cat("5. link2cluster ... ")
links_i <- c(0:199, 250:399)
links_j <- c(1:200, 251:400)
link <- data.frame(i = as.character(links_i), j = as.character(links_j))
nodes <- as.character(0:499)

b <- bench(link2cluster, link, nodes, repeats = 1)
results$link2cluster <- list(time = b$time, peak_mem_mb = b$peak_mem_mb)
cat(sprintf("%.4fs  mem=%.1fMB\n", b$time, b$peak_mem_mb))

# ---------------------------------------------------------------
# 6. nearest_knn (300 nodes, k=10)
# ---------------------------------------------------------------
cat("6. nearest_knn ... ")
set.seed(42)
x300 <- matrix(rnorm(300 * 10), ncol = 10)
dismat300 <- as.matrix(dist(x300))

b <- bench(nearest_knn, dismat300, k = 10, top = 20)
results$nearest_knn <- list(time = b$time, peak_mem_mb = b$peak_mem_mb)
cat(sprintf("%.4fs  mem=%.1fMB\n", b$time, b$peak_mem_mb))

# ---------------------------------------------------------------
# 7. DPT_T (N=200)
# ---------------------------------------------------------------
cat("7. DPT_T ... ")
set.seed(42)
x200 <- matrix(rnorm(200 * 10), ncol = 10)
knn200 <- embedding2knn(x200, k = 10)
T_mat <- knn200 / rowSums(knn200)

b <- bench(DPT_T, T_mat, start = 1, repeats = 1)
results$DPT_T <- list(time = b$time, peak_mem_mb = b$peak_mem_mb)
cat(sprintf("%.4fs  mem=%.1fMB\n", b$time, b$peak_mem_mb))

# ---------------------------------------------------------------
# 8. acct (N=200)
# ---------------------------------------------------------------
cat("8. acct ... ")
b <- bench(acct, T_mat, repeats = 1)
results$acct <- list(time = b$time, peak_mem_mb = b$peak_mem_mb)
cat(sprintf("%.4fs  mem=%.1fMB\n", b$time, b$peak_mem_mb))

# ---------------------------------------------------------------
# 9. clone_partition (50 clones)
# ---------------------------------------------------------------
cat("9. clone_partition ... ")
set.seed(42)
clone_prob <- matrix(0, nrow = 500, ncol = 50)
colnames(clone_prob) <- paste0("clone_", 1:50)
for (i in 1:500) {
  clone_prob[i, sample(50, 1)] <- runif(1)
}

b <- bench(clone_partition, clone_prob, k = 5)
results$clone_partition <- list(time = b$time, peak_mem_mb = b$peak_mem_mb)
cat(sprintf("%.4fs  mem=%.1fMB\n", b$time, b$peak_mem_mb))

# ---------------------------------------------------------------
# Save results
# ---------------------------------------------------------------
cat("\nSaving R results...\n")
results_json <- jsonlite::toJSON(results, auto_unbox = TRUE, pretty = TRUE)
writeLines(results_json, "/Users/yizhouw/Desktop/packages/Clonotrace_python/benchmark_R_results.json")
cat("Done!\n")
