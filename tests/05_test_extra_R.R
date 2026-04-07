#!/usr/bin/env Rscript
# 05_test_extra_R.R - Additional function tests with synthetic / real data
# Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/

setwd("/Users/yizhouw/Desktop/packages/Clonotrace_python")
library(Clonotrace)
library(Matrix)
library(future)
library(future.apply)
plan(sequential)

set.seed(42)

cat("=== Loading data ===\n")
pca         <- as.matrix(read.csv("tests/data/pca.csv", row.names = 1))
cell_names  <- read.csv("tests/data/cell_names.csv", stringsAsFactors = FALSE)$cell_name
clone_names <- read.csv("tests/data/clone_names.csv", stringsAsFactors = FALSE)$clone_name
triplets    <- read.csv("tests/data/cell_clone_binary_triplets.csv")

n_cells  <- length(cell_names)
n_clones <- length(clone_names)

# Reconstruct binary cell_clone_prob
cell_clone_prob <- sparseMatrix(
    i = triplets$cell_idx + 1L, j = triplets$clone_idx + 1L,
    x = triplets$value, dims = c(n_cells, n_clones)
)
rownames(cell_clone_prob) <- cell_names
colnames(cell_clone_prob) <- clone_names

# Build cell kNN for auxiliary function tests
cat("Building cell kNN...\n")
cell_knn  <- embedding2knn(as.matrix(pca[1:500, ]), k = 15)
T_mat_sub <- compute_transition(cell_knn)

# ─────────────────────────────────────────────────────────────────────────────
# Group A: Format conversion functions
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== Group A: Format Conversions ===\n")

# Build a test long data frame (one direction only — no duplicate pairs)
long_df <- data.frame(
    from  = c("A","A","B"),
    to    = c("B","C","C"),
    value = c(1.0, 2.0, 3.0),
    stringsAsFactors = FALSE
)

# A1: long2square
sq <- long2square(long_df, row_names_from="from", col_names_from="to",
                  values_from="value", symmetric=TRUE)
write.csv(as.data.frame(sq), "tests/outputs_R/long2square_out.csv")
cat("long2square:", dim(sq), "\n")

# A2: long2sparse
sp_mat <- long2sparse(long_df, row_names_from="from", col_names_from="to",
                      values_from="value", symmetric=TRUE)
sp_coo <- summary(sp_mat)
sp_df  <- data.frame(row = sp_coo$i, col = sp_coo$j, value = sp_coo$x)
write.csv(sp_df, "tests/outputs_R/long2sparse_out.csv", row.names = FALSE)
cat("long2sparse nnz:", nrow(sp_df), "\n")

# A3: long2wide
wide_df <- long2wide(long_df, row_names_from="from", col_names_from="to",
                     values_from="value", symmetric=TRUE)
write.csv(as.data.frame(wide_df), "tests/outputs_R/long2wide_out.csv")
cat("long2wide:", dim(wide_df), "\n")

# A4: wide2long
wide_mat <- as.matrix(data.frame(A=c(1,4,7),B=c(2,5,8),C=c(3,6,9),
                                  row.names=c("X","Y","Z")))
wl <- wide2long(wide_mat)
write.csv(wl, "tests/outputs_R/wide2long_out.csv", row.names = FALSE)
cat("wide2long:", dim(wl), "\n")

# A5: long_symmetry
sym_df <- long_symmetry(long_df, row_names_from="from", col_names_from="to")
write.csv(sym_df, "tests/outputs_R/long_symmetry_out.csv", row.names = FALSE)
cat("long_symmetry:", dim(sym_df), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Group B: Core Algorithms
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== Group B: Core Algorithms ===\n")

# B1: compute_transition (double-stochastic normalization)
T_df <- data.frame(summary(T_mat_sub[1:200, ]))
colnames(T_df) <- c("row", "col", "value")
write.csv(T_df, "tests/outputs_R/compute_transition_out.csv", row.names = FALSE)
cat("compute_transition T[1:200,] nnz:", nrow(T_df), "\n")

# B2: mat_sparsify
set.seed(42)
test_mat <- matrix(abs(rnorm(20 * 20)), 20, 20)
test_mat <- test_mat / rowSums(test_mat)  # normalize rows
sparse_mat <- mat_sparsify(test_mat, row_mass = 0.9, col_mass = 0.9)
write.csv(sparse_mat, "tests/outputs_R/mat_sparsify_out.csv")
cat("mat_sparsify: non-zero fraction =",
    sum(sparse_mat != 0) / length(sparse_mat), "\n")

# B3: dis2connec_sparse
set.seed(42)
dis_mat_test <- as(matrix(runif(15 * 15, 0.1, 2), 15, 15), "sparseMatrix")
diag(dis_mat_test) <- 0
dis_mat_test <- drop0(dis_mat_test)
connec <- dis2connec_sparse(dis_mat_test)
coo <- summary(connec)
write.csv(data.frame(row=coo$i, col=coo$j, value=coo$x),
          "tests/outputs_R/dis2connec_out.csv", row.names = FALSE)
cat("dis2connec nnz:", nrow(coo), "\n")

# B4: dismat_mst
set.seed(42)
dis_dense <- matrix(runif(10 * 10, 0.5, 3), 10, 10)
dis_dense <- (dis_dense + t(dis_dense)) / 2
diag(dis_dense) <- 0
mst_edges <- Clonotrace:::dismat_mst(dis_dense)
write.csv(mst_edges, "tests/outputs_R/dismat_mst_out.csv", row.names = FALSE)
cat("MST edges:", nrow(mst_edges), "(should be n-1 =", nrow(dis_dense)-1, ")\n")

# B5: dist2knn
knn_from_dist <- dist2knn(dis_dense, k = 3)
knn_coo <- summary(knn_from_dist)
write.csv(data.frame(row=knn_coo$i, col=knn_coo$j, value=knn_coo$x),
          "tests/outputs_R/dist2knn_out.csv", row.names = FALSE)
cat("dist2knn nnz:", nrow(knn_coo), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Group C: Co-embedding (use small subset)
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== Group C: Co-embedding ===\n")

# Use first 200 cells and first 50 clones for co-embedding
n_sub_cells  <- 200
n_sub_clones <- 50
sub_pca   <- pca[1:n_sub_cells, ]
sub_prob  <- cell_clone_prob[1:n_sub_cells, 1:n_sub_clones]
sub_prob  <- as.matrix(sub_prob)

# Build simple clone embedding from sub_prob co-occurrence
clone_overlap <- t(sub_prob) %*% sub_prob
clone_overlap_mat <- as.matrix(clone_overlap)
set.seed(42)
clone_emb_raw <- MASS::isoMDS(as.matrix(1 - clone_overlap_mat / max(clone_overlap_mat) + 1e-6),
                         k = 10, trace = FALSE)$points
clone_emb <- as.data.frame(clone_emb_raw)
rownames(clone_emb) <- colnames(sub_prob)
colnames(clone_emb) <- paste0("dim", 1:10)

# For R cell_clone_coembed, cell_clone_prob must be in global scope
cell_clone_prob_global <- sub_prob
assign("cell_clone_prob", cell_clone_prob_global, envir = .GlobalEnv)

coembed_dis <- cell_clone_coembed(
    cell_embedding = sub_pca,
    clone_embedding = as.matrix(clone_emb),
    cell_k = 15, clone_k = 10
)

# Save first 200x200 block (non-zero entries)
coo_co <- summary(coembed_dis[1:200, 1:200])
co_df  <- data.frame(row = coo_co$i, col = coo_co$j, value = coo_co$x)

# Also save as dense 200x10 (distances to first 10 cells) for Pearson comparison
co_dense <- as.matrix(coembed_dis[1:200, 1:10])
write.csv(as.data.frame(co_dense), "tests/outputs_R/coembed_out.csv")
cat("coembed_out saved:", dim(co_dense), "\n")

# Group D: Visualization embeddings
# Note: umap_from_knn and mds_from_knn in R require Python backends via reticulate.
# These are tested in Python only (06_test_extra_python.py).
cat("\n=== Group D: Visualization Embeddings (Python-only — skipped in R) ===\n")

cat("\n=== All extra R tests complete ===\n")
cat("Outputs saved to tests/outputs_R/\n")
