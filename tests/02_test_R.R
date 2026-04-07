#!/usr/bin/env Rscript
# 02_test_R.R - Run R vignette pipeline and save outputs for comparison
# Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/

setwd("/Users/yizhouw/Desktop/packages/Clonotrace_python")
dir.create("tests/outputs_R", showWarnings = FALSE, recursive = TRUE)

library(Clonotrace)
library(Matrix)
library(MASS)

cat("=== Loading exported data ===\n")
pca         <- as.matrix(read.csv("tests/data/pca.csv", row.names = 1))
cell_meta   <- read.csv("tests/data/cell_meta.csv", row.names = 1, stringsAsFactors = FALSE)
cell_names  <- read.csv("tests/data/cell_names.csv", stringsAsFactors = FALSE)$cell_name
clone_names <- read.csv("tests/data/clone_names.csv", stringsAsFactors = FALSE)$clone_name
triplets    <- read.csv("tests/data/cell_clone_binary_triplets.csv")
label_df    <- read.csv("tests/data/cell_clone_labels.csv", stringsAsFactors = FALSE)
exprs_cl4   <- as.matrix(read.csv("tests/data/exprs_cluster4.csv", row.names = 1))

cat(sprintf("PCA: %d x %d | Clones: %d | Cells: %d\n",
            nrow(pca), ncol(pca), length(clone_names), length(cell_names)))

# Reconstruct binary cell-clone probability matrix (1-indexed)
n_cells <- length(cell_names)
n_clones <- length(clone_names)
cell_clone_prob <- sparseMatrix(
    i = triplets$cell_idx + 1L,
    j = triplets$clone_idx + 1L,
    x = triplets$value,
    dims = c(n_cells, n_clones)
)
rownames(cell_clone_prob) <- cell_names
colnames(cell_clone_prob) <- clone_names
cat(sprintf("cell_clone_prob: %d x %d (sparse)\n", nrow(cell_clone_prob), ncol(cell_clone_prob)))

# ── Step 1: kNN + transition ──────────────────────────────────────────────────
cat("\n=== Step 1: kNN + transition ===\n")
cell_knn <- embedding2knn(as.matrix(pca), k = 30, mode = "connectivity", if_self = FALSE)
T_mat <- compute_transition(cell_knn)

# Save first 200 cells' non-zero entries of T_mat
T_sub <- T_mat[1:200, ]
T_coo <- summary(T_sub)
T_df <- data.frame(row = T_coo$i, col = T_coo$j, value = T_coo$x)
write.csv(T_df, "tests/outputs_R/cell_knn_sample.csv", row.names = FALSE)
cat(sprintf("T_mat[1:200,] non-zero entries: %d\n", nrow(T_df)))

# ── Step 2: Label spreading (small subset) ───────────────────────────────────
cat("\n=== Step 2: Label spreading (subset: 2000 cells, 20 clones) ===\n")
set.seed(42)
sub_cells <- 1:2000
sub_clones <- clone_names[1:20]
T_sub20 <- T_mat[sub_cells, sub_cells]

clone_labels_sub <- label_df$clone[sub_cells]
names(clone_labels_sub) <- cell_names[sub_cells]
clone_labels_sub[!(clone_labels_sub %in% sub_clones)] <- NA

spread_result <- label_spreading_bootstrap(
    adj = T_sub20,
    labels = clone_labels_sub,
    alpha = 0.6, sample_rate = 0.8, sample_n = 5
)
prob_sub <- spread_result[[1]]
rownames(prob_sub) <- cell_names[sub_cells]
write.csv(as.matrix(prob_sub), "tests/outputs_R/label_spread_small.csv")
cat(sprintf("Label spread result: %d x %d\n", nrow(prob_sub), ncol(prob_sub)))

# ── Step 3: Clone NN distance (full 802 clones) ──────────────────────────────
cat("\n=== Step 3: Clone NN distance (full, 802 clones) ===\n")
start_time <- Sys.time()
clone_nn_dis <- clone_disance(
    as.matrix(pca), cell_clone_prob,
    outpath = "tests/outputs_R/clone_dis_cache/",
    exact = FALSE, overwrite = TRUE
)
cat(sprintf("Clone NN distance time: %.1f min\n", as.numeric(Sys.time() - start_time, units = "mins")))
write.csv(clone_nn_dis, "tests/outputs_R/clone_nn_dis.csv", row.names = FALSE)
cat(sprintf("Clone NN pairs: %d\n", nrow(clone_nn_dis)))

# Convert to square matrix and save named version for comparison
clone_dis_sq <- long2square(
    long = as.data.frame(clone_nn_dis),
    row_names_from = "group1",
    col_names_from = "group2",
    values_from = "dis",
    symmetric = TRUE
)
diag(clone_dis_sq) <- 0
rownames(clone_dis_sq) <- colnames(clone_dis_sq) <- clone_names
# Save named square matrix for comparison (sample: first 50x50 to keep CSV small)
write.csv(clone_dis_sq[1:50, 1:50], "tests/outputs_R/clone_nn_dis_sq50.csv")

# ── Step 4: Clone OT distance (first 30 clones) ──────────────────────────────
cat("\n=== Step 4: Clone OT distance (first 30 clones) ===\n")
sub30_clones <- 1:30
cell_clone_sub30 <- cell_clone_prob[, sub30_clones]
start_time <- Sys.time()
# Disable future stdout capture to avoid FutureLaunchError with long vectors
options(future.stdout = NA)
clone_ot_dis <- clone_disance(
    as.matrix(pca), cell_clone_sub30,
    outpath = "tests/outputs_R/clone_ot_cache/",
    exact = TRUE, cores = 1, overwrite = TRUE
)
cat(sprintf("Clone OT distance time: %.1f min\n", as.numeric(Sys.time() - start_time, units = "mins")))
write.csv(clone_ot_dis, "tests/outputs_R/clone_ot_dis_subset.csv", row.names = FALSE)
cat(sprintf("Clone OT pairs: %d\n", nrow(clone_ot_dis)))

# ── Step 5: Clone MDS (30 dims) ──────────────────────────────────────────────
cat("\n=== Step 5: Clone MDS (30 dims) ===\n")
set.seed(42)
clone_mds_result <- MASS::isoMDS(as.matrix(clone_dis_sq), k = 30, trace = FALSE)
clone_embedding <- clone_mds_result$points
colnames(clone_embedding) <- paste("mds", 1:ncol(clone_embedding), sep = "_")
rownames(clone_embedding) <- clone_names
write.csv(clone_embedding, "tests/outputs_R/clone_mds.csv")
cat(sprintf("Clone MDS: %d x %d\n", nrow(clone_embedding), ncol(clone_embedding)))

# ── Step 6: Leiden clustering ─────────────────────────────────────────────────
cat("\n=== Step 6: Leiden clustering ===\n")
set.seed(1230)
clone_cluster <- leiden_dis(dismat = clone_dis_sq, k = 20, resolution = 0.5, if_umap = TRUE)
rownames(clone_cluster) <- clone_names
write.csv(clone_cluster, "tests/outputs_R/clone_cluster.csv")
cat(sprintf("Clone clusters: %d unique, %d clones\n",
            length(unique(clone_cluster$cluster)), nrow(clone_cluster)))

# ── Step 7: Clone pseudotime ─────────────────────────────────────────────────
cat("\n=== Step 7: Clone pseudotime ===\n")
clone_embedding_df <- as.data.frame(clone_embedding)
clone_t <- clone_dpt(
    clone_embedding = clone_embedding_df,
    cell_meta = cell_meta,
    clone_col = "clone",
    cluster_col = "cluster",
    start_cluster = "0"
)
clone_t_df <- data.frame(clone = clone_names, dpt = clone_t)
write.csv(clone_t_df, "tests/outputs_R/clone_t.csv", row.names = FALSE)
cat(sprintf("Clone pseudotime range: [%.3f, %.3f]\n", min(clone_t), max(clone_t)))

# ── Step 8: Cell pseudotime and profile ──────────────────────────────────────
cat("\n=== Step 8: Cell profile probabilities ===\n")
# Build clone_profile mapping (clone -> leiden cluster)
clone_profile_df <- data.frame(
    clone = clone_names,
    cluster = as.character(clone_cluster$cluster),
    flag = 1,
    stringsAsFactors = FALSE
)
clone_profile_sparse <- long2sparse(
    long = clone_profile_df,
    row_names_from = "clone",
    col_names_from = "cluster",
    values_from = "flag"
)
cluster_order <- sort(unique(as.character(clone_cluster$cluster)))

# Match columns
clone_profile_mat <- matrix(0, nrow = n_clones, ncol = length(cluster_order))
rownames(clone_profile_mat) <- clone_names
colnames(clone_profile_mat) <- cluster_order
for (ci in seq_along(clone_names)) {
    cl_label <- as.character(clone_cluster$cluster[ci])
    clone_profile_mat[ci, cl_label] <- 1
}
clone_profile_sp <- Matrix(clone_profile_mat, sparse = TRUE)

cell_profile_prob <- cell_clone_prob %*% clone_profile_sp[clone_names, ]
rownames(cell_profile_prob) <- cell_names
cell_profile_prob_norm <- sweep(cell_profile_prob, 1, rowSums(cell_profile_prob) + 1e-12, "/")

# Cell pseudotime
cell_t <- as.numeric(cell_clone_prob %*% clone_t)
names(cell_t) <- cell_names
cell_meta$cell_t <- cell_t

# Save cell_meta with cell_t added
write.csv(cell_meta, "tests/outputs_R/cell_meta_with_t.csv")

# Save cell profile probabilities
write.csv(as.matrix(cell_profile_prob_norm), "tests/outputs_R/cell_profile_prob.csv")

# ── Step 9: Profile enrichment ───────────────────────────────────────────────
cat("\n=== Step 9: Profile enrichment ===\n")
cluster_col_name <- if ("cluster" %in% colnames(cell_meta)) "cluster" else "seurat_clusters"
cell_clusters <- as.character(cell_meta$cluster)
names(cell_clusters) <- rownames(cell_meta)

# Use cells that have profile probs
valid_cells <- rownames(cell_profile_prob_norm)
enrich <- cluster_profile_enrich(
    cell_profile_prob = as.matrix(cell_profile_prob_norm),
    cluster_label = cell_clusters[valid_cells],
    permute_n = 300
)
write.csv(enrich[[1]], "tests/outputs_R/enrich_mass.csv")
write.csv(enrich[[2]], "tests/outputs_R/enrich_pval.csv")
cat(sprintf("Enrichment result: %d clusters x %d profiles\n",
            nrow(enrich[[1]]), ncol(enrich[[1]])))

# ── Step 10: Profile DEG ─────────────────────────────────────────────────────
cat("\n=== Step 10: Profile DEG (profile=1, cluster=4) ===\n")
# Get the profile column name for "1" (first leiden cluster sorted)
profile_name <- colnames(cell_profile_prob_norm)[1]
cat(sprintf("Testing profile: %s\n", profile_name))

deg_result <- profile_cluster_DEG(
    profile = profile_name,
    cluster = "4",
    exprs = exprs_cl4,   # genes x cells (200 x 3722)
    cell_meta = cell_meta,
    cell_profile_prob = as.data.frame(as.matrix(cell_profile_prob_norm))
)

if (!is.null(deg_result)) {
    write.csv(deg_result$stat, "tests/outputs_R/DEG_stats.csv")
    cat(sprintf("DEG result: %d genes\n", nrow(deg_result$stat)))
} else {
    cat("DEG returned NULL (insufficient cells)\n")
    write.csv(data.frame(note = "NULL result"), "tests/outputs_R/DEG_stats.csv", row.names = FALSE)
}

cat("\n=== All R pipeline steps complete ===\n")
cat("Outputs saved to tests/outputs_R/\n")
