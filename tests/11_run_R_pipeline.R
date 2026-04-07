#!/usr/bin/env Rscript
# 11_run_R_pipeline.R - Run full R vignette pipeline on real hematopoiesis data
# and save all intermediate outputs for comparison with Python.
#
# Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/
# Prerequisite: Run 10_export_real_data.R first

setwd("/Users/yizhouw/Desktop/packages/Clonotrace_python")
dir.create("tests/real_outputs_R", showWarnings = FALSE, recursive = TRUE)

library(Clonotrace)
library(Matrix)
library(MASS)
library(dplyr)
library(future)
library(future.apply)

options(future.globals.maxSize = 4 * 1024^3)

cat("=== Loading exported data ===\n")
pca         <- as.matrix(read.csv("tests/real_data/pca.csv", row.names = 1))
cell_meta   <- read.csv("tests/real_data/cell_meta.csv", row.names = 1,
                         stringsAsFactors = FALSE)
cell_names  <- read.csv("tests/real_data/cell_names.csv",
                         stringsAsFactors = FALSE)$cell_name
clone_names <- read.csv("tests/real_data/clone_names.csv",
                         stringsAsFactors = FALSE)$clone_name
triplets    <- read.csv("tests/real_data/cell_clone_binary_triplets.csv")
label_df    <- read.csv("tests/real_data/cell_clone_labels.csv",
                         stringsAsFactors = FALSE)

n_cells  <- length(cell_names)
n_clones <- length(clone_names)
cat(sprintf("PCA: %d x %d | Clones: %d | Cells: %d\n",
            nrow(pca), ncol(pca), n_clones, n_cells))

# Reconstruct binary cell-clone probability matrix (1-indexed)
cell_clone_binary <- sparseMatrix(
    i = triplets$cell_idx + 1L,
    j = triplets$clone_idx + 1L,
    x = triplets$value,
    dims = c(n_cells, n_clones)
)
rownames(cell_clone_binary) <- cell_names
colnames(cell_clone_binary) <- clone_names

# ── Step 1: kNN + transition ────────────────────────────────────────────────
cat("\n=== Step 1: kNN + transition ===\n")
cell_knn <- embedding2knn(as.matrix(pca), k = 30, mode = "connectivity",
                           if_self = FALSE)
T_mat <- compute_transition(cell_knn)

# Save sample for comparison
T_sub <- T_mat[1:200, ]
T_coo <- summary(T_sub)
write.csv(data.frame(row = T_coo$i, col = T_coo$j, value = T_coo$x),
          "tests/real_outputs_R/cell_knn_sample.csv", row.names = FALSE)
cat(sprintf("T_mat[1:200,] nnz: %d\n", nrow(T_coo)))

# ── Step 2: Label spreading (bootstrap) ─────────────────────────────────────
cat("\n=== Step 2: Label spreading (bootstrap, sample_n=48) ===\n")
set.seed(42)
clone_labels_vec <- label_df$clone
names(clone_labels_vec) <- label_df$cell

start_time <- Sys.time()
plan(multisession, workers = 8)
clone_spread <- label_spreading_bootstrap(
    adj = T_mat,
    labels = clone_labels_vec,
    alpha = 0.6,
    sample_rate = 0.8,
    sample_n = 48
)
elapsed <- as.numeric(Sys.time() - start_time, units = "mins")
cat(sprintf("Label spreading time: %.1f min\n", elapsed))

cell_clone_prob_raw <- clone_spread[[1]]
deviance <- clone_spread[[2]]
rownames(cell_clone_prob_raw) <- cell_names

# Save deviance
write.csv(data.frame(cell = cell_names, deviance = deviance),
          "tests/real_outputs_R/deviance.csv", row.names = FALSE)

# Save a sample of cell_clone_prob (first 500 cells, first 50 clones)
prob_sample <- as.matrix(cell_clone_prob_raw[1:500, 1:50])
write.csv(prob_sample, "tests/real_outputs_R/cell_clone_prob_sample.csv")
cat(sprintf("cell_clone_prob: %d x %d\n",
            nrow(cell_clone_prob_raw), ncol(cell_clone_prob_raw)))

# ── Step 3: Filter and sparsify ─────────────────────────────────────────────
cat("\n=== Step 3: Filter by deviance + sparsify ===\n")
cell_clone_prob <- cell_clone_prob_raw[deviance < 0.3, ]
cell_clone_prob <- cell_clone_prob / rowSums(cell_clone_prob)
cell_clone_prob <- mat_sparsify(mat = cell_clone_prob, row_mass = 0.9,
                                 col_mass = 0.9)
cell_clone_prob <- cell_clone_prob / rowSums(cell_clone_prob)
cell_clone_prob <- Matrix(cell_clone_prob, sparse = TRUE)
colnames(cell_clone_prob) <- clone_names
cat(sprintf("After filtering: %d cells x %d clones\n",
            nrow(cell_clone_prob), ncol(cell_clone_prob)))

# Save filtered cell_clone_prob as triplets
ccp_coo <- summary(cell_clone_prob)
write.csv(data.frame(row = ccp_coo$i, col = ccp_coo$j, value = ccp_coo$x),
          "tests/real_outputs_R/cell_clone_prob_filtered_triplets.csv",
          row.names = FALSE)
write.csv(data.frame(cell = rownames(cell_clone_prob)),
          "tests/real_outputs_R/cell_clone_prob_filtered_cells.csv",
          row.names = FALSE)

# ── Step 4: Clone NN distance (full) ────────────────────────────────────────
cat("\n=== Step 4: Clone NN distance (full, approximate) ===\n")
options(future.stdout = NA)  # Prevent long-vector stdout error in future workers
start_time <- Sys.time()
plan(multisession, workers = 8)
clone_nn_dis <- clone_disance(
    as.matrix(pca[rownames(cell_clone_prob), ]),
    cell_clone_prob,
    outpath = "tests/real_outputs_R/clone_nn_cache/",
    exact = FALSE,
    overwrite = TRUE
)
elapsed <- as.numeric(Sys.time() - start_time, units = "mins")
cat(sprintf("Clone NN distance time: %.1f min\n", elapsed))
write.csv(clone_nn_dis, "tests/real_outputs_R/clone_nn_dis.csv", row.names = FALSE)

# Convert to square matrix
clone_dis_sq <- long2square(
    long = as.data.frame(clone_nn_dis),
    row_names_from = "group1",
    col_names_from = "group2",
    values_from = "dis",
    symmetric = TRUE
)
diag(clone_dis_sq) <- 0
rownames(clone_dis_sq) <- colnames(clone_dis_sq) <- clone_names
# Save first 50x50
write.csv(clone_dis_sq[1:50, 1:50], "tests/real_outputs_R/clone_nn_dis_sq50.csv")

# ── Step 5: Clone OT distance (subset of ~50 clones) ────────────────────────
cat("\n=== Step 5: Clone OT distance (first 50 clones) ===\n")
sub50 <- 1:50
cell_clone_sub50 <- cell_clone_prob[, sub50]
start_time <- Sys.time()
options(future.stdout = NA)
plan(multisession, workers = 8)
clone_ot_dis <- clone_disance(
    as.matrix(pca[rownames(cell_clone_sub50), ]),
    cell_clone_sub50,
    outpath = "tests/real_outputs_R/clone_ot_cache/",
    exact = TRUE,
    cores = 8,
    overwrite = TRUE,
    verbose = FALSE
)
elapsed <- as.numeric(Sys.time() - start_time, units = "mins")
cat(sprintf("Clone OT distance time: %.1f min\n", elapsed))
write.csv(clone_ot_dis, "tests/real_outputs_R/clone_ot_dis_sub50.csv", row.names = FALSE)

# ── Step 6: Clone MDS (30 dims) ─────────────────────────────────────────────
cat("\n=== Step 6: Clone MDS (30 dims) ===\n")
set.seed(42)
clone_mds_result <- MASS::isoMDS(as.matrix(clone_dis_sq), k = 30, trace = FALSE)
clone_embedding <- clone_mds_result$points
colnames(clone_embedding) <- paste("mds", 1:ncol(clone_embedding), sep = "_")
rownames(clone_embedding) <- clone_names
write.csv(clone_embedding, "tests/real_outputs_R/clone_mds.csv")

# ── Step 7: Leiden clustering ────────────────────────────────────────────────
cat("\n=== Step 7: Leiden clustering ===\n")
set.seed(1230)
clone_cluster <- leiden_dis(dismat = clone_dis_sq, k = 20, resolution = 0.5,
                             if_umap = TRUE)
rownames(clone_cluster) <- clone_names
write.csv(clone_cluster, "tests/real_outputs_R/clone_cluster.csv")
cat(sprintf("Clusters: %d unique\n", length(unique(clone_cluster$cluster))))

# ── Step 8: Clone pseudotime ────────────────────────────────────────────────
cat("\n=== Step 8: Clone pseudotime ===\n")
clone_t <- clone_dpt(
    clone_embedding = as.data.frame(clone_embedding),
    cell_meta = cell_meta,
    clone_col = "clone",
    cluster_col = "cluster",
    start_cluster = "0"
)
write.csv(data.frame(clone = clone_names, dpt = clone_t),
          "tests/real_outputs_R/clone_t.csv", row.names = FALSE)
cat(sprintf("Clone pseudotime range: [%.3f, %.3f]\n", min(clone_t), max(clone_t)))

# ── Step 9: Cell profile probabilities ───────────────────────────────────────
cat("\n=== Step 9: Cell profile probabilities ===\n")
clone_cluster$clone <- rownames(clone_cluster)
cluster_order <- sort(unique(as.character(clone_cluster$cluster)))

clone_profile_mat <- matrix(0, nrow = n_clones, ncol = length(cluster_order))
rownames(clone_profile_mat) <- clone_names
colnames(clone_profile_mat) <- cluster_order
for (ci in seq_along(clone_names)) {
    cl_label <- as.character(clone_cluster$cluster[ci])
    clone_profile_mat[ci, cl_label] <- 1
}
clone_profile_sp <- Matrix(clone_profile_mat, sparse = TRUE)

cell_profile_prob <- cell_clone_prob %*% clone_profile_sp[clone_names, ]
rownames(cell_profile_prob) <- rownames(cell_clone_prob)
cell_profile_prob_norm <- sweep(cell_profile_prob, 1,
                                 rowSums(cell_profile_prob) + 1e-12, "/")

# Cell pseudotime
cell_t <- as.numeric(cell_clone_prob[, clone_names] %*% clone_t)
names(cell_t) <- rownames(cell_clone_prob)
cell_meta_sub <- cell_meta[rownames(cell_clone_prob), ]
cell_meta_sub$cell_t <- cell_t
write.csv(cell_meta_sub, "tests/real_outputs_R/cell_meta_with_t.csv")
write.csv(as.matrix(cell_profile_prob_norm),
          "tests/real_outputs_R/cell_profile_prob.csv")

# ── Step 10: Profile enrichment ─────────────────────────────────────────────
cat("\n=== Step 10: Profile enrichment ===\n")
cell_clusters <- as.character(cell_meta_sub$cluster)
enrich <- cluster_profile_enrich(
    cell_profile_prob = as.matrix(cell_profile_prob_norm),
    cluster_label = cell_clusters,
    permute_n = 300
)
write.csv(enrich[[1]], "tests/real_outputs_R/enrich_mass.csv")
write.csv(enrich[[2]], "tests/real_outputs_R/enrich_pval.csv")
cat(sprintf("Enrichment: %d clusters x %d profiles\n",
            nrow(enrich[[1]]), ncol(enrich[[1]])))

# ── Step 11: Profile DEG ────────────────────────────────────────────────────
cat("\n=== Step 11: Profile DEG ===\n")
exprs_path <- "tests/real_data/exprs_cluster4.csv"
if (file.exists(exprs_path)) {
    exprs_cl4 <- as.matrix(read.csv(exprs_path, row.names = 1))
    profile_name <- cluster_order[1]
    cat(sprintf("Testing profile: %s\n", profile_name))

    deg_result <- profile_cluster_DEG(
        profile = profile_name,
        cluster = "4",
        exprs = exprs_cl4,
        cell_meta = cell_meta_sub,
        cell_profile_prob = as.data.frame(as.matrix(cell_profile_prob_norm))
    )

    if (!is.null(deg_result)) {
        write.csv(deg_result$stat, "tests/real_outputs_R/DEG_stats.csv")
        cat(sprintf("DEG result: %d genes\n", nrow(deg_result$stat)))
    } else {
        cat("DEG returned NULL (insufficient cells)\n")
    }
} else {
    cat("Expression data not found. Run 10_export_real_data.R with Seurat object.\n")
}

# ── Step 12: SNN graph edge weights (for debugging SNN divergence) ───────────
cat("\n=== Step 12: Export SNN graph weights for comparison ===\n")
library(bluster)
snn_graph <- makeSNNGraph(pca[1:500, ], k = 10, type = "number")
snn_w <- igraph::edge_attr(snn_graph, "weight")
snn_edges <- igraph::as_edgelist(snn_graph)
snn_df <- data.frame(
    from = snn_edges[, 1],
    to = snn_edges[, 2],
    shared_count = snn_w,
    weight_exp = exp(-snn_w)
)
write.csv(snn_df, "tests/real_outputs_R/snn_graph_weights.csv", row.names = FALSE)
cat(sprintf("SNN graph edges: %d\n", nrow(snn_df)))

cat("\n=== All R pipeline steps complete ===\n")
cat("Outputs saved to tests/real_outputs_R/\n")
