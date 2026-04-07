#!/usr/bin/env Rscript
# 02b_resume_R.R - Resume R pipeline from step 4 (steps 1-3 already done)
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

n_cells <- length(cell_names)
n_clones <- length(clone_names)
cat(sprintf("PCA: %d x %d | Clones: %d | Cells: %d\n",
            nrow(pca), ncol(pca), n_clones, n_cells))

# Reconstruct binary cell-clone probability matrix
cell_clone_prob <- sparseMatrix(
    i = triplets$cell_idx + 1L,
    j = triplets$clone_idx + 1L,
    x = triplets$value,
    dims = c(n_cells, n_clones)
)
rownames(cell_clone_prob) <- cell_names
colnames(cell_clone_prob) <- clone_names

# Load pre-computed clone NN distance
cat("Loading pre-computed clone_nn_dis from step 3...\n")
clone_nn_dis <- read.csv("tests/outputs_R/clone_nn_dis.csv", stringsAsFactors = FALSE)
cat(sprintf("Clone NN pairs loaded: %d\n", nrow(clone_nn_dis)))

# Rebuild square distance matrix
clone_dis_sq <- long2square(
    long = as.data.frame(clone_nn_dis),
    row_names_from = "group1",
    col_names_from = "group2",
    values_from = "dis",
    symmetric = TRUE
)
diag(clone_dis_sq) <- 0
rownames(clone_dis_sq) <- colnames(clone_dis_sq) <- clone_names
write.csv(clone_dis_sq[1:50, 1:50], "tests/outputs_R/clone_nn_dis_sq50.csv")
cat("Saved clone_nn_dis_sq50.csv\n")

# ── Step 4: Clone OT distance (cells from first 5 clones only) ───────────────
# Note: R OT uses igraph::distances() to ALL cells (O(N^2)).
# Use only cells belonging to first 5 clones (~552 cells) for feasibility.
cat("\n=== Step 4: Clone OT distance (cells from first 5 clones) ===\n")
sub5_clones <- 1:5
cell_clone_sub5_full <- cell_clone_prob[, sub5_clones]
# Keep only cells that belong to at least one of the 5 clones
sub_cells_ot <- which(rowSums(cell_clone_sub5_full > 0) > 0)
pca_sub_ot        <- pca[sub_cells_ot, ]
cell_clone_sub_ot <- cell_clone_sub5_full[sub_cells_ot, ]
cat(sprintf("OT subset: %d cells x %d clones\n",
            nrow(pca_sub_ot), ncol(cell_clone_sub_ot)))
start_time <- Sys.time()
# Bypass future_lapply (which captures stdout and overflows) by calling internals directly.
# Workaround for R bug: graph_clone_ot_sub infinite-loops when the smallest clone == last column.
# Fix: append a dummy zero column so real clones never equal ncol(cell_clone_prob).
library(bluster)
pca_sub_mat <- as.matrix(pca_sub_ot)
cell_graph_ot <- bluster::makeSNNGraph(pca_sub_mat, k = 10, type = "number")
w <- igraph::edge_attr(cell_graph_ot, "weight")
cell_graph_ot <- igraph::set_edge_attr(cell_graph_ot, "weight", value = exp(-w))
cat(sprintf("OT cell graph: %d nodes, %d edges\n", igraph::vcount(cell_graph_ot), igraph::ecount(cell_graph_ot)))
# Add dummy zero column to avoid the ncol() == global_id infinite-loop bug
n_real_clones <- ncol(cell_clone_sub_ot)
dummy_col <- Matrix::sparseMatrix(i=integer(0), j=integer(0), x=numeric(0),
                                   dims=c(nrow(cell_clone_sub_ot), 1))
cc_with_dummy <- cbind(cell_clone_sub_ot, dummy_col)
clone_ot_raw <- Clonotrace:::graph_clone_ot_sub(
    cell_graph_ot, cc_with_dummy,
    target_clone = 1:ncol(cc_with_dummy),
    verbose = FALSE
)
# Remove rows involving the dummy column (column n_real_clones+1)
clone_ot_raw <- as.data.frame(clone_ot_raw)
colnames(clone_ot_raw) <- c("group1", "group2", "dis")
clone_ot_dis <- clone_ot_raw[clone_ot_raw$group1 <= n_real_clones &
                               clone_ot_raw$group2 <= n_real_clones, ]
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

# ── Step 8: Cell profile probabilities ──────────────────────────────────────
cat("\n=== Step 8: Cell profile probabilities ===\n")
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
rownames(cell_profile_prob) <- cell_names
cell_profile_prob_norm <- sweep(cell_profile_prob, 1, rowSums(cell_profile_prob) + 1e-12, "/")

# Cell pseudotime
cell_t <- as.numeric(cell_clone_prob %*% clone_t)
names(cell_t) <- cell_names
cell_meta$cell_t <- cell_t

write.csv(cell_meta, "tests/outputs_R/cell_meta_with_t.csv")
write.csv(as.matrix(cell_profile_prob_norm), "tests/outputs_R/cell_profile_prob.csv")

# ── Step 9: Profile enrichment ───────────────────────────────────────────────
cat("\n=== Step 9: Profile enrichment ===\n")
cell_clusters <- as.character(cell_meta$cluster)
names(cell_clusters) <- rownames(cell_meta)

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
profile_name <- colnames(cell_profile_prob_norm)[1]
cat(sprintf("Testing profile: %s\n", profile_name))

deg_result <- profile_cluster_DEG(
    profile = profile_name,
    cluster = "4",
    exprs = exprs_cl4,
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

cat("\n=== R pipeline (steps 4-10) complete ===\n")
cat("Outputs saved to tests/outputs_R/\n")
