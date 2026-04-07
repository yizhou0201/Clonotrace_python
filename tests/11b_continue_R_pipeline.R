#!/usr/bin/env Rscript
# 11b_continue_R_pipeline.R - Continue from step 5 (after step 4 completed)
# Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/

setwd("/Users/yizhouw/Desktop/packages/Clonotrace_python")

library(Clonotrace)
library(Matrix)
library(MASS)
library(dplyr)
library(future)
library(future.apply)

options(future.globals.maxSize = 4 * 1024^3)
options(future.stdout = NA)  # Prevent long-vector stdout error

cat("=== Reloading data and previous outputs ===\n")
pca <- as.matrix(read.csv("tests/real_data/pca.csv", row.names = 1))
cell_meta <- read.csv("tests/real_data/cell_meta.csv", row.names = 1, stringsAsFactors = FALSE)
cell_names <- read.csv("tests/real_data/cell_names.csv", stringsAsFactors = FALSE)$cell_name
clone_names <- read.csv("tests/real_data/clone_names.csv", stringsAsFactors = FALSE)$clone_name
n_cells <- length(cell_names)
n_clones <- length(clone_names)

# Reload filtered cell_clone_prob
ccp_triplets <- read.csv("tests/real_outputs_R/cell_clone_prob_filtered_triplets.csv")
ccp_cells <- read.csv("tests/real_outputs_R/cell_clone_prob_filtered_cells.csv", stringsAsFactors = FALSE)$cell
cell_clone_prob <- sparseMatrix(
    i = ccp_triplets$row, j = ccp_triplets$col, x = ccp_triplets$value,
    dims = c(length(ccp_cells), n_clones)
)
rownames(cell_clone_prob) <- ccp_cells
colnames(cell_clone_prob) <- clone_names

# Reload clone_nn_dis
clone_nn_dis <- read.csv("tests/real_outputs_R/clone_nn_dis.csv")
clone_dis_sq <- long2square(
    long = as.data.frame(clone_nn_dis),
    row_names_from = "group1", col_names_from = "group2",
    values_from = "dis", symmetric = TRUE
)
diag(clone_dis_sq) <- 0
rownames(clone_dis_sq) <- colnames(clone_dis_sq) <- clone_names

cat(sprintf("Loaded: %d cells, %d clones, %d NN pairs\n",
            nrow(cell_clone_prob), n_clones, nrow(clone_nn_dis)))

# ── Step 5: Clone OT distance (first 10 clones — reduced for speed) ─────────
cat("\n=== Step 5: Clone OT distance (first 10 clones) ===\n")
n_ot_clones <- 10
cell_clone_sub <- cell_clone_prob[, 1:n_ot_clones]

# Remove empty rows to reduce graph size
row_sums <- rowSums(cell_clone_sub)
keep <- row_sums > 0
cell_clone_sub <- cell_clone_sub[keep, ]
pca_sub <- pca[rownames(cell_clone_sub), ]

cat(sprintf("OT subset: %d cells x %d clones\n", nrow(cell_clone_sub), ncol(cell_clone_sub)))

start_time <- Sys.time()
options(future.stdout = FALSE)  # Disable stdout capture entirely to avoid long vector error
plan(sequential)
clone_ot_dis <- clone_disance(
    as.matrix(pca_sub),
    cell_clone_sub,
    outpath = "tests/real_outputs_R/clone_ot_cache/",
    exact = TRUE,
    cores = 1,
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
clone_cluster <- leiden_dis(dismat = clone_dis_sq, k = 20, resolution = 0.5, if_umap = TRUE)
rownames(clone_cluster) <- clone_names
write.csv(clone_cluster, "tests/real_outputs_R/clone_cluster.csv")
cat(sprintf("Clusters: %d unique\n", length(unique(clone_cluster$cluster))))

# ── Step 8: Clone pseudotime ────────────────────────────────────────────────
cat("\n=== Step 8: Clone pseudotime ===\n")
clone_t <- clone_dpt(
    clone_embedding = as.data.frame(clone_embedding),
    cell_meta = cell_meta,
    clone_col = "clone", cluster_col = "cluster", start_cluster = "0"
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

cell_t <- as.numeric(cell_clone_prob[, clone_names] %*% clone_t)
names(cell_t) <- rownames(cell_clone_prob)
cell_meta_sub <- cell_meta[rownames(cell_clone_prob), ]
cell_meta_sub$cell_t <- cell_t
write.csv(cell_meta_sub, "tests/real_outputs_R/cell_meta_with_t.csv")
write.csv(as.matrix(cell_profile_prob_norm), "tests/real_outputs_R/cell_profile_prob.csv")

# ── Step 10: Profile enrichment ─────────────────────────────────────────────
cat("\n=== Step 10: Profile enrichment ===\n")
cell_clusters <- as.character(cell_meta_sub$cluster)
enrich <- cluster_profile_enrich(
    cell_profile_prob = as.matrix(cell_profile_prob_norm),
    cluster_label = cell_clusters, permute_n = 300
)
write.csv(enrich[[1]], "tests/real_outputs_R/enrich_mass.csv")
write.csv(enrich[[2]], "tests/real_outputs_R/enrich_pval.csv")
cat(sprintf("Enrichment: %d clusters x %d profiles\n", nrow(enrich[[1]]), ncol(enrich[[1]])))

# ── Step 11: Profile DEG ────────────────────────────────────────────────────
cat("\n=== Step 11: Profile DEG ===\n")
exprs_path <- "tests/real_data/exprs_cluster4.csv"
if (file.exists(exprs_path)) {
    exprs_cl4 <- as.matrix(read.csv(exprs_path, row.names = 1))
    profile_name <- cluster_order[1]
    cat(sprintf("Testing profile: %s\n", profile_name))
    deg_result <- profile_cluster_DEG(
        profile = profile_name, cluster = "4",
        exprs = exprs_cl4, cell_meta = cell_meta_sub,
        cell_profile_prob = as.data.frame(as.matrix(cell_profile_prob_norm))
    )
    if (!is.null(deg_result)) {
        write.csv(deg_result$stat, "tests/real_outputs_R/DEG_stats.csv")
        cat(sprintf("DEG result: %d genes\n", nrow(deg_result$stat)))
    } else { cat("DEG returned NULL\n") }
} else { cat("Expression data not found.\n") }

# ── Step 12: SNN graph weights ──────────────────────────────────────────────
cat("\n=== Step 12: Export SNN graph weights ===\n")
library(bluster)
snn_graph <- makeSNNGraph(pca[1:500, ], k = 10, type = "number")
snn_w <- igraph::edge_attr(snn_graph, "weight")
snn_edges <- igraph::as_edgelist(snn_graph)
write.csv(data.frame(from = snn_edges[, 1], to = snn_edges[, 2],
                      shared_count = snn_w, weight_exp = exp(-snn_w)),
          "tests/real_outputs_R/snn_graph_weights.csv", row.names = FALSE)
cat(sprintf("SNN graph edges: %d\n", length(snn_w)))

cat("\n=== All remaining R pipeline steps complete ===\n")
