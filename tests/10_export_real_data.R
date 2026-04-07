#!/usr/bin/env Rscript
# 10_export_real_data.R - Export hematopoiesis data from RDS to CSV for Python testing
# Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/
#
# Data source: /Users/yizhouw/Desktop/projects/clonotrace_cd4/data/
# This script exports all data needed to run the full Clonotrace pipeline
# in both R and Python for cross-validation.

setwd("/Users/yizhouw/Desktop/packages/Clonotrace_python")
dir.create("tests/real_data", showWarnings = FALSE, recursive = TRUE)

cat("=== Loading hematopoiesis data ===\n")
data_dir <- "/Users/yizhouw/Desktop/projects/clonotrace_cd4/data"

pca <- readRDS(file.path(data_dir, "hematopoiesis_pca.rds"))
cell_meta <- readRDS(file.path(data_dir, "hematopoiesis_cell_meta.rds"))

cat(sprintf("PCA: %d x %d\n", nrow(pca), ncol(pca)))
cat(sprintf("Cell meta: %d x %d\n", nrow(cell_meta), ncol(cell_meta)))
cat(sprintf("Columns: %s\n", paste(colnames(cell_meta), collapse = ", ")))

# ── Export PCA embeddings ────────────────────────────────────────────────────
cat("\n=== Exporting PCA ===\n")
write.csv(as.data.frame(pca), "tests/real_data/pca.csv")
cat(sprintf("PCA saved: %d x %d\n", nrow(pca), ncol(pca)))

# ── Export cell metadata ─────────────────────────────────────────────────────
cat("\n=== Exporting cell metadata ===\n")
write.csv(cell_meta, "tests/real_data/cell_meta.csv")
cat(sprintf("Cell meta saved: %d rows\n", nrow(cell_meta)))

# ── Export cell names ────────────────────────────────────────────────────────
cell_names <- rownames(cell_meta)
write.csv(data.frame(cell_name = cell_names), "tests/real_data/cell_names.csv",
          row.names = FALSE)

# ── Identify expanded clones ────────────────────────────────────────────────
cat("\n=== Identifying expanded clones ===\n")
library(dplyr)
clone_size <- cell_meta %>% group_by(clone) %>% summarise(count = n())
expanded_clones <- clone_size %>% filter(count >= 10)
cat(sprintf("Total clones: %d, Expanded (>=10): %d\n",
            nrow(clone_size), nrow(expanded_clones)))

write.csv(expanded_clones, "tests/real_data/expanded_clones.csv", row.names = FALSE)

# ── Export clone names and labels ────────────────────────────────────────────
clone_names <- expanded_clones$clone
write.csv(data.frame(clone_name = clone_names), "tests/real_data/clone_names.csv",
          row.names = FALSE)

# Clone labels (with non-expanded set to NA)
clone_labels <- data.frame(
    cell = rownames(cell_meta),
    clone = ifelse(cell_meta$clone %in% clone_names, cell_meta$clone, NA),
    stringsAsFactors = FALSE
)
write.csv(clone_labels, "tests/real_data/cell_clone_labels.csv", row.names = FALSE)

# ── Export binary cell-clone matrix (triplet format) ─────────────────────────
cat("\n=== Exporting cell-clone binary matrix ===\n")
# Build binary cell-clone assignment for expanded clones
cells_with_clone <- which(cell_meta$clone %in% clone_names)
clone_idx_map <- setNames(seq_along(clone_names) - 1L, clone_names)  # 0-indexed
cell_idx <- cells_with_clone - 1L  # 0-indexed
clone_idx <- clone_idx_map[cell_meta$clone[cells_with_clone]]

triplets <- data.frame(
    cell_idx = cell_idx,
    clone_idx = as.integer(clone_idx),
    value = 1.0
)
write.csv(triplets, "tests/real_data/cell_clone_binary_triplets.csv", row.names = FALSE)
cat(sprintf("Binary triplets: %d entries (%d cells, %d clones)\n",
            nrow(triplets), length(unique(triplets$cell_idx)),
            length(unique(triplets$clone_idx))))

# ── Export expression matrix for a target cluster (for DEG testing) ──────────
cat("\n=== Exporting expression data for DEG testing ===\n")
# Try to load expression from the full Seurat object
seurat_path <- file.path(data_dir, "hematopoiesis.rds")
if (file.exists(seurat_path)) {
    cat("Loading Seurat object (this may take a minute)...\n")
    seurat_object <- readRDS(seurat_path)

    # Get expression for cluster 4 cells (top 200 variable genes)
    cluster4_cells <- rownames(cell_meta)[cell_meta$cluster == "4"]
    cat(sprintf("Cluster 4 cells: %d\n", length(cluster4_cells)))

    exprs <- seurat_object@assays$RNA$data
    # Select top variable genes
    gene_var <- apply(exprs[, cluster4_cells], 1, var)
    top_genes <- names(sort(gene_var, decreasing = TRUE))[1:200]
    exprs_cl4 <- as.matrix(exprs[top_genes, cluster4_cells])
    write.csv(exprs_cl4, "tests/real_data/exprs_cluster4.csv")
    cat(sprintf("Expression saved: %d genes x %d cells\n",
                nrow(exprs_cl4), ncol(exprs_cl4)))

    # Also export UMAP coordinates
    if ("umap" %in% names(seurat_object@reductions)) {
        umap_coords <- as.data.frame(seurat_object@reductions$umap@cell.embeddings)
        write.csv(umap_coords, "tests/real_data/umap.csv")
        cat(sprintf("UMAP saved: %d x %d\n", nrow(umap_coords), ncol(umap_coords)))
    }
} else {
    cat("Seurat object not found, skipping expression export.\n")
    cat("DEG tests will use existing test data.\n")
}

cat("\n=== Data export complete ===\n")
cat("Files saved to tests/real_data/\n")
