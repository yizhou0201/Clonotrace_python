#!/usr/bin/env Rscript
# 01_export_data.R - Export data from hematopoiesis.rds for comparison tests
# Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/

setwd("/Users/yizhouw/Desktop/packages/Clonotrace_python")
dir.create("tests/data", showWarnings = FALSE, recursive = TRUE)

library(Seurat)
library(Clonotrace)

cat("Loading hematopoiesis.rds...\n")
sobj <- readRDS("/Users/yizhouw/Desktop/packages/Clonotrace_yuntian/vignettes/hematopoiesis.rds")

# PCA embeddings
pca <- sobj@reductions$pca@cell.embeddings
cat(sprintf("PCA: %d cells x %d dims\n", nrow(pca), ncol(pca)))
write.csv(pca, "tests/data/pca.csv")

# Cell metadata
cell_meta <- sobj@meta.data
cat(sprintf("Cell meta columns: %s\n", paste(colnames(cell_meta), collapse = ", ")))
write.csv(cell_meta, "tests/data/cell_meta.csv")

# Expanded clones (>=10 cells) - sorted by size descending
clone_counts <- table(cell_meta$clone)
expanded_clones <- names(sort(clone_counts[clone_counts >= 10], decreasing = TRUE))
cat(sprintf("Expanded clones: %d\n", length(expanded_clones)))
writeLines(expanded_clones, "tests/data/expanded_clones.txt")

# Cell-clone hard assignment (NA if not in expanded_clones)
clone_label <- rep(NA_character_, nrow(cell_meta))
names(clone_label) <- rownames(cell_meta)
mask <- !is.na(cell_meta$clone) & (cell_meta$clone %in% expanded_clones)
clone_label[mask] <- as.character(cell_meta$clone[mask])
label_df <- data.frame(cell = rownames(cell_meta), clone = clone_label, stringsAsFactors = FALSE)
write.csv(label_df, "tests/data/cell_clone_labels.csv", row.names = FALSE)

# Cell names and clone names (needed for sparse matrix reconstruction)
write.csv(data.frame(cell_name = rownames(cell_meta)), "tests/data/cell_names.csv", row.names = FALSE)
write.csv(data.frame(clone_name = expanded_clones), "tests/data/clone_names.csv", row.names = FALSE)

# Cell-clone binary assignment as 0-based sparse triplets for Python
rows_vec <- c(); cols_vec <- c()
for (ci in seq_along(expanded_clones)) {
    cl <- expanded_clones[ci]
    cell_idx <- which(!is.na(cell_meta$clone) & cell_meta$clone == cl)
    rows_vec <- c(rows_vec, cell_idx - 1L)   # 0-based
    cols_vec <- c(cols_vec, rep(ci - 1L, length(cell_idx)))
}
triplets <- data.frame(cell_idx = rows_vec, clone_idx = cols_vec, value = 1.0)
write.csv(triplets, "tests/data/cell_clone_binary_triplets.csv", row.names = FALSE)
cat(sprintf("Binary triplets: %d entries (cells x clones)\n", nrow(triplets)))

# Expression matrix for cluster 4 (top 200 HVGs by variance in cluster 4)
cat("Extracting cluster 4 expression matrix...\n")
cluster_col <- if ("cluster" %in% colnames(cell_meta)) "cluster" else "seurat_clusters"
cluster4_cells <- rownames(cell_meta)[as.character(cell_meta[[cluster_col]]) == "4"]
cat(sprintf("Cluster 4 cells: %d\n", length(cluster4_cells)))

exprs_full <- tryCatch(
    GetAssayData(sobj, layer = "data", assay = "RNA"),
    error = function(e) GetAssayData(sobj, slot = "data", assay = "RNA")
)
exprs_cl4 <- exprs_full[, cluster4_cells]
gene_var <- apply(exprs_cl4, 1, var)
top_genes <- names(sort(gene_var, decreasing = TRUE))[1:200]
exprs_sub <- as.matrix(exprs_cl4[top_genes, ])
cat(sprintf("Expression matrix: %d genes x %d cells\n", nrow(exprs_sub), ncol(exprs_sub)))
write.csv(exprs_sub, "tests/data/exprs_cluster4.csv")

cat("\nDone! Data exported to tests/data/\n")
cat("Files written:\n")
cat("  pca.csv, cell_meta.csv, expanded_clones.txt\n")
cat("  cell_clone_labels.csv, cell_names.csv, clone_names.csv\n")
cat("  cell_clone_binary_triplets.csv, exprs_cluster4.csv\n")
