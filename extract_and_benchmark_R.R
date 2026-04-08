# ============================================================
# Extract data from Seurat RDS + Benchmark R Clonotrace pipeline
# ============================================================
# This script:
#   1. Extracts PCA, UMAP, cell_meta, expression matrix from hematopoiesis.rds
#      into Python-readable formats (.csv, .mtx)
#   2. Runs the Clonotrace vignette pipeline with per-step timing
#   3. Saves timing results to JSON
# ============================================================

library(Matrix)
library(dplyr)
library(igraph)
library(dbscan)
library(RANN)
library(future)
library(future.apply)
library(splines)

# Use sequential plan for fair single-core comparison
plan(sequential)

# Source Clonotrace R functions
R_DIR <- "/Users/yizhouw/Desktop/packages/Clonotrace_yuntian/R"
source(file.path(R_DIR, "auxiliary.R"))
source(file.path(R_DIR, "pseudotime.R"))
source(file.path(R_DIR, "cluster.R"))
source(file.path(R_DIR, "clone_dis.R"))
source(file.path(R_DIR, "label_propagation.R"))
source(file.path(R_DIR, "profile_DEG.R"))

DATA_DIR <- "/Users/yizhouw/Desktop/packages/Clonotrace_python/real_data"
dir.create(DATA_DIR, showWarnings = FALSE)

results <- list()

# ============================================================
# STEP 0: Load Seurat object and extract data
# ============================================================
cat("Loading Seurat object...\n")
suppressPackageStartupMessages(library(Seurat))
seurat_object <- readRDS("/Users/yizhouw/Desktop/packages/Clonotrace_yuntian/hematopoiesis.rds")

pca <- seurat_object@reductions$pca@cell.embeddings
umap <- as.data.frame(seurat_object@reductions$umap@cell.embeddings)
cell_meta <- seurat_object@meta.data
exprs <- seurat_object@assays$RNA$data

cat(sprintf("  Cells: %d, PCA dims: %d, Genes: %d\n", nrow(pca), ncol(pca), nrow(exprs)))

# Save for Python (skip if already done)
if (!file.exists(file.path(DATA_DIR, "pca.csv"))) {
  cat("Extracting data for Python...\n")
  write.csv(pca, file.path(DATA_DIR, "pca.csv"))
  write.csv(umap, file.path(DATA_DIR, "umap.csv"))
  write.csv(cell_meta, file.path(DATA_DIR, "cell_meta.csv"))
  Matrix::writeMM(exprs, file.path(DATA_DIR, "exprs.mtx"))
  writeLines(rownames(exprs), file.path(DATA_DIR, "gene_names.txt"))
  writeLines(colnames(exprs), file.path(DATA_DIR, "cell_names_exprs.txt"))
  file.copy(
    "/Users/yizhouw/Desktop/packages/Clonotrace_yuntian/inst/extdata/clone_graph_dis.tsv",
    file.path(DATA_DIR, "clone_graph_dis.tsv"), overwrite = TRUE
  )
  file.copy(
    "/Users/yizhouw/Desktop/packages/Clonotrace_yuntian/inst/extdata/clone_mds.tsv",
    file.path(DATA_DIR, "clone_mds.tsv"), overwrite = TRUE
  )
  cat("Data extraction complete.\n\n")
} else {
  cat("Data already extracted, skipping.\n\n")
}

# ============================================================
# STEP 1: embedding2knn + compute_transition
# ============================================================
cat("Step 1: embedding2knn (k=30)...\n")
t0 <- proc.time()["elapsed"]
cell_knn <- embedding2knn(embedding = as.matrix(pca), k = 30, mode = "connectivity", if_self = FALSE)
t1 <- proc.time()["elapsed"]
results$embedding2knn <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs\n", t1 - t0))

cat("Step 1b: compute_transition...\n")
t0 <- proc.time()["elapsed"]
cell_knn <- compute_transition(cell_knn)
t1 <- proc.time()["elapsed"]
results$compute_transition <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs\n", t1 - t0))

# ============================================================
# STEP 2: Prepare clone labels
# ============================================================
cat("Step 2: Prepare clone labels...\n")
clone_size <- cell_meta %>% group_by(clone) %>% summarise(count = n())
expanded_clones <- clone_size %>% filter(count >= 10)
cell_clone <- data.frame(cell = rownames(cell_meta), clone = cell_meta[, "clone"]) %>%
  mutate(clone = if_else(clone %in% expanded_clones$clone, clone, NA_character_))

cat(sprintf("  Total clones: %d, Expanded (>=10 cells): %d\n",
            nrow(clone_size), nrow(expanded_clones)))

# ============================================================
# STEP 3: Label spreading bootstrap
# ============================================================
cat("Step 3: label_spreading_bootstrap (alpha=0.6, sample_n=48)...\n")
t0 <- proc.time()["elapsed"]
clone_labels <- label_spreading_bootstrap(
  adj = cell_knn, labels = cell_clone$clone,
  alpha = 0.6, sample_rate = 0.8, sample_n = 48
)
t1 <- proc.time()["elapsed"]
results$label_spreading_bootstrap <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs\n", t1 - t0))

cell_clone_prob_raw <- clone_labels[[1]]
deviance <- clone_labels[[2]]
rownames(cell_clone_prob_raw) <- rownames(cell_meta)

# ============================================================
# STEP 4: Filter by deviance + sparsify
# ============================================================
cat("Step 4: Filter deviance + mat_sparsify...\n")
t0 <- proc.time()["elapsed"]
cell_clone_prob <- cell_clone_prob_raw[deviance < 0.3, ]
cell_clone_prob <- cell_clone_prob / rowSums(cell_clone_prob)
cell_clone_prob <- mat_sparsify(mat = cell_clone_prob, row_mass = 0.9, col_mass = 0.9)
cell_clone_prob <- cell_clone_prob / rowSums(cell_clone_prob)
cell_clone_prob <- Matrix(cell_clone_prob, sparse = TRUE)
colnames(cell_clone_prob) <- names(table(cell_clone$clone))
t1 <- proc.time()["elapsed"]
results$filter_sparsify <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs, Cells remaining: %d, Clones: %d\n",
            t1 - t0, nrow(cell_clone_prob), ncol(cell_clone_prob)))

# Save cell_clone_prob for Python (so both use identical data)
Matrix::writeMM(cell_clone_prob, file.path(DATA_DIR, "cell_clone_prob.mtx"))
writeLines(rownames(cell_clone_prob), file.path(DATA_DIR, "cell_clone_prob_rows.txt"))
writeLines(colnames(cell_clone_prob), file.path(DATA_DIR, "cell_clone_prob_cols.txt"))
write.csv(data.frame(deviance = deviance), file.path(DATA_DIR, "deviance.csv"), row.names = FALSE)

# ============================================================
# STEP 5: Clone distance (use pre-computed)
# ============================================================
cat("Step 5: Load pre-computed clone distances...\n")
t0 <- proc.time()["elapsed"]
clone_dis_raw <- read.table(file.path(DATA_DIR, "clone_graph_dis.tsv"), sep = "\t")
clone_dis <- long2square(
  long = as.data.frame(clone_dis_raw),
  row_names_from = "group1", col_names_from = "group2",
  values_from = "dis", symmetric = TRUE
)
diag(clone_dis) <- 0
rownames(clone_dis) <- colnames(clone_dis) <- colnames(cell_clone_prob)
t1 <- proc.time()["elapsed"]
results$load_clone_dis <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs\n", t1 - t0))

# ============================================================
# STEP 6: Clone clustering (Louvain)
# ============================================================
cat("Step 6: leiden_dis (clone clustering, Louvain, k=20, resolution=0.5)...\n")
# Override leiden_dis to use Louvain as user requested
leiden_dis_louvain <- function(dismat, k = 10, prune.snn = 0, weight = "jaccard",
                               resolution = 1, if_umap = TRUE) {
  if (if_umap) {
    group_umap <- umap::umap(dismat, input = "dist")
    group_umap <- as.data.frame(group_umap$layout)
    colnames(group_umap) <- c("umap1", "umap2")
  }
  dis_snn <- sNN(as.dist(dismat), k = k)
  dis_snn$jaccard <- dis_snn$shared / (2 * k - dis_snn$shared)
  dis_snn_edge <- as.data.frame(cbind(
    rep(1:nrow(dismat), each = k), c(t(dis_snn$id)),
    c(t(dis_snn$dis)), c(t(dis_snn$jaccard))
  ))
  colnames(dis_snn_edge) <- c("start", "end", "dis", "jaccard")
  dis_snn_edge <- dis_snn_edge %>% filter(jaccard > prune.snn)
  colnames(dis_snn_edge)[which(colnames(dis_snn_edge) == weight)] <- "weight"
  dis_snn_graph <- graph_from_data_frame(dis_snn_edge, directed = FALSE,
                                          vertices = 1:nrow(dismat))
  cluster <- igraph::cluster_louvain(dis_snn_graph, resolution = resolution)
  if (if_umap) {
    group_umap$cluster <- as.factor(cluster$membership)
    return(group_umap)
  }
  return(as.factor(cluster$membership))
}

set.seed(1230)
t0 <- proc.time()["elapsed"]
clone_cluster <- leiden_dis_louvain(dismat = clone_dis, k = 20, resolution = 0.5, if_umap = TRUE)
t1 <- proc.time()["elapsed"]
results$clone_clustering <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs, Clusters: %d\n", t1 - t0, length(unique(clone_cluster$cluster))))

clone_cluster$mass <- colSums(cell_clone_prob)

# ============================================================
# STEP 7: Clone pseudotime
# ============================================================
cat("Step 7: clone_dpt...\n")
clone_embedding <- read.table(file.path(DATA_DIR, "clone_mds.tsv"), sep = "\t", header = TRUE)

# Need to add clone names to clone_cluster for clone_dpt
clone_cluster_with_names <- clone_cluster
clone_cluster_with_names$clone <- colnames(cell_clone_prob)

t0 <- proc.time()["elapsed"]
clone_t <- clone_dpt(
  clone_embedding = clone_embedding,
  cell_meta = cell_meta,
  clone_col = "clone", cluster_col = "cluster",
  start_cluster = "0"
)
t1 <- proc.time()["elapsed"]
results$clone_dpt <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs\n", t1 - t0))

clone_cluster_with_names$dpt <- clone_t

# Smooth to cell level
cell_meta$cell_t <- NA
cell_meta[rownames(cell_clone_prob), ]$cell_t <-
  cell_clone_prob[, clone_cluster_with_names$clone] %*% clone_cluster_with_names$dpt

# ============================================================
# STEP 8: Profile assignment
# ============================================================
cat("Step 8: Profile assignment...\n")
t0 <- proc.time()["elapsed"]
clone_profile <- clone_cluster_with_names %>%
  dplyr::select(clone, cluster) %>%
  mutate(flag = 1)
clone_profile <- long2sparse(
  long = clone_profile,
  row_names_from = "clone", col_names_from = "cluster",
  values_from = "flag"
)
cell_profile_prob <- cell_clone_prob %*% clone_profile[colnames(cell_clone_prob), ]
rownames(cell_profile_prob) <- rownames(cell_clone_prob)
t1 <- proc.time()["elapsed"]
results$profile_assignment <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs\n", t1 - t0))

# Assign hard labels
cell_meta$profile <- NA
cell_meta[rownames(cell_profile_prob), ]$profile <- apply(cell_profile_prob, 1, function(x) {
  if (max(x) > 0.5) which.max(x) else NA
})

# ============================================================
# STEP 9: Profile enrichment
# ============================================================
cat("Step 9: cluster_profile_enrich (permute_n=300)...\n")
t0 <- proc.time()["elapsed"]
enrich <- cluster_profile_enrich(
  cell_profile_prob,
  cell_meta[rownames(cell_profile_prob), "cluster"],
  permute_n = 300
)
t1 <- proc.time()["elapsed"]
results$cluster_profile_enrich <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs\n", t1 - t0))

# ============================================================
# STEP 10: Profile DEG
# ============================================================
cat("Step 10: profile_cluster_DEG (profile=1, cluster=4)...\n")
t0 <- proc.time()["elapsed"]
DEG_result <- profile_cluster_DEG(
  profile = "1", cluster = "4",
  exprs = exprs, cell_meta = cell_meta,
  cell_profile_prob = cell_profile_prob,
  permute_n = 50
)
t1 <- proc.time()["elapsed"]
results$profile_cluster_DEG <- list(time = t1 - t0)
cat(sprintf("  Time: %.2fs\n", t1 - t0))

# ============================================================
# Save timing results
# ============================================================
cat("\nSaving R timing results...\n")
results_json <- jsonlite::toJSON(results, auto_unbox = TRUE, pretty = TRUE)
writeLines(results_json, "/Users/yizhouw/Desktop/packages/Clonotrace_python/real_data_R_results.json")
cat("Done!\n")

# Print summary
cat("\n=== R Pipeline Timing Summary ===\n")
total <- 0
for (name in names(results)) {
  t <- results[[name]]$time
  total <- total + t
  cat(sprintf("  %-30s %7.2fs\n", name, t))
}
cat(sprintf("  %-30s %7.2fs\n", "TOTAL", total))
