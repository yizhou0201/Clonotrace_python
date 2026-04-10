# Benchmark R's clone distance computation (NN method)
# Uses the same cell_clone_prob and PCA from the real dataset

library(Matrix)
library(igraph)
library(RANN)
library(dbscan)

R_DIR <- "/Users/yizhouw/Desktop/packages/Clonotrace_yuntian/R"
source(file.path(R_DIR, "clone_dis.R"))
source(file.path(R_DIR, "auxiliary.R"))

DATA_DIR <- "/Users/yizhouw/Desktop/packages/Clonotrace_python/real_data"

# Load data
cat("Loading data...\n")
pca <- read.csv(file.path(DATA_DIR, "pca.csv"), row.names = 1)
cell_clone_prob <- readMM(file.path(DATA_DIR, "cell_clone_prob.mtx"))
prob_rows <- readLines(file.path(DATA_DIR, "cell_clone_prob_rows.txt"))
prob_cols <- readLines(file.path(DATA_DIR, "cell_clone_prob_cols.txt"))
rownames(cell_clone_prob) <- prob_rows
colnames(cell_clone_prob) <- prob_cols

# Align PCA to cell_clone_prob rows
pca_aligned <- as.matrix(pca[prob_rows, ])
cat(sprintf("  Cells: %d, Clones: %d\n", nrow(pca_aligned), ncol(cell_clone_prob)))

# Step 1: Build SNN graph (matching Python's _build_snn_graph)
cat("\nStep 1: Build SNN graph (k=10)...\n")
t0 <- proc.time()["elapsed"]

# Use bluster-style SNN graph
nn <- nn2(pca_aligned, k = 11)  # k+1 for self
nn_idx <- nn$nn.idx[, -1]  # remove self

n <- nrow(pca_aligned)
k <- 10

# Build adjacency with self included (matching R's bluster behavior)
rows <- rep(1:n, each = k + 1)
cols <- c(t(cbind(1:n, nn_idx)))
adj <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n, n))

# SNN: shared neighbor count
shared <- adj %*% t(adj)
shared_coo <- summary(shared)
# Upper triangle only
mask <- shared_coo$i < shared_coo$j & shared_coo$x > 0
edges_from <- shared_coo$i[mask]
edges_to <- shared_coo$j[mask]
weights <- exp(-shared_coo$x[mask])

cell_graph <- make_empty_graph(n = n, directed = FALSE)
cell_graph <- add_edges(cell_graph, c(rbind(edges_from, edges_to)))
E(cell_graph)$weight <- weights

t1 <- proc.time()["elapsed"]
cat(sprintf("  Time: %.2fs\n", t1 - t0))
cat(sprintf("  Nodes: %d, Edges: %d\n", vcount(cell_graph), ecount(cell_graph)))

# Step 2: Clone NN distances
cat("\nStep 2: Clone NN distances (802 clones)...\n")
cat("  (Using R's graph_clone_nn equivalent)\n")

# R implementation of graph_clone_nn
prob_thresh <- 0.1
knn <- 2

cell_group_mat <- as.matrix(cell_clone_prob)
cell_group_mat <- (cell_group_mat >= prob_thresh) * 1.0
n_groups <- ncol(cell_group_mat)

t0 <- proc.time()["elapsed"]

# Process first 20 clones for timing estimate
results_20 <- list()
for (i in 1:min(20, n_groups - 1)) {
  from_cells <- which(cell_group_mat[, i] > 0)
  if (length(from_cells) == 0) next

  to_cells <- which(rowSums(cell_group_mat[, (i + 1):n_groups, drop = FALSE]) > 0)
  if (length(to_cells) == 0) next

  dis_i <- distances(cell_graph, v = from_cells, to = to_cells, weights = E(cell_graph)$weight)

  for (j in (i + 1):min(20, n_groups)) {
    id_j <- which(cell_group_mat[, j] > 0)
    to_in_j <- which(to_cells %in% id_j)
    if (length(to_in_j) == 0) next
    sub_dis <- dis_i[, to_in_j, drop = FALSE]
    # group_2_min equivalent
    row_means <- mean(apply(sub_dis, 1, function(row) mean(sort(row)[1:min(knn, length(row))])))
    col_means <- mean(apply(sub_dis, 2, function(col) mean(sort(col)[1:min(knn, length(col))])))
    d <- mean(c(row_means, col_means))
    results_20[[length(results_20) + 1]] <- c(i, j, d)
  }
}
t1 <- proc.time()["elapsed"]
t_20 <- t1 - t0
cat(sprintf("  20 clones: %.2fs (%d pairs)\n", t_20, length(results_20)))
cat(sprintf("  Estimated full 802 clones: %.0fs\n", t_20 * n_groups / 20))

# Save results
results_json <- jsonlite::toJSON(list(
  snn_graph = list(time = t1 - t0),
  clone_nn_20 = list(time = t_20, pairs = length(results_20)),
  estimated_full = list(time = t_20 * n_groups / 20)
), auto_unbox = TRUE, pretty = TRUE)
writeLines(results_json, "/Users/yizhouw/Desktop/packages/Clonotrace_python/clone_dis_R_results.json")
cat("\nDone! Results saved.\n")
