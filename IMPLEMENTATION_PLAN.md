# Technical Implementation Specification: Rust-based Graph Cluster Analysis System

## 1. Project Structure

```
graph-cluster-analyzer/
├── Cargo.toml
├── build.rs               # Custom build for CUDA integration
├── src/
│   ├── main.rs            # CLI entry point
│   ├── lib.rs             # Core library functions
│   ├── config.rs          # Configuration management
│   ├── data/              # Data loading and processing
│   │   ├── mod.rs
│   │   ├── parquet.rs     # Parquet file handling
│   │   └── preprocessing.rs
│   ├── graph/             # Graph representation
│   │   ├── mod.rs
│   │   ├── compressed.rs  # Memory-efficient graph structures
│   │   ├── builder.rs     # Graph construction
│   │   └── algorithms.rs  # Graph algorithms
│   ├── cluster/           # Cluster analysis
│   │   ├── mod.rs
│   │   ├── detection.rs   # Cluster identification
│   │   └── metrics.rs     # Cluster statistics
│   ├── gpu/               # GPU acceleration
│   │   ├── mod.rs
│   │   ├── context.rs     # CUDA context management
│   │   ├── kernels.rs     # Rust-side CUDA kernel wrappers
│   │   └── memory.rs      # GPU memory management
│   ├── storage/           # Results persistence
│   └── viz/               # Visualization generation
└── cuda/                  # CUDA kernel implementations
    ├── kernels.cu
    ├── mutual_follows.cu
    └── connected_components.cu
```

## 2. Core Dependencies

```toml
[dependencies]
# Data processing
arrow = "37.0.0"
parquet = "37.0.0"
polars = { version = "0.36.2", features = ["parquet", "lazy"] }

# Parallelism
rayon = "1.8.0"
crossbeam = "0.8.2"
dashmap = "5.5.3"

# GPU acceleration
rustacuda = "0.1.3"
rustacuda_core = "0.1.2"
rustacuda_derive = "0.1.2"

# Graph processing
petgraph = "0.6.4"

# Memory management
memmap2 = "0.9.0"
bumpalo = "3.14.0"

# Serialization and storage
serde = { version = "1.0.193", features = ["derive"] }
serde_json = "1.0.108"
bincode = "1.3.3"

# Error handling
thiserror = "1.0.50"
anyhow = "1.0.75"

# Utilities
itertools = "0.12.0"
bytes = "1.5.0"
ndarray = "0.15.6"
num_cpus = "1.16.0"
statrs = "0.16.0"

# Logging and diagnostics
log = "0.4.20"
env_logger = "0.10.1"
tracing = "0.1.40"

# CLI
clap = { version = "4.4.8", features = ["derive"] }

[build-dependencies]
cc = "1.0.83"
```

## 3. Data Structures

### 3.1 Memory-Efficient Graph Representation

```rust
/// Compressed sparse representation of a directed graph optimized for memory efficiency
pub struct CompressedGraph {
    /// Number of nodes in the graph
    pub node_count: usize,
    
    /// Offset array: index where each node's edges begin
    /// offsets[i] to offsets[i+1] defines the edge range for node i
    pub offsets: Vec<u32>,
    
    /// Edge array: concatenated lists of target nodes
    pub edges: Vec<u32>,
    
    /// Optional mapping from internal node IDs to original string IDs
    pub node_ids: Option<Vec<String>>,
    
    /// Optional node metadata (stored separately for cache efficiency)
    pub metadata: Option<NodeMetadata>,
}

/// Store node metadata separately to improve cache locality during traversal
pub struct NodeMetadata {
    /// Number of followers per node
    pub follower_counts: Vec<u32>,
    
    /// Number of following per node
    pub following_counts: Vec<u32>,
}

impl CompressedGraph {
    /// Create a new graph with pre-allocated capacity
    pub fn with_capacity(node_count: usize, edge_count: usize) -> Self {
        Self {
            node_count,
            offsets: Vec::with_capacity(node_count + 1),
            edges: Vec::with_capacity(edge_count),
            node_ids: None,
            metadata: None,
        }
    }
    
    /// Get outgoing edges for a node
    pub fn outgoing_edges(&self, node: usize) -> &[u32] {
        let start = self.offsets[node] as usize;
        let end = self.offsets[node + 1] as usize;
        &self.edges[start..end]
    }
    
    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let offsets = self.offsets.capacity() * std::mem::size_of::<u32>();
        let edges = self.edges.capacity() * std::mem::size_of::<u32>();
        
        // Add metadata if present
        let ids = self.node_ids.as_ref()
            .map(|ids| ids.iter().map(|s| s.capacity()).sum::<usize>())
            .unwrap_or(0);
            
        let metadata = self.metadata.as_ref()
            .map(|m| m.memory_usage())
            .unwrap_or(0);
            
        base + offsets + edges + ids + metadata
    }
}
```

### 3.2 Cluster Representation

```rust
/// Represents a cluster (connected component) in the graph
pub struct Cluster {
    /// Unique identifier for this cluster
    pub id: u32,
    
    /// Members of this cluster (node indices)
    pub members: Vec<u32>,
    
    /// Size of the cluster
    pub size: usize,
    
    /// Density: actual edges / potential edges
    pub density: f32,
    
    /// Central nodes using various centrality measures
    pub central_nodes: ClusterCentralNodes,
}

/// Key nodes in a cluster identified by different centrality measures
pub struct ClusterCentralNodes {
    /// Nodes with highest degree centrality
    pub degree: Vec<u32>,
    
    /// Nodes with highest betweenness centrality (if computed)
    pub betweenness: Option<Vec<u32>>,
    
    /// Nodes with highest closeness centrality (if computed)
    pub closeness: Option<Vec<u32>>,
}

/// Union-Find data structure for efficient connected component analysis
pub struct DisjointSets {
    /// Parent pointers (parent[i] = parent of node i)
    parent: Vec<u32>,
    
    /// Rank/size of each set (for union by rank)
    rank: Vec<u32>,
}
```

## 4. Core Algorithms

### 4.1 Optimized Parquet Loading

```rust
/// Load Farcaster links data with minimal memory usage
pub fn load_follow_data(
    path: &str,
    min_followings: usize,
    sample_ratio: f32,
    chunk_size: usize,
) -> Result<CompressedGraph> {
    // Create polars lazy frame with optimized predicates
    let lazy_df = LazyFrame::scan_parquet(
        path,
        ScanArgsParquet::default()
            .with_parallel(true)
            .with_row_count(None)
    )?
    .filter(col("LinkType").eq(lit("follow")));
    
    // If sampling is requested, apply it
    let lazy_df = if sample_ratio < 1.0 {
        lazy_df.sample_frac(sample_ratio, false, None)
    } else {
        lazy_df
    };
    
    // First compute all users with sufficient following count
    log::info!("Finding users with {} or more followings...", min_followings);
    let follow_counts = lazy_df
        .clone()
        .group_by([col("Fid")])
        .agg([count().alias("following_count")])
        .filter(col("following_count").gt_eq(lit(min_followings)))
        .collect()?;
    
    // Extract FIDs of active users
    let active_users: HashSet<String> = follow_counts["Fid"]
        .str()?
        .to_vec()
        .into_iter()
        .map(|v| v.unwrap_or_default())
        .collect();
    
    log::info!("Found {} users with sufficient activity", active_users.len());
    
    // Build string ID to index mapping
    let mut id_to_index: HashMap<String, u32> = HashMap::with_capacity(active_users.len() * 2);
    let mut node_ids: Vec<String> = Vec::with_capacity(active_users.len() * 2);
    
    // Process the data to construct the graph
    log::info!("Building compressed graph representation...");
    
    // First, determine the degree of each node to preallocate
    let mut temp_degrees: Vec<u32> = Vec::new();
    let mut next_id: u32 = 0;
    
    // Process in chunks to limit memory usage
    let mut streaming_df = lazy_df
        .filter(col("Fid").is_in(lit(Series::from_iter(active_users.iter()))))
        .collect()?;
    
    // First pass: count degrees and assign node IDs
    for batch in streaming_df.iter_chunks(chunk_size) {
        let fids = batch["Fid"].str()?;
        let target_fids = batch["TargetFid"].str()?;
        
        for i in 0..batch.height() {
            let src = fids.get(i).unwrap_or_default();
            let dst = target_fids.get(i).unwrap_or_default();
            
            // Ensure both source and target have node IDs
            let src_idx = *id_to_index.entry(src.to_string()).or_insert_with(|| {
                let idx = next_id;
                next_id += 1;
                node_ids.push(src.to_string());
                temp_degrees.resize(next_id as usize, 0);
                idx
            });
            
            // Only count the target if we'll include it in our graph
            let _ = id_to_index.entry(dst.to_string()).or_insert_with(|| {
                let idx = next_id;
                next_id += 1;
                node_ids.push(dst.to_string());
                temp_degrees.resize(next_id as usize, 0);
                idx
            });
            
            // Increment degree for this node
            temp_degrees[src_idx as usize] += 1;
        }
    }
    
    // Allocate graph structure using exact sizes
    let node_count = next_id as usize;
    let edge_count: usize = temp_degrees.iter().map(|&d| d as usize).sum();
    
    log::info!("Allocating graph with {} nodes and {} edges", node_count, edge_count);
    
    let mut graph = CompressedGraph::with_capacity(node_count, edge_count);
    
    // Set up offsets
    graph.offsets.push(0);
    let mut offset = 0;
    for &degree in &temp_degrees {
        offset += degree;
        graph.offsets.push(offset);
    }
    
    // Create temporary counters for current insertion positions
    let mut current_offsets = vec![0; node_count];
    
    // Second pass: fill the edge array
    streaming_df = lazy_df
        .filter(col("Fid").is_in(lit(Series::from_iter(active_users.iter()))))
        .collect()?;
    
    // Resize edges array to final size
    graph.edges.resize(edge_count, 0);
    
    for batch in streaming_df.iter_chunks(chunk_size) {
        let fids = batch["Fid"].str()?;
        let target_fids = batch["TargetFid"].str()?;
        
        for i in 0..batch.height() {
            let src = fids.get(i).unwrap_or_default();
            let dst = target_fids.get(i).unwrap_or_default();
            
            // Get indices (they should already exist)
            if let (Some(&src_idx), Some(&dst_idx)) = (id_to_index.get(src), id_to_index.get(dst)) {
                // Calculate position in edge array
                let pos = graph.offsets[src_idx as usize] as usize + current_offsets[src_idx as usize];
                graph.edges[pos] = dst_idx;
                current_offsets[src_idx as usize] += 1;
            }
        }
    }
    
    // Set node metadata
    graph.node_count = node_count;
    graph.node_ids = Some(node_ids);
    
    // Sort edges for each node (optimizes later operations)
    graph.sort_adjacency_lists();
    
    Ok(graph)
}
```

### 4.2 Finding Mutual Follows with GPU Acceleration

```rust
/// Find mutual follow relationships using CUDA acceleration
pub fn find_mutual_follows_gpu(
    graph: &CompressedGraph,
    chunk_size: usize,
) -> Result<CompressedGraph> {
    log::info!("Finding mutual follows using GPU acceleration");
    
    // Initialize CUDA
    let _context = CudaContext::new(0)?;
    
    // Allocate device memory for the graph structure
    let d_offsets = DeviceBuffer::from_slice(&graph.offsets)?;
    let d_edges = DeviceBuffer::from_slice(&graph.edges)?;
    
    // Allocate output buffer for mutual follows
    // We'll use a counter and a buffer for each node
    let node_count = graph.node_count;
    let mut d_mutual_counts = DeviceBuffer::<u32>::zeroed(node_count)?;
    
    // Create host buffer for results
    let mut h_mutual_counts = vec![0u32; node_count];
    
    // First kernel: count mutual follows for each node
    let block_size = 256;
    let grid_size = (node_count as u32 + block_size - 1) / block_size;
    
    unsafe {
        // Launch kernel to count mutual follows
        launch!(
            _context.module.count_mutual_follows<<<grid_size, block_size, 0, _context.stream>>>(
                d_offsets.as_device_ptr(),
                d_edges.as_device_ptr(),
                d_mutual_counts.as_device_ptr(),
                node_count as u32
            )
        )?;
    }
    
    // Copy results back to host
    d_mutual_counts.copy_to(&mut h_mutual_counts)?;
    
    // Calculate total mutual follows and prepare offsets
    let mut mutual_offsets = Vec::with_capacity(node_count + 1);
    mutual_offsets.push(0);
    
    let mut current_offset = 0;
    for &count in &h_mutual_counts {
        current_offset += count;
        mutual_offsets.push(current_offset);
    }
    
    let total_mutual_edges = current_offset as usize;
    log::info!("Found {} mutual follow relationships", total_mutual_edges);
    
    // Allocate buffer for mutual edges
    let mut d_mutual_offsets = DeviceBuffer::from_slice(&mutual_offsets)?;
    let mut d_mutual_edges = DeviceBuffer::<u32>::zeroed(total_mutual_edges as usize)?;
    let mut h_mutual_edges = vec![0u32; total_mutual_edges as usize];
    
    // Second kernel: fill in the mutual edges
    unsafe {
        launch!(
            _context.module.populate_mutual_follows<<<grid_size, block_size, 0, _context.stream>>>(
                d_offsets.as_device_ptr(),
                d_edges.as_device_ptr(),
                d_mutual_offsets.as_device_ptr(),
                d_mutual_edges.as_device_ptr(),
                node_count as u32
            )
        )?;
    }
    
    // Copy mutual edges back to host
    d_mutual_edges.copy_to(&mut h_mutual_edges)?;
    
    // Create mutual graph
    let mutual_graph = CompressedGraph {
        node_count,
        offsets: mutual_offsets,
        edges: h_mutual_edges,
        node_ids: graph.node_ids.clone(),
        metadata: graph.metadata.clone(),
    };
    
    Ok(mutual_graph)
}

/// CUDA context and module management
pub struct CudaContext {
    pub context: Context,
    pub stream: Stream,
    pub module: Module,
}

impl CudaContext {
    pub fn new(device_id: usize) -> Result<Self> {
        // Initialize CUDA
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id)?;
        
        // Create context with default flags
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, 
            device
        )?;
        
        // Create stream
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        // Load module from PTX
        let module_data = CString::new(include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx")))?;
        let module = Module::load_from_string(&module_data)?;
        
        Ok(Self {
            context,
            stream,
            module,
        })
    }
}
```

### 4.3 Connected Component Analysis with GPU

```rust
/// Find connected components (clusters) in the mutual follow graph
pub fn find_connected_components_gpu(
    graph: &CompressedGraph,
    min_cluster_size: usize,
) -> Result<Vec<Cluster>> {
    log::info!("Finding connected components using GPU acceleration");
    
    let node_count = graph.node_count;
    
    // Initialize CUDA
    let _context = CudaContext::new(0)?;
    
    // Allocate device memory for the graph
    let d_offsets = DeviceBuffer::from_slice(&graph.offsets)?;
    let d_edges = DeviceBuffer::from_slice(&graph.edges)?;
    
    // Allocate arrays for component labeling
    let mut d_labels = DeviceBuffer::<u32>::from_slice(&(0..node_count as u32).collect::<Vec<_>>())?;
    let mut h_labels = vec![0u32; node_count];
    
    // Initialize iteration counters
    let mut changed = true;
    let mut iterations = 0;
    let max_iterations = (node_count as f64).log2() as usize + 1;
    
    // Prepare kernel launch parameters
    let block_size = 256;
    let grid_size = (node_count as u32 + block_size - 1) / block_size;
    
    // Allocate a flag to track changes
    let mut d_changed = DeviceBuffer::<u32>::from_slice(&[0])?;
    let mut h_changed = vec![0u32; 1];
    
    // Perform label propagation until convergence
    while changed && iterations < max_iterations {
        changed = false;
        
        // Reset change flag
        h_changed[0] = 0;
        d_changed.copy_from(&h_changed)?;
        
        unsafe {
            // Launch connected components kernel
            launch!(
                _context.module.label_propagation<<<grid_size, block_size, 0, _context.stream>>>(
                    d_offsets.as_device_ptr(),
                    d_edges.as_device_ptr(),
                    d_labels.as_device_ptr(),
                    d_changed.as_device_ptr(),
                    node_count as u32
                )
            )?;
        }
        
        // Check if any labels changed
        d_changed.copy_to(&mut h_changed)?;
        changed = h_changed[0] > 0;
        iterations += 1;
        
        if iterations % 5 == 0 {
            log::info!("Connected components iteration {}", iterations);
        }
    }
    
    log::info!("Connected components converged after {} iterations", iterations);
    
    // Copy final labels back to host
    d_labels.copy_to(&mut h_labels)?;
    
    // Group nodes by component
    let mut component_map: HashMap<u32, Vec<u32>> = HashMap::new();
    for (i, &label) in h_labels.iter().enumerate() {
        component_map.entry(label).or_default().push(i as u32);
    }
    
    // Filter by minimum size
    let mut clusters: Vec<Cluster> = component_map
        .into_iter()
        .filter(|(_, members)| members.len() >= min_cluster_size)
        .enumerate()
        .map(|(id, (_, members))| {
            Cluster {
                id: id as u32,
                size: members.len(),
                members,
                density: calculate_cluster_density(graph, &members),
                central_nodes: identify_central_nodes(graph, &members),
            }
        })
        .collect();
    
    // Sort clusters by size (largest first)
    clusters.sort_by(|a, b| b.size.cmp(&a.size));
    
    log::info!(
        "Found {} clusters with {} or more members",
        clusters.len(),
        min_cluster_size
    );
    
    Ok(clusters)
}
```

## 5. Memory Management Strategy

### 5.1 Custom Memory Allocations

```rust
/// Memory arena for efficient temporary allocations
pub struct Arena<'a> {
    /// The memory buffer
    buffer: &'a mut [u8],
    /// Current position
    position: usize,
}

impl<'a> Arena<'a> {
    /// Create a new arena with the given buffer
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }
    
    /// Allocate memory from the arena with alignment
    pub fn alloc<T>(&mut self, count: usize) -> Option<&'a mut [T]> {
        let elem_size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        
        // Calculate aligned position
        let align_mask = align - 1;
        let aligned_pos = (self.position + align_mask) & !align_mask;
        
        // Calculate total size needed
        let bytes_needed = count * elem_size;
        let new_pos = aligned_pos + bytes_needed;
        
        // Check if we have enough space
        if new_pos > self.buffer.len() {
            return None;
        }
        
        // Update position
        self.position = new_pos;
        
        // Create slice
        let ptr = unsafe {
            (self.buffer.as_mut_ptr().add(aligned_pos) as *mut T)
                .as_mut()
                .unwrap()
        };
        
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, count) })
    }
    
    /// Reset the arena
    pub fn reset(&mut self) {
        self.position = 0;
    }
}
```

### 5.2 Memory-Mapped File I/O

```rust
/// Save graph to disk using memory mapping for efficient I/O
pub fn save_graph_mmap(graph: &CompressedGraph, path: &str) -> Result<()> {
    // Create file
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    
    // Calculate total size
    let header_size = std::mem::size_of::<u64>() * 3; // node_count, offsets_size, edges_size
    let offsets_size = graph.offsets.len() * std::mem::size_of::<u32>();
    let edges_size = graph.edges.len() * std::mem::size_of::<u32>();
    let total_size = header_size + offsets_size + edges_size;
    
    // Set file size
    file.set_len(total_size as u64)?;
    
    // Memory map the file
    let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };
    
    // Write header
    let mut cursor = std::io::Cursor::new(&mut mmap[..]);
    cursor.write_u64::<LittleEndian>(graph.node_count as u64)?;
    cursor.write_u64::<LittleEndian>(graph.offsets.len() as u64)?;
    cursor.write_u64::<LittleEndian>(graph.edges.len() as u64)?;
    
    // Write offsets
    let offsets_start = header_size;
    let offsets_end = offsets_start + offsets_size;
    let offsets_slice = &mut mmap[offsets_start..offsets_end];
    
    for (i, &offset) in graph.offsets.iter().enumerate() {
        let pos = i * std::mem::size_of::<u32>();
        LittleEndian::write_u32(&mut offsets_slice[pos..pos + 4], offset);
    }
    
    // Write edges
    let edges_start = offsets_end;
    let edges_slice = &mut mmap[edges_start..];
    
    for (i, &edge) in graph.edges.iter().enumerate() {
        let pos = i * std::mem::size_of::<u32>();
        LittleEndian::write_u32(&mut edges_slice[pos..pos + 4], edge);
    }
    
    // Flush changes
    mmap.flush()?;
    
    Ok(())
}
```

### 5.3 Pinned Memory for GPU Transfers

```rust
/// Allocate pinned memory for efficient GPU transfers
pub struct PinnedBuffer<T> {
    /// The data buffer
    pub data: Vec<T>,
    /// Raw pointer for CUDA operations
    ptr: *mut T,
    /// Length of the buffer
    len: usize,
}

impl<T: Copy> PinnedBuffer<T> {
    /// Create a new pinned buffer
    pub fn new(size: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        
        unsafe {
            // Allocate pinned host memory
            rustacuda_sys::cudaHostAlloc(
                &mut ptr as *mut *mut _ as *mut *mut std::ffi::c_void,
                size * std::mem::size_of::<T>(),
                rustacuda_sys::cudaHostAllocDefault
            ).to_result()?;
        }
        
        // Create a Vec that wraps this memory
        let data = unsafe {
            Vec::from_raw_parts(
                ptr as *mut T,
                size,
                size
            )
        };
        
        Ok(Self {
            data,
            ptr,
            len: size,
        })
    }
    
    /// Get a reference to the underlying data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    /// Get a mutable reference to the underlying data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    /// Get a raw pointer for CUDA operations
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    
    /// Get a mutable raw pointer for CUDA operations
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        // Take ownership of the Vec to prevent double-free
        let mut data = std::mem::take(&mut self.data);
        
        // Clear the Vec without running destructors
        data.set_len(0);
        
        // Free the pinned memory
        unsafe {
            rustacuda_sys::cudaFreeHost(self.ptr as *mut std::ffi::c_void);
        }
    }
}
```

## 6. CUDA Kernel Implementations

### 6.1 Mutual Follow Detection Kernel

```cuda
// In cuda/mutual_follows.cu

extern "C" __global__ void count_mutual_follows(
    const unsigned int* offsets,
    const unsigned int* edges,
    unsigned int* mutual_counts,
    unsigned int node_count
) {
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node >= node_count) return;
    
    // Get edges for this node
    const unsigned int start = offsets[node];
    const unsigned int end = offsets[node + 1];
    unsigned int count = 0;
    
    // For each edge (node -> target)
    for (unsigned int i = start; i < end; i++) {
        const unsigned int target = edges[i];
        
        // Skip self-loops
        if (target == node) continue;
        
        // Check if target also follows node (target -> node)
        const unsigned int target_start = offsets[target];
        const unsigned int target_end = offsets[target + 1];
        
        // Binary search for efficiency on large adjacency lists
        bool found = false;
        unsigned int low = target_start;
        unsigned int high = target_end;
        
        while (low < high) {
            const unsigned int mid = low + (high - low) / 2;
            const unsigned int mid_val = edges[mid];
            
            if (mid_val == node) {
                found = true;
                break;
            } else if (mid_val < node) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        if (found) {
            count++;
        }
    }
    
    mutual_counts[node] = count;
}

extern "C" __global__ void populate_mutual_follows(
    const unsigned int* offsets,
    const unsigned int* edges,
    const unsigned int* mutual_offsets,
    unsigned int* mutual_edges,
    unsigned int node_count
) {
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node >= node_count) return;
    
    // Get edges for this node
    const unsigned int start = offsets[node];
    const unsigned int end = offsets[node + 1];
    
    // Get output position
    unsigned int out_pos = mutual_offsets[node];
    
    // For each edge (node -> target)
    for (unsigned int i = start; i < end; i++) {
        const unsigned int target = edges[i];
        
        // Skip self-loops
        if (target == node) continue;
        
        // Check if target also follows node (target -> node)
        const unsigned int target_start = offsets[target];
        const unsigned int target_end = offsets[target + 1];
        
        // Binary search
        bool found = false;
        unsigned int low = target_start;
        unsigned int high = target_end;
        
        while (low < high) {
            const unsigned int mid = low + (high - low) / 2;
            const unsigned int mid_val = edges[mid];
            
            if (mid_val == node) {
                found = true;
                break;
            } else if (mid_val < node) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        if (found) {
            // Only store the edge in one direction to avoid duplicates
            if (node < target) {
                mutual_edges[out_pos++] = target;
            }
        }
    }
}
```

### 6.2 Connected Component Analysis Kernel

```cuda
// In cuda/connected_components.cu

extern "C" __global__ void label_propagation(
    const unsigned int* offsets,
    const unsigned int* edges,
    unsigned int* labels,
    unsigned int* changed,
    unsigned int node_count
) {
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node >= node_count) return;
    
    unsigned int my_label = labels[node];
    unsigned int new_label = my_label;
    
    // Get edges for this node
    const unsigned int start = offsets[node];
    const unsigned int end = offsets[node + 1];
    
    // Find minimum label among neighbors
    for (unsigned int i = start; i < end; i++) {
        const unsigned int neighbor = edges[i];
        const unsigned int neighbor_label = labels[neighbor];
        
        if (neighbor_label < new_label) {
            new_label = neighbor_label;
        }
    }
    
    // Update label if smaller found
    if (new_label != my_label) {
        labels[node] = new_label;
        // Signal that we've made a change
        atomicExch(changed, 1);
    }
}
```

## 7. Build Integration

```rust
// In build.rs

fn main() {
    // Only compile CUDA if feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda();
}

#[cfg(feature = "cuda")]
fn compile_cuda() {
    // Set up CUDA paths - adapt this to your system
    let cuda_path = std::env::var("CUDA_PATH").expect("CUDA_PATH must be set");
    
    // Tell cargo to rebuild if CUDA files change
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/mutual_follows.cu");
    println!("cargo:rerun-if-changed=cuda/connected_components.cu");
    
    // Find NVCC
    let nvcc_path = format!("{}/bin/nvcc", cuda_path);
    
    // GPU architecture for A10 is compute capability 8.6
    let compute_cap = "86";
    
    // Compile CUDA kernels to PTX
    let output = std::process::Command::new(&nvcc_path)
        .args(&[
            "--ptx",
            "-O3",
            &format!("--gpu-architecture=sm_{}", compute_cap),
            "-o",
            &format!("{}/kernels.ptx", std::env::var("OUT_DIR").unwrap()),
            "cuda/kernels.cu",
            "cuda/mutual_follows.cu",
            "cuda/connected_components.cu"
        ])
        .output()
        .expect("Failed to execute NVCC");
    
    if !output.status.success() {
        panic!(
            "Failed to compile CUDA kernels: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
```

## 8. Performance Optimization Specifics

### 8.1 CPU-GPU Coordination

```rust
/// Hybrid CPU-GPU processing pipeline
pub fn process_hybrid(
    graph: &CompressedGraph,
    min_cluster_size: usize,
    gpu_memory_limit: usize,
) -> Result<Vec<Cluster>> {
    let node_count = graph.node_count;
    
    // Decide whether to use GPU based on graph size and available memory
    let edge_count = graph.edges.len();
    let estimated_memory = (node_count + edge_count) * std::mem::size_of::<u32>() * 3;
    
    if estimated_memory <= gpu_memory_limit {
        // Graph fits in GPU memory, use GPU implementation
        log::info!("Using GPU for entire graph processing");
        find_connected_components_gpu(graph, min_cluster_size)
    } else {
        // Use chunked hybrid approach
        log::info!("Using hybrid CPU-GPU processing for large graph");
        
        // Calculate optimal chunk size
        let chunk_size = (gpu_memory_limit / (std::mem::size_of::<u32>() * 3))
            .min(node_count / 2);
            
        log::info!("Processing in chunks of {} nodes", chunk_size);
        
        // Initialize disjoint-set data structure
        let mut disjoint_sets = DisjointSets::new(node_count);
        
        // Process in chunks
        for chunk_start in (0..node_count).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(node_count);
            log::info!("Processing chunk [{}, {})", chunk_start, chunk_end);
            
            // Extract subgraph for this chunk
            let subgraph = extract_subgraph(graph, chunk_start, chunk_end);
            
            // Process subgraph on GPU
            let sub_clusters = find_connected_components_gpu(&subgraph, 1)?;
            
            // Map back to original graph and merge with global labels
            merge_subgraph_labels(&sub_clusters, &mut disjoint_sets, chunk_start);
        }
        
        // Find final connected components
        extract_clusters_from_disjoint_set(&disjoint_sets, min_cluster_size)
    }
}
```

### 8.2 Cache-Friendly Data Structure Organization

```rust
/// Optimize graph for cache locality
pub fn optimize_for_cache(graph: &mut CompressedGraph) -> Result<()> {
    log::info!("Optimizing graph for cache locality");
    
    // Step 1: Reindex nodes for better locality
    let node_count = graph.node_count;
    
    // Build adjacency matrix density to guide reordering
    let mut node_degrees = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let degree = (graph.offsets[i+1] - graph.offsets[i]) as usize;
        node_degrees.push((i, degree));
    }
    
    // Sort nodes by degree for better cache locality
    node_degrees.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Create new indexing
    let mut old_to_new = vec![0; node_count];
    for (new_idx, &(old_idx, _)) in node_degrees.iter().enumerate() {
        old_to_new[old_idx] = new_idx;
    }
    
    // Create a new graph with reordered nodes
    let mut new_graph = CompressedGraph::with_capacity(node_count, graph.edges.len());
    
    // Initialize offsets
    new_graph.offsets.push(0);
    let mut current_offset = 0;
    
    // Build new adjacency lists
    for new_idx in 0..node_count {
        let old_idx = node_degrees[new_idx].0;
        let start = graph.offsets[old_idx] as usize;
        let end = graph.offsets[old_idx + 1] as usize;
        
        // Reindex edges and store in new graph
        let mut new_edges: Vec<(u32, u32)> = Vec::with_capacity(end - start);
        
        for edge_idx in start..end {
            let target_old = graph.edges[edge_idx] as usize;
            let target_new = old_to_new[target_old];
            new_edges.push((target_new as u32, graph.edges[edge_idx]));
        }
        
        // Sort by new index for better cache locality
        new_edges.sort_by_key(|&(idx, _)| idx);
        
        // Add to new graph
        for (_, edge) in new_edges {
            new_graph.edges.push(edge);
            current_offset += 1;
        }
        
        new_graph.offsets.push(current_offset);
    }
    
    // Replace original graph
    *graph = new_graph;
    
    Ok(())
}
```

### 8.3 Custom CUDA Memory Management

```rust
/// Manage GPU memory with a custom pool allocator
pub struct GpuMemoryPool {
    /// Available memory buffers
    buffers: Vec<(usize, DeviceBuffer<u8>)>,
    /// Total allocated memory
    total_allocated: usize,
    /// Maximum allowed allocation
    max_allocation: usize,
}

impl GpuMemoryPool {
    /// Create a new memory pool with the given maximum size
    pub fn new(max_allocation: usize) -> Result<Self> {
        Ok(Self {
            buffers: Vec::new(),
            total_allocated: 0,
            max_allocation,
        })
    }
    
    /// Allocate a buffer of the given size
    pub fn allocate<T>(&mut self, count: usize) -> Result<DeviceBuffer<T>> {
        let size = count * std::mem::size_of::<T>();
        
        // Check if we have a suitable buffer in the pool
        for i in 0..self.buffers.len() {
            let (buffer_size, _) = self.buffers[i];
            
            if buffer_size >= size {
                // Found a suitable buffer
                let (_, buffer) = self.buffers.swap_remove(i);
                
                // Convert to the requested type
                let ptr = buffer.as_device_ptr().as_raw_mut() as *mut T;
                let new_buffer = unsafe {
                    DeviceBuffer::from_raw_parts(
                        ptr,
                        count,
                        count,
                        buffer.context().clone()
                    )
                };
                
                return Ok(new_buffer);
            }
        }
        
        // No suitable buffer found, allocate a new one
        if self.total_allocated + size > self.max_allocation {
            return Err(anyhow!("GPU memory pool exceeded maximum allocation"));
        }
        
        // Allocate new buffer
        let buffer = DeviceBuffer::<T>::zeroed(count)?;
        self.total_allocated += size;
        
        Ok(buffer)
    }
    
    /// Return a buffer to the pool
    pub fn release<T>(&mut self, buffer: DeviceBuffer<T>) {
        let count = buffer.len();
        let size = count * std::mem::size_of::<T>();
        
        // Convert to u8 buffer
        let ptr = buffer.as_device_ptr().as_raw_mut() as *mut u8;
        let context = buffer.context().clone();
        
        // Prevent drop of the original buffer
        std::mem::forget(buffer);
        
        // Create new u8 buffer
        let new_buffer = unsafe {
            DeviceBuffer::from_raw_parts(
                ptr,
                size,
                size,
                context
            )
        };
        
        // Add to pool
        self.buffers.push((size, new_buffer));
    }
}
```

## 9. Command-Line Interface

```rust
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(
    name = "graph-cluster-analyzer",
    about = "High-performance graph cluster analysis of Farcaster data"
)]
struct Cli {
    /// Path to input Parquet file
    #[clap(long)]
    input: String,
    
    /// Output directory for results
    #[clap(long, default_value = "cluster_results")]
    output_dir: String,
    
    /// Minimum number of followings for a user to be included
    #[clap(long, default_value = "50")]
    min_followings: usize,
    
    /// Minimum cluster size
    #[clap(long, default_value = "3")]
    min_cluster_size: usize,
    
    /// Sample ratio (0.0-1.0) for testing with smaller datasets
    #[clap(long, default_value = "1.0")]
    sample: f32,
    
    /// Processing chunk size
    #[clap(long, default_value = "5000000")]
    chunk_size: usize,
    
    /// GPU memory limit in MB
    #[clap(long, default_value = "20000")]
    gpu_memory: usize,
    
    /// Skip visualizations
    #[clap(long)]
    skip_viz: bool,
    
    /// Number of worker threads (0 = use all available cores)
    #[clap(long, default_value = "0")]
    threads: usize,
    
    /// Verbose logging
    #[clap(long, short)]
    verbose: bool,
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Cli::parse();
    
    // Configure logging
    let log_level = if args.verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    
    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp_millis()
        .init();
    
    // Set number of threads
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()?;
    }
    
    log::info!("Starting graph cluster analysis");
    log::info!("Input: {}", args.input);
    log::info!("Output: {}", args.output_dir);
    
    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;
    
    // 1. Load data
    let graph = load_follow_data(
        &args.input,
        args.min_followings,
        args.sample,
        args.chunk_size
    )?;
    
    log::info!("Loaded graph with {} nodes and {} edges", 
              graph.node_count, graph.edges.len());
    
    // 2. Find mutual follows
    let mutual_graph = find_mutual_follows_gpu(&graph, args.chunk_size)?;
    
    log::info!("Found mutual follow graph with {} edges", mutual_graph.edges.len());
    
    // 3. Find clusters
    let clusters = find_connected_components_gpu(
        &mutual_graph, 
        args.min_cluster_size
    )?;
    
    log::info!("Found {} clusters", clusters.len());
    
    // 4. Save results
    save_results(&clusters, &graph, &mutual_graph, &args.output_dir)?;
    
    // 5. Generate visualizations if requested
    if !args.skip_viz {
        generate_visualizations(&clusters, &mutual_graph, &args.output_dir)?;
    }
    
    log::info!("Analysis complete. Results saved to {}", args.output_dir);
    
    Ok(())
}
```

This technical implementation specification provides a comprehensive roadmap for developing a high-performance Rust-based graph cluster analysis system for your Farcaster dataset, optimized for your A10 GPU with 200GB RAM and 24GB VRAM. The implementation focuses on memory efficiency, CUDA acceleration, and parallel processing to handle the 152M+ row dataset.