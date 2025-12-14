#include <mpi.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sched.h>

using namespace std;

#define PI 3.14159265358979323846

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

#if __CUDA_ARCH__ >= 350
#define LDG(x) __ldg(&(x))
#else
#define LDG(x) (x)
#endif

void set_cpu_affinity(int rank) {
    int smt_stride = 8; 
    int num_logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_logical_cpus < 1) return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int physical_core_id = rank % (num_logical_cpus / smt_stride);
    int target_cpu = physical_core_id * smt_stride;
    CPU_SET(target_cpu, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

__device__ __host__ inline int get_idx(int i, int j, int k, int nyg, int nzg) {
    return (i * nyg + j) * nzg + k;
}

__global__ void precompute_trig_kernel(double* __restrict__ sx, double* __restrict__ sy, double* __restrict__ sz, 
                                       int nx_loc, int ny_loc, int nz_loc,
                                       int i_start, int j_start, int k_start,
                                       double hx, double hy, double hz,
                                       double Lx, double Ly, double Lz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (idx <= nx_loc) {
        double x = (i_start + idx - 1) * hx;
        sx[idx] = sin(3.0 * PI * x / Lx);
    }
    if (idx <= ny_loc) {
        double y = (j_start + idx - 1) * hy;
        sy[idx] = sin(2.0 * PI * y / Ly);
    }
    if (idx <= nz_loc) {
        double z = (k_start + idx - 1) * hz;
        sz[idx] = sin(2.0 * PI * z / Lz);
    }
}

__global__ void init_kernel(double* __restrict__ u, 
                            const double* __restrict__ sx, 
                            const double* __restrict__ sy, 
                            const double* __restrict__ sz,
                            int nx_loc, int ny_loc, int nz_loc,
                            int nyg, int nzg) {
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= nx_loc && j <= ny_loc && k <= nz_loc) {
        u[get_idx(i, j, k, nyg, nzg)] = LDG(sx[i]) * LDG(sy[j]) * LDG(sz[k]); 
    }
}

__global__ void step1_kernel(double* __restrict__ u_curr, 
                             const double* __restrict__ u_prev, 
                             int nx_loc, int ny_loc, int nz_loc,
                             double a2, double tau, double hx, double hy, double hz,
                             int nyg, int nzg) {
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= nx_loc && j <= ny_loc && k <= nz_loc) {
        int idx = get_idx(i, j, k, nyg, nzg);
        double center = LDG(u_prev[idx]);
        double lap = (LDG(u_prev[get_idx(i-1, j, k, nyg, nzg)]) - 2.0*center + LDG(u_prev[get_idx(i+1, j, k, nyg, nzg)])) / (hx*hx) +
                     (LDG(u_prev[get_idx(i, j-1, k, nyg, nzg)]) - 2.0*center + LDG(u_prev[get_idx(i, j+1, k, nyg, nzg)])) / (hy*hy) +
                     (LDG(u_prev[get_idx(i, j, k-1, nyg, nzg)]) - 2.0*center + LDG(u_prev[get_idx(i, j, k+1, nyg, nzg)])) / (hz*hz);
        u_curr[idx] = center + 0.5 * a2 * tau * tau * lap;
    }
}

__global__ void step_kernel(double* __restrict__ u_next, 
                            const double* __restrict__ u_curr, 
                            const double* __restrict__ u_prev,
                            int nx_loc, int ny_loc, int nz_loc,
                            double a2, double tau, double hx, double hy, double hz,
                            int nyg, int nzg) {
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= nx_loc && j <= ny_loc && k <= nz_loc) {
        int idx = get_idx(i, j, k, nyg, nzg);
        double center = LDG(u_curr[idx]);
        double lap = (LDG(u_curr[get_idx(i-1, j, k, nyg, nzg)]) - 2.0*center + LDG(u_curr[get_idx(i+1, j, k, nyg, nzg)])) / (hx*hx) +
                     (LDG(u_curr[get_idx(i, j-1, k, nyg, nzg)]) - 2.0*center + LDG(u_curr[get_idx(i, j+1, k, nyg, nzg)])) / (hy*hy) +
                     (LDG(u_curr[get_idx(i, j, k-1, nyg, nzg)]) - 2.0*center + LDG(u_curr[get_idx(i, j, k+1, nyg, nzg)])) / (hz*hz);
        u_next[idx] = 2.0*center - LDG(u_prev[idx]) + a2 * tau * tau * lap;
    }
}

__global__ void boundary_kernel(double* __restrict__ u, int nx_loc, int ny_loc, int nz_loc, 
                                int i_start, int nx_global, int nyg, int nzg) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (j < nyg && k < nzg) {
        if (i_start == 0) {
            u[get_idx(0, j, k, nyg, nzg)] = 0.0;
            u[get_idx(1, j, k, nyg, nzg)] = 0.0;
        }
        if (i_start + nx_loc == nx_global) {
            u[get_idx(nx_loc, j, k, nyg, nzg)] = 0.0;
            u[get_idx(nx_loc+1, j, k, nyg, nzg)] = 0.0;
        }
    }
}

struct ErrorFunctor {
    const double *u, *sx, *sy, *sz;
    int nx_loc, ny_loc, nz_loc, nyg, nzg;
    double t_factor;

    ErrorFunctor(const double* _u, const double* _sx, const double* _sy, const double* _sz,
                 int _nx, int _ny, int _nz, int _nyg, int _nzg, double _t_factor)
        : u(_u), sx(_sx), sy(_sy), sz(_sz), 
          nx_loc(_nx), ny_loc(_ny), nz_loc(_nz), nyg(_nyg), nzg(_nzg), t_factor(_t_factor) {}

    __host__ __device__
    double operator()(const int& idx_linear) const {
        int k_local = idx_linear % nz_loc + 1;
        int rem = idx_linear / nz_loc;
        int j_local = rem % ny_loc + 1;
        int i_local = rem / ny_loc + 1;

        double spatial = sx[i_local] * sy[j_local] * sz[k_local];
        double ua = spatial * t_factor;

        int grid_idx = (i_local * nyg + j_local) * nzg + k_local;
        double val = u[grid_idx];
        return fabs(val - ua);
    }
};

__global__ void pack_kernel(const double* __restrict__ u, double* __restrict__ buf, int face_dim1, int face_dim2, int fix_dim_idx, int dim_code, int nyg, int nzg) {
    int d2 = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int d1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (d1 <= face_dim1 && d2 <= face_dim2) {
        int idx_dst = (d1-1) * face_dim2 + (d2-1);
        int idx_src = 0;
        
        if (dim_code == 0) idx_src = get_idx(fix_dim_idx, d1, d2, nyg, nzg);
        else if (dim_code == 1) idx_src = get_idx(d1, fix_dim_idx, d2, nyg, nzg);
        else idx_src = get_idx(d1, d2, fix_dim_idx, nyg, nzg);
        
        buf[idx_dst] = u[idx_src];
    }
}

__global__ void unpack_kernel(double* __restrict__ u, const double* __restrict__ buf, int face_dim1, int face_dim2, int fix_dim_idx, int dim_code, int nyg, int nzg) {
    int d2 = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int d1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (d1 <= face_dim1 && d2 <= face_dim2) {
        int idx_src = (d1-1) * face_dim2 + (d2-1);
        int idx_dst = 0;
        if (dim_code == 0) idx_dst = get_idx(fix_dim_idx, d1, d2, nyg, nzg);
        else if (dim_code == 1) idx_dst = get_idx(d1, fix_dim_idx, d2, nyg, nzg);
        else idx_dst = get_idx(d1, d2, fix_dim_idx, nyg, nzg);
        u[idx_dst] = buf[idx_src];
    }
}

struct HaloCtx {
    int nx_loc, ny_loc, nz_loc, nyg, nzg;
    int left, right, down, up, back, front;
    double *d_sb_x, *d_rb_x, *h_sb_x, *h_rb_x;
    double *d_sb_y, *d_rb_y, *h_sb_y, *h_rb_y;
    double *d_sb_z, *d_rb_z, *h_sb_z, *h_rb_z;
    int sz_x, sz_y, sz_z;
    MPI_Comm comm;
    dim3 grid_x, grid_y, grid_z, block2d;
};

void run_halo_exchange(double* d_arr, HaloCtx& ctx, 
                       float& time_calc, float& time_copy, float& time_comm,
                       cudaEvent_t& start, cudaEvent_t& stop) {
    float t_k, t_c, t_m;
    
    cudaDeviceSynchronize();

    cudaEventRecord(start); pack_kernel<<<ctx.grid_x, ctx.block2d>>>(d_arr, ctx.d_sb_x, ctx.ny_loc, ctx.nz_loc, ctx.nx_loc, 0, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_x, ctx.d_sb_x, ctx.sz_x, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); MPI_Sendrecv(ctx.h_sb_x, ctx.ny_loc*ctx.nz_loc, MPI_DOUBLE, ctx.right, 0, ctx.h_rb_x, ctx.ny_loc*ctx.nz_loc, MPI_DOUBLE, ctx.left, 0, ctx.comm, MPI_STATUS_IGNORE);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;
    if (ctx.left != MPI_PROC_NULL) {
        cudaEventRecord(start); cudaMemcpy(ctx.d_rb_x, ctx.h_rb_x, ctx.sz_x, cudaMemcpyHostToDevice); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
        cudaEventRecord(start); unpack_kernel<<<ctx.grid_x, ctx.block2d>>>(d_arr, ctx.d_rb_x, ctx.ny_loc, ctx.nz_loc, 0, 0, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    }

    cudaEventRecord(start); pack_kernel<<<ctx.grid_x, ctx.block2d>>>(d_arr, ctx.d_sb_x, ctx.ny_loc, ctx.nz_loc, 1, 0, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_x, ctx.d_sb_x, ctx.sz_x, cudaMemcpyDeviceToHost); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); MPI_Sendrecv(ctx.h_sb_x, ctx.ny_loc*ctx.nz_loc, MPI_DOUBLE, ctx.left, 1, ctx.h_rb_x, ctx.ny_loc*ctx.nz_loc, MPI_DOUBLE, ctx.right, 1, ctx.comm, MPI_STATUS_IGNORE); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;
    if (ctx.right != MPI_PROC_NULL) {
        cudaEventRecord(start); cudaMemcpy(ctx.d_rb_x, ctx.h_rb_x, ctx.sz_x, cudaMemcpyHostToDevice); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
        cudaEventRecord(start); unpack_kernel<<<ctx.grid_x, ctx.block2d>>>(d_arr, ctx.d_rb_x, ctx.ny_loc, ctx.nz_loc, ctx.nx_loc+1, 0, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    }

    cudaEventRecord(start); pack_kernel<<<ctx.grid_y, ctx.block2d>>>(d_arr, ctx.d_sb_y, ctx.nx_loc, ctx.nz_loc, ctx.ny_loc, 1, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_y, ctx.d_sb_y, ctx.sz_y, cudaMemcpyDeviceToHost); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); MPI_Sendrecv(ctx.h_sb_y, ctx.nx_loc*ctx.nz_loc, MPI_DOUBLE, ctx.up, 2, ctx.h_rb_y, ctx.nx_loc*ctx.nz_loc, MPI_DOUBLE, ctx.down, 2, ctx.comm, MPI_STATUS_IGNORE); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;
    cudaEventRecord(start); cudaMemcpy(ctx.d_rb_y, ctx.h_rb_y, ctx.sz_y, cudaMemcpyHostToDevice); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); unpack_kernel<<<ctx.grid_y, ctx.block2d>>>(d_arr, ctx.d_rb_y, ctx.nx_loc, ctx.nz_loc, 0, 1, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); pack_kernel<<<ctx.grid_y, ctx.block2d>>>(d_arr, ctx.d_sb_y, ctx.nx_loc, ctx.nz_loc, 1, 1, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_y, ctx.d_sb_y, ctx.sz_y, cudaMemcpyDeviceToHost); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); MPI_Sendrecv(ctx.h_sb_y, ctx.nx_loc*ctx.nz_loc, MPI_DOUBLE, ctx.down, 3, ctx.h_rb_y, ctx.nx_loc*ctx.nz_loc, MPI_DOUBLE, ctx.up, 3, ctx.comm, MPI_STATUS_IGNORE); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;
    cudaEventRecord(start); cudaMemcpy(ctx.d_rb_y, ctx.h_rb_y, ctx.sz_y, cudaMemcpyHostToDevice); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); unpack_kernel<<<ctx.grid_y, ctx.block2d>>>(d_arr, ctx.d_rb_y, ctx.nx_loc, ctx.nz_loc, ctx.ny_loc+1, 1, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); pack_kernel<<<ctx.grid_z, ctx.block2d>>>(d_arr, ctx.d_sb_z, ctx.nx_loc, ctx.ny_loc, ctx.nz_loc, 2, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_z, ctx.d_sb_z, ctx.sz_z, cudaMemcpyDeviceToHost); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); MPI_Sendrecv(ctx.h_sb_z, ctx.nx_loc*ctx.ny_loc, MPI_DOUBLE, ctx.front, 4, ctx.h_rb_z, ctx.nx_loc*ctx.ny_loc, MPI_DOUBLE, ctx.back, 4, ctx.comm, MPI_STATUS_IGNORE); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;
    cudaEventRecord(start); cudaMemcpy(ctx.d_rb_z, ctx.h_rb_z, ctx.sz_z, cudaMemcpyHostToDevice); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); unpack_kernel<<<ctx.grid_z, ctx.block2d>>>(d_arr, ctx.d_rb_z, ctx.nx_loc, ctx.ny_loc, 0, 2, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); pack_kernel<<<ctx.grid_z, ctx.block2d>>>(d_arr, ctx.d_sb_z, ctx.nx_loc, ctx.ny_loc, 1, 2, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_z, ctx.d_sb_z, ctx.sz_z, cudaMemcpyDeviceToHost); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); MPI_Sendrecv(ctx.h_sb_z, ctx.nx_loc*ctx.ny_loc, MPI_DOUBLE, ctx.back, 5, ctx.h_rb_z, ctx.nx_loc*ctx.ny_loc, MPI_DOUBLE, ctx.front, 5, ctx.comm, MPI_STATUS_IGNORE); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;
    cudaEventRecord(start); cudaMemcpy(ctx.d_rb_z, ctx.h_rb_z, ctx.sz_z, cudaMemcpyHostToDevice); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
    cudaEventRecord(start); unpack_kernel<<<ctx.grid_z, ctx.block2d>>>(d_arr, ctx.d_rb_z, ctx.nx_loc, ctx.ny_loc, ctx.nz_loc+1, 2, ctx.nyg, ctx.nzg); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    set_cpu_affinity(rank);

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    
    if (num_devices == 0) {
        if (rank == 0) cerr << "Error: No CUDA devices found." << endl;
        MPI_Finalize();
        return 1;
    }
    cudaSetDevice(rank % num_devices);

    size_t N = 512; 
    size_t Nx_global = N + 1;
    size_t Ny_global = N;
    size_t Nz_global = N;

    double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    double hx = Lx / N, hy = Ly / N, hz = Lz / N;
    double a_t = 2.0 * PI;
    double a2 = 4.0 / ((9.0/(Lx*Lx)) + (4.0/(Ly*Ly)) + (4.0/(Lz*Lz)));
    double tau = 0.8 / (sqrt(a2) * sqrt(1.0/(hx*hx) + 1.0/(hy*hy) + 1.0/(hz*hz))); 
    size_t Nt = 50;

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(nprocs, 3, dims);
    int periods[3] = {0, 1, 1}; 
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    if (cart_comm == MPI_COMM_NULL) { MPI_Finalize(); return 0; }

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    auto get_dist = [&](int global_n, int dim_idx) {
        int q = global_n / dims[dim_idx];
        int r = global_n % dims[dim_idx];
        int s = coords[dim_idx] < r ? q + 1 : q;
        int start = coords[dim_idx] < r ? (q + 1) * coords[dim_idx] : (q + 1) * r + q * (coords[dim_idx] - r);
        return std::make_pair(s, start);
    };

    auto px = get_dist(Nx_global, 0);
    auto py = get_dist(Ny_global, 1);
    auto pz = get_dist(Nz_global, 2);

    int nx_loc = px.first, i_start = px.second;
    int ny_loc = py.first, j_start = py.second;
    int nz_loc = pz.first, k_start = pz.second;

    int nyg = ny_loc + 2; 
    int nzg = nz_loc + 2; 
    size_t total_elements = (size_t)(nx_loc + 2) * nyg * nzg;

    if (rank == 0) {
        cout << fixed << setprecision(8);
        cout << "MPI+CUDA+Thrust Implementation. Grid: " << N << "^3. Procs: " << nprocs << endl;
    }

    double *d_u_prev, *d_u_curr, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, total_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_curr, total_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_next, total_elements * sizeof(double)));
    
    double *d_sx, *d_sy, *d_sz;
    CUDA_CHECK(cudaMalloc(&d_sx, (nx_loc + 2) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sy, (ny_loc + 2) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sz, (nz_loc + 2) * sizeof(double)));

    HaloCtx ctx;
    ctx.nx_loc = nx_loc; ctx.ny_loc = ny_loc; ctx.nz_loc = nz_loc;
    ctx.nyg = nyg; ctx.nzg = nzg;
    ctx.comm = cart_comm;
    ctx.sz_x = ny_loc * nz_loc * sizeof(double);
    ctx.sz_y = nx_loc * nz_loc * sizeof(double);
    ctx.sz_z = nx_loc * ny_loc * sizeof(double);
    CUDA_CHECK(cudaMalloc(&ctx.d_sb_x, ctx.sz_x)); CUDA_CHECK(cudaMalloc(&ctx.d_rb_x, ctx.sz_x));
    CUDA_CHECK(cudaMalloc(&ctx.d_sb_y, ctx.sz_y)); CUDA_CHECK(cudaMalloc(&ctx.d_rb_y, ctx.sz_y));
    CUDA_CHECK(cudaMalloc(&ctx.d_sb_z, ctx.sz_z)); CUDA_CHECK(cudaMalloc(&ctx.d_rb_z, ctx.sz_z));
    CUDA_CHECK(cudaMallocHost(&ctx.h_sb_x, ctx.sz_x)); CUDA_CHECK(cudaMallocHost(&ctx.h_rb_x, ctx.sz_x));
    CUDA_CHECK(cudaMallocHost(&ctx.h_sb_y, ctx.sz_y)); CUDA_CHECK(cudaMallocHost(&ctx.h_rb_y, ctx.sz_y));
    CUDA_CHECK(cudaMallocHost(&ctx.h_sb_z, ctx.sz_z)); CUDA_CHECK(cudaMallocHost(&ctx.h_rb_z, ctx.sz_z));
    
    MPI_Cart_shift(cart_comm, 0, 1, &ctx.left, &ctx.right);
    MPI_Cart_shift(cart_comm, 1, 1, &ctx.down, &ctx.up);   
    MPI_Cart_shift(cart_comm, 2, 1, &ctx.back, &ctx.front); 

    dim3 block(8, 8, 8);
    dim3 grid((nz_loc + 7)/8, (ny_loc + 7)/8, (nx_loc + 7)/8);
    ctx.block2d = dim3(16, 16);
    ctx.grid_x = dim3((nz_loc+15)/16, (ny_loc+15)/16);
    ctx.grid_y = dim3((nz_loc+15)/16, (nx_loc+15)/16);
    ctx.grid_z = dim3((ny_loc+15)/16, (nx_loc+15)/16);
    dim3 bound_grid((nzg+15)/16, (nyg+15)/16);
    dim3 grid_1d((max(nx_loc, max(ny_loc, nz_loc)) + 255) / 256);

    float time_calc = 0.0f, time_copy = 0.0f, time_comm = 0.0f, time_init = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    precompute_trig_kernel<<<grid_1d, 256>>>(d_sx, d_sy, d_sz, nx_loc, ny_loc, nz_loc, i_start, j_start, k_start, hx, hy, hz, Lx, Ly, Lz);
    init_kernel<<<grid, block>>>(d_u_prev, d_sx, d_sy, d_sz, nx_loc, ny_loc, nz_loc, nyg, nzg);
    boundary_kernel<<<bound_grid, ctx.block2d>>>(d_u_prev, nx_loc, ny_loc, nz_loc, i_start, Nx_global, nyg, nzg);
    
    run_halo_exchange(d_u_prev, ctx, time_calc, time_copy, time_comm, start, stop);
    
    step1_kernel<<<grid, block>>>(d_u_curr, d_u_prev, nx_loc, ny_loc, nz_loc, a2, tau, hx, hy, hz, nyg, nzg);
    boundary_kernel<<<bound_grid, ctx.block2d>>>(d_u_curr, nx_loc, ny_loc, nz_loc, i_start, Nx_global, nyg, nzg);

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_init, start, stop);

    double global_max_err = 0.0;
    int calc_size = nx_loc * ny_loc * nz_loc;

    for (size_t n = 2; n <= Nt; ++n) {
        double t_physical = n * tau;
        
        run_halo_exchange(d_u_curr, ctx, time_calc, time_copy, time_comm, start, stop);

        cudaEventRecord(start);
        step_kernel<<<grid, block>>>(d_u_next, d_u_curr, d_u_prev, nx_loc, ny_loc, nz_loc, a2, tau, hx, hy, hz, nyg, nzg);
        boundary_kernel<<<bound_grid, ctx.block2d>>>(d_u_next, nx_loc, ny_loc, nz_loc, i_start, Nx_global, nyg, nzg);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float t_step; cudaEventElapsedTime(&t_step, start, stop); time_calc += t_step;

        cudaEventRecord(start);
        double t_factor = cos(a_t * t_physical + 4.0 * PI);
        ErrorFunctor func(d_u_next, d_sx, d_sy, d_sz, nx_loc, ny_loc, nz_loc, nyg, nzg, t_factor);
        double local_max_err = thrust::transform_reduce(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(calc_size), func, 0.0, thrust::maximum<double>());
        
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_step, start, stop); time_calc += t_step;

        cudaEventRecord(start);
        double proc_err = 0.0;
        MPI_Reduce(&local_max_err, &proc_err, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float t_red; cudaEventElapsedTime(&t_red, start, stop); time_comm += t_red;

        if (rank == 0) {
            if (proc_err > global_max_err) global_max_err = proc_err;
            cout << "Step " << n << " (t=" << t_physical << "): Error = " << proc_err << endl;
        }

        double* temp = d_u_prev; d_u_prev = d_u_curr; d_u_curr = d_u_next; d_u_next = temp;
    }

    MPI_Barrier(cart_comm);
    if (rank == 0) {
        cout << endl << "--- RESULTS ---" << endl;
        cout << "Final Max Error: " << global_max_err << endl;
        cout << endl << "--- TIMING (Seconds) ---" << endl;
        cout << "Initialization:                         " << time_init / 1000.0 << endl;
        cout << "Parallel Cycles (Kernels+Halo+Thrust):  " << time_calc / 1000.0 << endl;
        cout << "Data Copy (Host <-> Device):            " << time_copy / 1000.0 << endl;
        cout << "Communication (MPI):                    " << time_comm / 1000.0 << endl;
        cout << "Total Halo Overhead:                    " << (time_copy + time_comm) / 1000.0 << endl;
        cout << "Total Time:                             " << (time_init + time_calc + time_copy + time_comm) / 1000.0 << endl;
    }

    cudaFree(d_u_prev); cudaFree(d_u_curr); cudaFree(d_u_next);
    cudaFree(d_sx); cudaFree(d_sy); cudaFree(d_sz); 
    cudaFree(ctx.d_sb_x); cudaFree(ctx.d_rb_x); cudaFree(ctx.d_sb_y); cudaFree(ctx.d_rb_y); cudaFree(ctx.d_sb_z); cudaFree(ctx.d_rb_z);
    
    cudaFreeHost(ctx.h_sb_x); cudaFreeHost(ctx.h_rb_x); cudaFreeHost(ctx.h_sb_y); 
    cudaFreeHost(ctx.h_rb_y); cudaFreeHost(ctx.h_sb_z); cudaFreeHost(ctx.h_rb_z);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}