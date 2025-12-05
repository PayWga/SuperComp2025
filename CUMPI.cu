#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <sched.h>
#include <unistd.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

#define PI 3.14159265358979323846

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

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double max_val = fmax(val, old_val);
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(max_val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ inline int get_idx(int i, int j, int k, int nyg, int nzg) {
    return (i * nyg + j) * nzg + k;
}

__device__ double u_analytic_dev(double x, double y, double z, double t, 
                                 double Lx, double Ly, double Lz, double a_t) {
    return sin(3.0 * PI * x / Lx) * sin(2.0 * PI * y / Ly) * sin(2.0 * PI * z / Lz) * cos(a_t * t + 4.0 * PI);
}

__global__ void init_kernel(double* u, int nx_loc, int ny_loc, int nz_loc,
                            int i_start, int j_start, int k_start,
                            double hx, double hy, double hz,
                            double Lx, double Ly, double Lz, double a_t,
                            int nyg, int nzg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= nx_loc && j <= ny_loc && k <= nz_loc) {
        double x = (i_start + i - 1) * hx;
        double y = (j_start + j - 1) * hy;
        double z = (k_start + k - 1) * hz;
        u[get_idx(i, j, k, nyg, nzg)] = u_analytic_dev(x, y, z, 0.0, Lx, Ly, Lz, a_t);
    }
}

__global__ void step1_kernel(double* u_curr, const double* u_prev, 
                             int nx_loc, int ny_loc, int nz_loc,
                             double a2, double tau, double hx, double hy, double hz,
                             int nyg, int nzg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= nx_loc && j <= ny_loc && k <= nz_loc) {
        int idx = get_idx(i, j, k, nyg, nzg);
        double center = u_prev[idx];
        double lap = (u_prev[get_idx(i-1, j, k, nyg, nzg)] - 2.0*center + u_prev[get_idx(i+1, j, k, nyg, nzg)]) / (hx*hx) +
                     (u_prev[get_idx(i, j-1, k, nyg, nzg)] - 2.0*center + u_prev[get_idx(i, j+1, k, nyg, nzg)]) / (hy*hy) +
                     (u_prev[get_idx(i, j, k-1, nyg, nzg)] - 2.0*center + u_prev[get_idx(i, j, k+1, nyg, nzg)]) / (hz*hz);
        u_curr[idx] = center + 0.5 * a2 * tau * tau * lap;
    }
}

__global__ void step_kernel(double* u_next, const double* u_curr, const double* u_prev,
                            int nx_loc, int ny_loc, int nz_loc,
                            double a2, double tau, double hx, double hy, double hz,
                            int nyg, int nzg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= nx_loc && j <= ny_loc && k <= nz_loc) {
        int idx = get_idx(i, j, k, nyg, nzg);
        double center = u_curr[idx];
        double lap = (u_curr[get_idx(i-1, j, k, nyg, nzg)] - 2.0*center + u_curr[get_idx(i+1, j, k, nyg, nzg)]) / (hx*hx) +
                     (u_curr[get_idx(i, j-1, k, nyg, nzg)] - 2.0*center + u_curr[get_idx(i, j+1, k, nyg, nzg)]) / (hy*hy) +
                     (u_curr[get_idx(i, j, k-1, nyg, nzg)] - 2.0*center + u_curr[get_idx(i, j, k+1, nyg, nzg)]) / (hz*hz);
        u_next[idx] = 2.0*center - u_prev[idx] + a2 * tau * tau * lap;
    }
}

__global__ void boundary_kernel(double* u, int nx_loc, int ny_loc, int nz_loc, 
                                int i_start, int nx_global, int nyg, int nzg) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int k = blockIdx.y * blockDim.y + threadIdx.y; 

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

__global__ void error_kernel(const double* u, double* max_err_dev, double t,
                             int nx_loc, int ny_loc, int nz_loc,
                             int i_start, int j_start, int k_start,
                             double hx, double hy, double hz,
                             double Lx, double Ly, double Lz, double a_t,
                             int nyg, int nzg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= nx_loc && j <= ny_loc && k <= nz_loc) {
        double x = (i_start + i - 1) * hx;
        double y = (j_start + j - 1) * hy;
        double z = (k_start + k - 1) * hz;
        double ua = u_analytic_dev(x, y, z, t, Lx, Ly, Lz, a_t);
        double val = u[get_idx(i, j, k, nyg, nzg)];
        atomicMaxDouble(max_err_dev, fabs(val - ua));
    }
}

__global__ void pack_kernel(const double* u, double* buf, 
                            int face_dim1, int face_dim2,
                            int fix_dim_idx, int dim_code, int nyg, int nzg) {
    int d1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int d2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (d1 <= face_dim1 && d2 <= face_dim2) {
        int idx_dst = (d1-1) * face_dim2 + (d2-1);
        int idx_src = 0;
        if (dim_code == 0) idx_src = get_idx(fix_dim_idx, d1, d2, nyg, nzg);
        else if (dim_code == 1) idx_src = get_idx(d1, fix_dim_idx, d2, nyg, nzg);
        else idx_src = get_idx(d1, d2, fix_dim_idx, nyg, nzg);
        buf[idx_dst] = u[idx_src];
    }
}

__global__ void unpack_kernel(double* u, const double* buf, 
                              int face_dim1, int face_dim2,
                              int fix_dim_idx, int dim_code, int nyg, int nzg) {
    int d1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int d2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
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

    cudaEventRecord(start);
    pack_kernel<<<ctx.grid_x, ctx.block2d>>>(d_arr, ctx.d_sb_x, ctx.ny_loc, ctx.nz_loc, ctx.nx_loc, 0, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_x, ctx.d_sb_x, ctx.sz_x, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start);
    MPI_Sendrecv(ctx.h_sb_x, ctx.ny_loc*ctx.nz_loc, MPI_DOUBLE, ctx.right, 0,
                 ctx.h_rb_x, ctx.ny_loc*ctx.nz_loc, MPI_DOUBLE, ctx.left, 0, ctx.comm, MPI_STATUS_IGNORE);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;

    if (ctx.left != MPI_PROC_NULL) {
        cudaEventRecord(start); cudaMemcpy(ctx.d_rb_x, ctx.h_rb_x, ctx.sz_x, cudaMemcpyHostToDevice);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;
        
        cudaEventRecord(start);
        unpack_kernel<<<ctx.grid_x, ctx.block2d>>>(d_arr, ctx.d_rb_x, ctx.ny_loc, ctx.nz_loc, 0, 0, ctx.nyg, ctx.nzg);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    }

    cudaEventRecord(start);
    pack_kernel<<<ctx.grid_x, ctx.block2d>>>(d_arr, ctx.d_sb_x, ctx.ny_loc, ctx.nz_loc, 1, 0, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_x, ctx.d_sb_x, ctx.sz_x, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start);
    MPI_Sendrecv(ctx.h_sb_x, ctx.ny_loc*ctx.nz_loc, MPI_DOUBLE, ctx.left, 1,
                 ctx.h_rb_x, ctx.ny_loc*ctx.nz_loc, MPI_DOUBLE, ctx.right, 1, ctx.comm, MPI_STATUS_IGNORE);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;

    if (ctx.right != MPI_PROC_NULL) {
        cudaEventRecord(start); cudaMemcpy(ctx.d_rb_x, ctx.h_rb_x, ctx.sz_x, cudaMemcpyHostToDevice);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

        cudaEventRecord(start);
        unpack_kernel<<<ctx.grid_x, ctx.block2d>>>(d_arr, ctx.d_rb_x, ctx.ny_loc, ctx.nz_loc, ctx.nx_loc+1, 0, ctx.nyg, ctx.nzg);
        cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
    }

    cudaEventRecord(start); pack_kernel<<<ctx.grid_y, ctx.block2d>>>(d_arr, ctx.d_sb_y, ctx.nx_loc, ctx.nz_loc, ctx.ny_loc, 1, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_y, ctx.d_sb_y, ctx.sz_y, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start);
    MPI_Sendrecv(ctx.h_sb_y, ctx.nx_loc*ctx.nz_loc, MPI_DOUBLE, ctx.up, 2, 
                 ctx.h_rb_y, ctx.nx_loc*ctx.nz_loc, MPI_DOUBLE, ctx.down, 2, ctx.comm, MPI_STATUS_IGNORE);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;

    cudaEventRecord(start); cudaMemcpy(ctx.d_rb_y, ctx.h_rb_y, ctx.sz_y, cudaMemcpyHostToDevice);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start); unpack_kernel<<<ctx.grid_y, ctx.block2d>>>(d_arr, ctx.d_rb_y, ctx.nx_loc, ctx.nz_loc, 0, 1, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); pack_kernel<<<ctx.grid_y, ctx.block2d>>>(d_arr, ctx.d_sb_y, ctx.nx_loc, ctx.nz_loc, 1, 1, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_y, ctx.d_sb_y, ctx.sz_y, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start);
    MPI_Sendrecv(ctx.h_sb_y, ctx.nx_loc*ctx.nz_loc, MPI_DOUBLE, ctx.down, 3, 
                 ctx.h_rb_y, ctx.nx_loc*ctx.nz_loc, MPI_DOUBLE, ctx.up, 3, ctx.comm, MPI_STATUS_IGNORE);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;

    cudaEventRecord(start); cudaMemcpy(ctx.d_rb_y, ctx.h_rb_y, ctx.sz_y, cudaMemcpyHostToDevice);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start); unpack_kernel<<<ctx.grid_y, ctx.block2d>>>(d_arr, ctx.d_rb_y, ctx.nx_loc, ctx.nz_loc, ctx.ny_loc+1, 1, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); pack_kernel<<<ctx.grid_z, ctx.block2d>>>(d_arr, ctx.d_sb_z, ctx.nx_loc, ctx.ny_loc, ctx.nz_loc, 2, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_z, ctx.d_sb_z, ctx.sz_z, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start);
    MPI_Sendrecv(ctx.h_sb_z, ctx.nx_loc*ctx.ny_loc, MPI_DOUBLE, ctx.front, 4, 
                 ctx.h_rb_z, ctx.nx_loc*ctx.ny_loc, MPI_DOUBLE, ctx.back, 4, ctx.comm, MPI_STATUS_IGNORE);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;

    cudaEventRecord(start); cudaMemcpy(ctx.d_rb_z, ctx.h_rb_z, ctx.sz_z, cudaMemcpyHostToDevice);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start); unpack_kernel<<<ctx.grid_z, ctx.block2d>>>(d_arr, ctx.d_rb_z, ctx.nx_loc, ctx.ny_loc, 0, 2, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); pack_kernel<<<ctx.grid_z, ctx.block2d>>>(d_arr, ctx.d_sb_z, ctx.nx_loc, ctx.ny_loc, 1, 2, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;

    cudaEventRecord(start); cudaMemcpy(ctx.h_sb_z, ctx.d_sb_z, ctx.sz_z, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start);
    MPI_Sendrecv(ctx.h_sb_z, ctx.nx_loc*ctx.ny_loc, MPI_DOUBLE, ctx.back, 5, 
                 ctx.h_rb_z, ctx.nx_loc*ctx.ny_loc, MPI_DOUBLE, ctx.front, 5, ctx.comm, MPI_STATUS_IGNORE);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_m, start, stop); time_comm += t_m;

    cudaEventRecord(start); cudaMemcpy(ctx.d_rb_z, ctx.h_rb_z, ctx.sz_z, cudaMemcpyHostToDevice);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_c, start, stop); time_copy += t_c;

    cudaEventRecord(start); unpack_kernel<<<ctx.grid_z, ctx.block2d>>>(d_arr, ctx.d_rb_z, ctx.nx_loc, ctx.ny_loc, ctx.nz_loc+1, 2, ctx.nyg, ctx.nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&t_k, start, stop); time_calc += t_k;
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
        if (rank == 0) cerr << "Error: No GPUs detected. Check LSF script includes -gpu flag." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaSetDevice(rank % num_devices);

    double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    size_t N = 256; 
    size_t Nx_global = N + 1;
    size_t Ny_global = N;
    size_t Nz_global = N;

    double hx = Lx / N, hy = Ly / N, hz = Lz / N;
    double a_t = 2.0 * PI;
    double denom = (9.0/(Lx*Lx)) + (4.0/(Ly*Ly)) + (4.0/(Lz*Lz));
    double a2 = 4.0 / denom;
    double inv_h_sq = 1.0/(hx*hx) + 1.0/(hy*hy) + 1.0/(hz*hz);
    double tau = 0.8 / (sqrt(a2) * sqrt(inv_h_sq)); 
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
        cout << "MPI+CUDA Variant 4. Grid: " << N << "^3. Procs: " << nprocs << " (" << dims[0] << "x" << dims[1] << "x" << dims[2] << ")." << endl;
        cout << "Configuration: a^2=" << a2 << ", tau=" << tau << endl;
    }

    double *d_u_prev, *d_u_curr, *d_u_next, *d_error;
    CUDA_CHECK(cudaMalloc(&d_u_prev, total_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_curr, total_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_next, total_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_error, sizeof(double)));

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

    ctx.h_sb_x = (double*)malloc(ctx.sz_x); ctx.h_rb_x = (double*)malloc(ctx.sz_x);
    ctx.h_sb_y = (double*)malloc(ctx.sz_y); ctx.h_rb_y = (double*)malloc(ctx.sz_y);
    ctx.h_sb_z = (double*)malloc(ctx.sz_z); ctx.h_rb_z = (double*)malloc(ctx.sz_z);

    MPI_Cart_shift(cart_comm, 0, 1, &ctx.left, &ctx.right);
    MPI_Cart_shift(cart_comm, 1, 1, &ctx.down, &ctx.up);   
    MPI_Cart_shift(cart_comm, 2, 1, &ctx.back, &ctx.front); 

    dim3 block(8, 8, 8);
    dim3 grid((nx_loc + 7)/8, (ny_loc + 7)/8, (nz_loc + 7)/8);
    
    ctx.block2d = dim3(16, 16);
    ctx.grid_x = dim3((ny_loc+15)/16, (nz_loc+15)/16);
    ctx.grid_y = dim3((nx_loc+15)/16, (nz_loc+15)/16);
    ctx.grid_z = dim3((nx_loc+15)/16, (ny_loc+15)/16);

    dim3 bound_grid((nyg+15)/16, (nzg+15)/16);

    float time_calc = 0.0f, time_copy = 0.0f, time_comm = 0.0f, time_init = 0.0f;
    cudaEvent_t start, stop, e_start, e_stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventCreate(&e_start); cudaEventCreate(&e_stop);

    cudaEventRecord(start);
    init_kernel<<<grid, block>>>(d_u_prev, nx_loc, ny_loc, nz_loc, i_start, j_start, k_start, 
                                 hx, hy, hz, Lx, Ly, Lz, a_t, nyg, nzg);
    boundary_kernel<<<bound_grid, ctx.block2d>>>(d_u_prev, nx_loc, ny_loc, nz_loc, i_start, Nx_global, nyg, nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_init, start, stop);

    run_halo_exchange(d_u_prev, ctx, time_calc, time_copy, time_comm, e_start, e_stop);

    cudaEventRecord(start);
    step1_kernel<<<grid, block>>>(d_u_curr, d_u_prev, nx_loc, ny_loc, nz_loc, a2, tau, hx, hy, hz, nyg, nzg);
    boundary_kernel<<<bound_grid, ctx.block2d>>>(d_u_curr, nx_loc, ny_loc, nz_loc, i_start, Nx_global, nyg, nzg);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float t_step; cudaEventElapsedTime(&t_step, start, stop); time_calc += t_step;

    double global_max_err = 0.0;

    for (size_t n = 2; n <= Nt; ++n) {
        double t_physical = n * tau;
        
        run_halo_exchange(d_u_curr, ctx, time_calc, time_copy, time_comm, e_start, e_stop);

        cudaEventRecord(start);
        step_kernel<<<grid, block>>>(d_u_next, d_u_curr, d_u_prev, nx_loc, ny_loc, nz_loc, a2, tau, hx, hy, hz, nyg, nzg);
        boundary_kernel<<<bound_grid, ctx.block2d>>>(d_u_next, nx_loc, ny_loc, nz_loc, i_start, Nx_global, nyg, nzg);
        
        double zero = 0.0;
        cudaMemcpy(d_error, &zero, sizeof(double), cudaMemcpyHostToDevice);
        error_kernel<<<grid, block>>>(d_u_next, d_error, t_physical, nx_loc, ny_loc, nz_loc, 
                                      i_start, j_start, k_start, hx, hy, hz, Lx, Ly, Lz, a_t, nyg, nzg);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_step, start, stop); time_calc += t_step;

        double local_err;
        cudaMemcpy(&local_err, d_error, sizeof(double), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(e_start);
        double proc_err = 0.0;
        MPI_Reduce(&local_err, &proc_err, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
        cudaEventRecord(e_stop); cudaEventSynchronize(e_stop); 
        float t_red; cudaEventElapsedTime(&t_red, e_start, e_stop); time_comm += t_red;

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
        cout << "Initialization:                        " << time_init / 1000.0 << endl;
        cout << "Parallel Cycles (Kernels+Pack+Unpack): " << time_calc / 1000.0 << endl;
        cout << "Data Copy (Host <-> Device):           " << time_copy / 1000.0 << endl;
        cout << "Communication (MPI):                   " << time_comm / 1000.0 << endl;
        cout << "Total Halo Overhead:                   " << (time_copy + time_comm) / 1000.0 << endl;
        cout << "Total Time:                            " << (time_init + time_calc + time_copy + time_comm) / 1000.0 << endl;
    }

    cudaFree(d_u_prev); cudaFree(d_u_curr); cudaFree(d_u_next); cudaFree(d_error);
    cudaFree(ctx.d_sb_x); cudaFree(ctx.d_rb_x); cudaFree(ctx.d_sb_y); cudaFree(ctx.d_rb_y); cudaFree(ctx.d_sb_z); cudaFree(ctx.d_rb_z);
    free(ctx.h_sb_x); free(ctx.h_rb_x); free(ctx.h_sb_y); free(ctx.h_rb_y); free(ctx.h_sb_z); free(ctx.h_rb_z);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}