#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <functional>
#include <cstdlib>
using namespace std;

inline size_t idx_local(size_t i, size_t j, size_t k, size_t nyg, size_t nzg) {
    return (i * nyg + j) * nzg + k;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    size_t N = 128; 
    size_t Nx_nodes = N + 1; 
    size_t Ny_nodes = N;     
    size_t Nz_nodes = N;     
    double hx = Lx / N, hy = Ly / N, hz = Lz / N;

    double a_t = 2.0 * M_PI;
    double a2 = 4.0 / (9.0 / (Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz));

    double tau_max = 1.0 / (sqrt(a2) *
                      sqrt(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz)));
    double safety = 0.8;
    double tau = safety * tau_max;

    size_t Nt = 50; 
    
    int dims[3] = {0,0,0};
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Dims_create(nprocs, 3, dims);
    
    int periods[3] = {0, 1, 1}; 
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    
    if (cart_comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    int rank;
    MPI_Comm_rank(cart_comm, &rank); 
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords); 

    auto compute_local = [](size_t globalN, int dim_size, int coord)->pair<int,int>{
        int base = static_cast<int>(globalN) / dim_size;
        int rem = static_cast<int>(globalN) % dim_size;
        int local = base + (coord < rem ? 1 : 0);
        int start = base * coord + min(coord, rem);
        return {local, start};
    };

    auto px = compute_local(Nx_nodes, dims[0], coords[0]);
    auto py = compute_local(Ny_nodes, dims[1], coords[1]);
    auto pz = compute_local(Nz_nodes, dims[2], coords[2]);

    int nx_loc = px.first; int i_start = px.second;
    int ny_loc = py.first; int j_start = py.second;
    int nz_loc = pz.first; int k_start = pz.second;

    int nxg = nx_loc + 2;
    int nyg = ny_loc + 2;
    int nzg = nz_loc + 2;
    size_t local_total = (size_t)nxg * nyg * nzg;

    if (rank==0) {
        cout << "# Grid: " << N << "^3. Topology: " << dims[0] << "x" << dims[1] << "x" << dims[2] << endl;
        cout << fixed << setprecision(8);
    }

    vector<double> u_prev(local_total, 0.0), u_curr(local_total, 0.0), u_next(local_total, 0.0);

    int nbr_left, nbr_right, nbr_down, nbr_up, nbr_back, nbr_front;
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_left, &nbr_right); 
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_down, &nbr_up);    
    MPI_Cart_shift(cart_comm, 2, 1, &nbr_back, &nbr_front); 
    
    vector<double> sin_x(nx_loc + 1), sin_y(ny_loc + 1), sin_z(nz_loc + 1);
    for(int i=1; i<=nx_loc; i++) sin_x[i] = sin(3.0 * M_PI * (i_start + i - 1) * hx / Lx);
    for(int j=1; j<=ny_loc; j++) sin_y[j] = sin(2.0 * M_PI * (j_start + j - 1) * hy / Ly);
    for(int k=1; k<=nz_loc; k++) sin_z[k] = sin(2.0 * M_PI * (k_start + k - 1) * hz / Lz);

    auto exchange_halos = [&](vector<double> &arr) {
        MPI_Status st;
        vector<double> sbuf, rbuf; 
        
        int face_size_x = ny_loc * nz_loc;
        sbuf.resize(face_size_x); rbuf.resize(face_size_x);

        int p = 0;
        for(int j=1; j<=ny_loc; j++) for(int k=1; k<=nz_loc; k++) sbuf[p++] = arr[idx_local(nx_loc, j, k, nyg, nzg)];
        
        MPI_Sendrecv(sbuf.data(), face_size_x, MPI_DOUBLE, nbr_right, 0,
                     rbuf.data(), face_size_x, MPI_DOUBLE, nbr_left, 0,
                     cart_comm, &st);
        
        if (nbr_left != MPI_PROC_NULL) {
            int p2=0;
            for(int j=1; j<=ny_loc; j++) for(int k=1; k<=nz_loc; k++) arr[idx_local(0, j, k, nyg, nzg)] = rbuf[p2++];
        } else {
            for(int j=0; j<nyg; j++) for(int k=0; k<nzg; k++) arr[idx_local(0, j, k, nyg, nzg)] = 0.0;
        }

        p = 0;
        for(int j=1; j<=ny_loc; j++) for(int k=1; k<=nz_loc; k++) sbuf[p++] = arr[idx_local(1, j, k, nyg, nzg)];

        MPI_Sendrecv(sbuf.data(), face_size_x, MPI_DOUBLE, nbr_left, 1,
                     rbuf.data(), face_size_x, MPI_DOUBLE, nbr_right, 1,
                     cart_comm, &st);

        if (nbr_right != MPI_PROC_NULL) {
            int p2=0;
            for(int j=1; j<=ny_loc; j++) for(int k=1; k<=nz_loc; k++) arr[idx_local(nx_loc+1, j, k, nyg, nzg)] = rbuf[p2++];
        } else {
            for(int j=0; j<nyg; j++) for(int k=0; k<nzg; k++) arr[idx_local(nx_loc+1, j, k, nyg, nzg)] = 0.0;
        }

        int face_size_y = nx_loc * nz_loc;
        sbuf.resize(face_size_y); rbuf.resize(face_size_y);

        p = 0;
        for(int i=1; i<=nx_loc; i++) for(int k=1; k<=nz_loc; k++) sbuf[p++] = arr[idx_local(i, ny_loc, k, nyg, nzg)];
        
        MPI_Sendrecv(sbuf.data(), face_size_y, MPI_DOUBLE, nbr_up, 2,
                     rbuf.data(), face_size_y, MPI_DOUBLE, nbr_down, 2,
                     cart_comm, &st);

        int p2=0;
        for(int i=1; i<=nx_loc; i++) for(int k=1; k<=nz_loc; k++) arr[idx_local(i, 0, k, nyg, nzg)] = rbuf[p2++];

        p = 0;
        for(int i=1; i<=nx_loc; i++) for(int k=1; k<=nz_loc; k++) sbuf[p++] = arr[idx_local(i, 1, k, nyg, nzg)];

        MPI_Sendrecv(sbuf.data(), face_size_y, MPI_DOUBLE, nbr_down, 3,
                     rbuf.data(), face_size_y, MPI_DOUBLE, nbr_up, 3,
                     cart_comm, &st);
        
        p2=0;
        for(int i=1; i<=nx_loc; i++) for(int k=1; k<=nz_loc; k++) arr[idx_local(i, ny_loc+1, k, nyg, nzg)] = rbuf[p2++];


        int face_size_z = nx_loc * ny_loc;
        sbuf.resize(face_size_z); rbuf.resize(face_size_z);

        p = 0;
        for(int i=1; i<=nx_loc; i++) for(int j=1; j<=ny_loc; j++) sbuf[p++] = arr[idx_local(i, j, nz_loc, nyg, nzg)];

        MPI_Sendrecv(sbuf.data(), face_size_z, MPI_DOUBLE, nbr_front, 4,
                     rbuf.data(), face_size_z, MPI_DOUBLE, nbr_back, 4,
                     cart_comm, &st);
        
        p2=0;
        for(int i=1; i<=nx_loc; i++) for(int j=1; j<=ny_loc; j++) arr[idx_local(i, j, 0, nyg, nzg)] = rbuf[p2++];

        p = 0;
        for(int i=1; i<=nx_loc; i++) for(int j=1; j<=ny_loc; j++) sbuf[p++] = arr[idx_local(i, j, 1, nyg, nzg)];

        MPI_Sendrecv(sbuf.data(), face_size_z, MPI_DOUBLE, nbr_back, 5,
                     rbuf.data(), face_size_z, MPI_DOUBLE, nbr_front, 5,
                     cart_comm, &st);
        
        p2=0;
        for(int i=1; i<=nx_loc; i++) for(int j=1; j<=ny_loc; j++) arr[idx_local(i, j, nz_loc+1, nyg, nzg)] = rbuf[p2++];
    };

    auto force_boundaries = [&](vector<double> &arr) {
        if (i_start == 0) {
             for(int j=0; j<nyg; j++) for(int k=0; k<nzg; k++) arr[idx_local(1, j, k, nyg, nzg)] = 0.0;
        }
        if (i_start + nx_loc == (int)Nx_nodes) {
             for(int j=0; j<nyg; j++) for(int k=0; k<nzg; k++) arr[idx_local(nx_loc, j, k, nyg, nzg)] = 0.0;
        }
    };

    double t0_factor = cos(a_t * 0.0 + 4.0 * M_PI);
    for (int i = 1; i <= nx_loc; ++i) {
        for (int j = 1; j <= ny_loc; ++j) {
            for (int k = 1; k <= nz_loc; ++k) {
                u_prev[idx_local(i,j,k,nyg,nzg)] = sin_x[i] * sin_y[j] * sin_z[k] * t0_factor;
            }
        }
    }
    force_boundaries(u_prev);

    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();

    exchange_halos(u_prev);
    
    for (int i = 1; i <= nx_loc; ++i) {
        for (int j = 1; j <= ny_loc; ++j) {
            for (int k = 1; k <= nz_loc; ++k) {
                double center = u_prev[idx_local(i,j,k,nyg,nzg)];
                double lap = 
                    (u_prev[idx_local(i-1,j,k,nyg,nzg)] - 2.0*center + u_prev[idx_local(i+1,j,k,nyg,nzg)])/(hx*hx) +
                    (u_prev[idx_local(i,j-1,k,nyg,nzg)] - 2.0*center + u_prev[idx_local(i,j+1,k,nyg,nzg)])/(hy*hy) +
                    (u_prev[idx_local(i,j,k-1,nyg,nzg)] - 2.0*center + u_prev[idx_local(i,j,k+1,nyg,nzg)])/(hz*hz);
                
                u_curr[idx_local(i,j,k,nyg,nzg)] = center + 0.5 * a2 * tau * tau * lap;
            }
        }
    }
    force_boundaries(u_curr);

    
    double global_max_err = 0.0;
    
    for (size_t n = 2; n <= Nt; ++n) {
        double t_physical = n * tau;
        
        exchange_halos(u_curr);

        for (int i = 1; i <= nx_loc; ++i) {
            for (int j = 1; j <= ny_loc; ++j) {
                for (int k = 1; k <= nz_loc; ++k) {
                    double center = u_curr[idx_local(i,j,k,nyg,nzg)];
                    double lap = 
                        (u_curr[idx_local(i-1,j,k,nyg,nzg)] - 2.0*center + u_curr[idx_local(i+1,j,k,nyg,nzg)])/(hx*hx) +
                        (u_curr[idx_local(i,j-1,k,nyg,nzg)] - 2.0*center + u_curr[idx_local(i,j+1,k,nyg,nzg)])/(hy*hy) +
                        (u_curr[idx_local(i,j,k-1,nyg,nzg)] - 2.0*center + u_curr[idx_local(i,j,k+1,nyg,nzg)])/(hz*hz);
                    
                    u_next[idx_local(i,j,k,nyg,nzg)] = 2.0*center - u_prev[idx_local(i,j,k,nyg,nzg)] + a2 * tau * tau * lap;
                }
            }
        }
        force_boundaries(u_next);

        
        double local_max_err = 0.0;
        double time_factor_t = cos(a_t * t_physical + 4.0 * M_PI);
        
        for (int i = 1; i <= nx_loc; ++i) {
            for (int j = 1; j <= ny_loc; ++j) {
                for (int k = 1; k <= nz_loc; ++k) {
                    double ua = sin_x[i] * sin_y[j] * sin_z[k] * time_factor_t;
                    double diff = fabs(u_next[idx_local(i,j,k,nyg,nzg)] - ua);
                    if (diff > local_max_err) local_max_err = diff;
                }
            }
        }

        double proc_max_err = 0.0;
        MPI_Reduce(&local_max_err, &proc_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
        
        if (rank == 0) {
            cout << t_physical << " \t " << proc_max_err << endl;
            if (proc_max_err > global_max_err) global_max_err = proc_max_err;
        }
        
        u_prev.swap(u_curr);
        u_curr.swap(u_next);
    }

    MPI_Barrier(cart_comm);
    double t_end = MPI_Wtime();

    if (rank == 0) {
        cout << "Final max error = " << global_max_err << "\n";
        cout << "Elapsed time: " << (t_end - t_start) << " seconds" << endl;
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}