#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;


inline size_t idx(size_t i, size_t j, size_t k, size_t nyg, size_t nzg) {
    return (i * nyg + j) * nzg + k;
}

int main(int argc, char** argv) {
    double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    size_t N = 128; 
    
    
    size_t nx = N; 
    size_t ny = N;
    size_t nz = N;

    double hx = Lx / N, hy = Ly / N, hz = Lz / N;

    
    double a_t = 2.0 * M_PI;
    double a2 = 4.0 / (9.0 / (Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz));
    double tau_max = 1.0 / (sqrt(a2) * sqrt(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz)));
    double tau = 0.8 * tau_max; 
    size_t Nt = 50; 

    
    size_t nx_alloc = nx + 1; 
    size_t nyg = ny + 2;      
    size_t nzg = nz + 2;      

    size_t total_size = nx_alloc * nyg * nzg;

    vector<double> u_prev(total_size, 0.0);
    vector<double> u_curr(total_size, 0.0);
    vector<double> u_next(total_size, 0.0);

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    cout << "# OpenMP Pure Implementation" << endl;
    cout << "# Grid: " << N << "^3. Threads: " << num_threads << endl;
    cout << fixed << setprecision(8);

    
    vector<double> sin_x(nx + 1);
    vector<double> sin_y(ny + 1);
    vector<double> sin_z(nz + 1);

    #pragma omp parallel for
    for(int i=0; i<=nx; i++) sin_x[i] = sin(3.0 * M_PI * (i * hx) / Lx);
    
    #pragma omp parallel for
    for(int j=1; j<=ny; j++) sin_y[j] = sin(2.0 * M_PI * ((j-1) * hy) / Ly); 
    
    #pragma omp parallel for
    for(int k=1; k<=nz; k++) sin_z[k] = sin(2.0 * M_PI * ((k-1) * hz) / Lz); 

    
    auto apply_boundaries = [&](vector<double>& arr) {
        
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx; ++i) { 
            for (int k = 1; k <= nz; ++k) {
                arr[idx(i, 0, k, nyg, nzg)]      = arr[idx(i, ny, k, nyg, nzg)]; 
                arr[idx(i, ny + 1, k, nyg, nzg)] = arr[idx(i, 1, k, nyg, nzg)];  
            }
        }

        
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx; ++i) { 
            for (int j = 0; j < nyg; ++j) { 
                arr[idx(i, j, 0, nyg, nzg)]      = arr[idx(i, j, nz, nyg, nzg)];
                arr[idx(i, j, nz + 1, nyg, nzg)] = arr[idx(i, j, 1, nyg, nzg)];
            }
        }
    };

    
    double t_start = omp_get_wtime();
    double t0_val = cos(4.0 * M_PI);

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            double v_xy = sin_x[i] * sin_y[j] * t0_val;
            for (int k = 1; k <= nz; ++k) {
                u_prev[idx(i,j,k,nyg,nzg)] = v_xy * sin_z[k];
            }
        }
    }
    apply_boundaries(u_prev);

    
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            for (int k = 1; k <= nz; ++k) {
                double center = u_prev[idx(i,j,k,nyg,nzg)];
                double lap = 
                    (u_prev[idx(i-1,j,k,nyg,nzg)] - 2.0*center + u_prev[idx(i+1,j,k,nyg,nzg)])/(hx*hx) +
                    (u_prev[idx(i,j-1,k,nyg,nzg)] - 2.0*center + u_prev[idx(i,j+1,k,nyg,nzg)])/(hy*hy) +
                    (u_prev[idx(i,j,k-1,nyg,nzg)] - 2.0*center + u_prev[idx(i,j,k+1,nyg,nzg)])/(hz*hz);
                
                u_curr[idx(i,j,k,nyg,nzg)] = center + 0.5 * a2 * tau * tau * lap;
            }
        }
    }
    apply_boundaries(u_curr);

    
    double global_max_err = 0.0;

    for (size_t n = 2; n <= Nt; ++n) {
        double t_physical = n * tau;

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx; ++i) {
            for (int j = 1; j <= ny; ++j) {
                for (int k = 1; k <= nz; ++k) {
                    double center = u_curr[idx(i,j,k,nyg,nzg)];
                    double lap = 
                        (u_curr[idx(i-1,j,k,nyg,nzg)] - 2.0*center + u_curr[idx(i+1,j,k,nyg,nzg)])/(hx*hx) +
                        (u_curr[idx(i,j-1,k,nyg,nzg)] - 2.0*center + u_curr[idx(i,j+1,k,nyg,nzg)])/(hy*hy) +
                        (u_curr[idx(i,j,k-1,nyg,nzg)] - 2.0*center + u_curr[idx(i,j,k+1,nyg,nzg)])/(hz*hz);
                    
                    u_next[idx(i,j,k,nyg,nzg)] = 2.0*center - u_prev[idx(i,j,k,nyg,nzg)] + a2 * tau * tau * lap;
                }
            }
        }
        
        apply_boundaries(u_next);

        
        double local_max_err = 0.0;
        double t_val = cos(a_t * t_physical + 4.0 * M_PI);

        #pragma omp parallel for collapse(2) reduction(max: local_max_err)
        for (int i = 1; i < nx; ++i) {
            for (int j = 1; j <= ny; ++j) {
                double v_xy = sin_x[i] * sin_y[j] * t_val;
                for (int k = 1; k <= nz; ++k) {
                    double ua = v_xy * sin_z[k];
                    double diff = fabs(u_next[idx(i,j,k,nyg,nzg)] - ua);
                    if (diff > local_max_err) local_max_err = diff;
                }
            }
        }

        cout << t_physical << " \t " << local_max_err << endl;
        if (local_max_err > global_max_err) global_max_err = local_max_err;

        
        u_prev.swap(u_curr);
        u_curr.swap(u_next);
    }

    double t_end = omp_get_wtime();

    cout << "Final max error = " << global_max_err << "\n";
    cout << "Elapsed time: " << (t_end - t_start) << " seconds" << endl;

    return 0;
}