#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include <numeric>
#include <limits>
#include <tuple>
#include <functional>
using namespace std;


inline size_t idx(size_t i, size_t j, size_t k, size_t Ny, size_t Nz) {
    return (i * Ny + j) * Nz + k;
}


double u_analytic(double x, double y, double z, double t,
                  double Lx, double Ly, double Lz, double a_t) {
    return sin(3.0 * M_PI * x / Lx)
         * sin(2.0 * M_PI * y / Ly)
         * sin(2.0 * M_PI * z / Lz)
         * cos(a_t * t + 4.0 * M_PI);
}


int main() {
    //int THREADS_NUM = 16;
    //omp_set_num_threads(THREADS_NUM);
    double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    size_t N = 128;                  
    size_t Nx_nodes = N + 1;         
    size_t Ny_nodes = N;             
    size_t Nz_nodes = N;             
    double hx = Lx / N, hy = Ly / N, hz = Lz / N;

    //alphas
    double a_t = 2.0 * M_PI;
    double a2 = 4.0 / (9.0 / (Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz));

    
    double tau_max = 1.0 / (sqrt(a2) *
                     sqrt(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz)));
    double safety = 0.8;
    double tau = safety * tau_max;

    
    size_t Nt = 50; 
    double T = Nt * tau;            

    cout << "tau=" << tau << " CFL=" 
         << a2 * tau * tau * (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz)) 
         << "\n";

    cout << std::setprecision(10) << std::fixed;
    cout << "time" << "\t  " << "    delta" << "\n";
    //<< std::scientific 

    //memory
    size_t total = Nx_nodes * Ny_nodes * Nz_nodes;
    vector<double> u_prev(total, 0.0), u_curr(total, 0.0), u_next(total, 0.0);
    
    double start = omp_get_wtime();
    // u_analytic
    #pragma omp parallel for collapse(3)
    for (size_t i = 0; i < Nx_nodes; ++i)
        for (size_t j = 0; j < Ny_nodes; ++j)
            for (size_t k = 0; k < Nz_nodes; ++k) {
                double x = i * hx, y = j * hy, z = k * hz;
                u_prev[idx(i,j,k,Ny_nodes,Nz_nodes)] =
                    u_analytic(x, y, z, 0.0, Lx, Ly, Lz, a_t);
            }

    // x
    for (size_t j = 0; j < Ny_nodes; ++j)
        for (size_t k = 0; k < Nz_nodes; ++k) {
            u_prev[idx(0,j,k,Ny_nodes,Nz_nodes)] = 0.0;
            u_prev[idx(Nx_nodes-1,j,k,Ny_nodes,Nz_nodes)] = 0.0;
        }

    // u_1
    #pragma omp parallel for collapse(3)
    for (size_t i = 1; i <= Nx_nodes - 2; ++i)
        for (size_t j = 0; j < Ny_nodes; ++j)
            for (size_t k = 0; k < Nz_nodes; ++k) {
                size_t jm = (j == 0) ? Ny_nodes - 1 : j - 1;
                size_t jp = (j + 1 == Ny_nodes) ? 0 : j + 1;
                size_t km = (k == 0) ? Nz_nodes - 1 : k - 1;
                size_t kp = (k + 1 == Nz_nodes) ? 0 : k + 1;

                size_t id = idx(i,j,k,Ny_nodes,Nz_nodes);
                double lap =
                    (u_prev[idx(i-1,j,k,Ny_nodes,Nz_nodes)] - 2.0*u_prev[id] + u_prev[idx(i+1,j,k,Ny_nodes,Nz_nodes)])/(hx*hx)
                  + (u_prev[idx(i,jm,k,Ny_nodes,Nz_nodes)] - 2.0*u_prev[id] + u_prev[idx(i,jp,k,Ny_nodes,Nz_nodes)])/(hy*hy)
                  + (u_prev[idx(i,j,km,Ny_nodes,Nz_nodes)] - 2.0*u_prev[id] + u_prev[idx(i,j,kp,Ny_nodes,Nz_nodes)])/(hz*hz);

                u_curr[id] = u_prev[id] + 0.5 * a2 * tau * tau * lap;
            }

    
    for (size_t j = 0; j < Ny_nodes; ++j)
        for (size_t k = 0; k < Nz_nodes; ++k) {
            u_curr[idx(0,j,k,Ny_nodes,Nz_nodes)] = 0.0;
            u_curr[idx(Nx_nodes-1,j,k,Ny_nodes,Nz_nodes)] = 0.0;
        }

    
    double global_max_err = 0.0;
    for (size_t n = 2; n <= Nt; ++n) {
        double t = n * tau;

        // u
        #pragma omp parallel for collapse(3)
        for (size_t i = 1; i <= Nx_nodes - 2; ++i)
            for (size_t j = 0; j < Ny_nodes; ++j)
                for (size_t k = 0; k < Nz_nodes; ++k) {
                    size_t jm = (j == 0) ? Ny_nodes - 1 : j - 1;
                    size_t jp = (j + 1 == Ny_nodes) ? 0 : j + 1;
                    size_t km = (k == 0) ? Nz_nodes - 1 : k - 1;
                    size_t kp = (k + 1 == Nz_nodes) ? 0 : k + 1;

                    size_t id = idx(i,j,k,Ny_nodes,Nz_nodes);
                    double lap =
                        (u_curr[idx(i-1,j,k,Ny_nodes,Nz_nodes)] - 2.0*u_curr[id] + u_curr[idx(i+1,j,k,Ny_nodes,Nz_nodes)])/(hx*hx)
                      + (u_curr[idx(i,jm,k,Ny_nodes,Nz_nodes)] - 2.0*u_curr[id] + u_curr[idx(i,jp,k,Ny_nodes,Nz_nodes)])/(hy*hy)
                      + (u_curr[idx(i,j,km,Ny_nodes,Nz_nodes)] - 2.0*u_curr[id] + u_curr[idx(i,j,kp,Ny_nodes,Nz_nodes)])/(hz*hz);

                    u_next[id] = 2.0 * u_curr[id] - u_prev[id] + a2 * tau * tau * lap;
                }

        
        for (size_t j = 0; j < Ny_nodes; ++j)
            for (size_t k = 0; k < Nz_nodes; ++k) {
                u_next[idx(0,j,k,Ny_nodes,Nz_nodes)] = 0.0;
                u_next[idx(Nx_nodes-1,j,k,Ny_nodes,Nz_nodes)] = 0.0;
            }

        // delta
        double max_err = 0.0;
        #pragma omp parallel for collapse(3) reduction(max:max_err)
        for (size_t i = 0; i < Nx_nodes; ++i)
            for (size_t j = 0; j < Ny_nodes; ++j)
                for (size_t k = 0; k < Nz_nodes; ++k) {
                    double x = i * hx, y = j * hy, z = k * hz;
                    double ua = u_analytic(x, y, z, t, Lx, Ly, Lz, a_t);
                    double diff = fabs(u_next[idx(i,j,k,Ny_nodes,Nz_nodes)] - ua);
                    if (diff > max_err) max_err = diff;
                }
        cout << t << "  " << max_err << "\n";
        if (max_err > global_max_err) global_max_err = max_err;
        
        u_prev.swap(u_curr);
        u_curr.swap(u_next);
    }

    double end = omp_get_wtime();
    cout << "Final max error = " << global_max_err << "\n";
    cout << "Elapsed time: " << (end - start) << " seconds\n";
    return 0;
}
