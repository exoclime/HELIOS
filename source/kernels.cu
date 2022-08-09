// =================================================================================
// This file contains all the device functions and CUDA kernels.
// Copyright (C) 2018 - 2022 Matej Malik
// =================================================================================
// This file is part of HELIOS.
//
//     HELIOS is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     HELIOS is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.
//
//     You find a copy of the GNU General Public License in the main
//     HELIOS directory under <license.txt>. If not, see
//     <http://www.gnu.org/licenses/>.
// =================================================================================

#include<stdio.h>

// switch between double and single precision (currently, single prec. provides no speed-up and thus appears to be useless)
/***
#define USE_SINGLE
***/
#ifdef USE_SINGLE
typedef float utype;
#else
typedef double utype;
#endif


// physical constants
const utype PI = 3.141592653589793;
const utype HCONST = 6.62607004e-27;
const utype CSPEED = 29979245800.0;
const utype KBOLTZMANN = 1.38064852e-16;
const utype STEFANBOLTZMANN = 5.6703669999999995e-5; // yes, it needs to have this exact value to be consistent with astropy
const utype AMU = 1.6605390666e-24; // atomic mass unit (1/12 of mass of a C-12 atom)

// calculates the normal distribution
__device__ utype norm_pdf(
        utype x, 
        utype mu, 
        utype s
        ){

    return 1.0 / (s * sqrt(2.0 * PI)) * exp(-((x-mu)*(x-mu))/(2.0*(s*s)));
}


// computes the blackbody function for given wavelength & temperature
__device__ utype planck_func(utype lamda, utype T){
    
    utype num = 2.0 * HCONST * CSPEED * CSPEED;
    utype denom = pow(1.0*lamda, 5.0) * (exp(HCONST*CSPEED/(lamda*KBOLTZMANN*T)) - 1.0);
    
    return num / denom;
}


// atomicAdd for single precision
__device__ float atomicAdd_single(float* address, float value){

    float old = value;  
    float ret=atomicExch(address, 0.0f);
    float new_old=ret+old;

    while ((old = atomicExch(address, new_old))!=0.0f)
    {
        new_old = atomicExch(address, 0.0f);
        new_old += old;
    }
    return ret;
}


// atomicAdd for double precision
__device__ double atomicAdd_double(double* address, double val) {
    
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


// calculates analytically the integral of the planck function
__device__ utype analyt_planck(
        int 	n, 
        utype 	y1, 
        utype 	y2
){

    utype dn=n;

    return exp(-dn*y2) * ((y2*y2*y2)/dn + 3.0*(y2*y2)/(dn*dn) + 6.0*y2/(dn*dn*dn) + 6.0/(dn*dn*dn*dn))
            - exp(-dn*y1) * ((y1*y1*y1)/dn + 3.0*(y1*y1)/(dn*dn) + 6.0*y1/(dn*dn*dn) + 6.0/(dn*dn*dn*dn));
}


// fitting function for the E parameter according to "Heng, Malik & Kitzmann 2018
__device__ utype E_parameter(
        utype w0, 
        utype g0,
        utype i2s_transition
){
    utype E;
    
    if (w0 > i2s_transition && g0 >= 0){
        
        E = max(1.0, 1.225 - 0.1582*g0 - 0.1777*w0 - 0.07465*pow(1.0*g0, 2.0) + 0.2351*w0*g0 - 0.05582*pow(w0, 2.0));
    }
    else{
        E = 1.0;
    }
    return E;
}


//  calculates the transmission function
__device__ utype trans_func(
        utype epsi, 
        utype delta_tau, 
        utype w0, 
        utype g0, 
        int   scat_corr,
        utype i2s_transition
){

    utype E = 1.0;

    // improved scattering correction disabled for the following terms -- at least for the moment   
    if(scat_corr==1){
        E = E_parameter(w0, g0, i2s_transition);
    }
    
    return exp(-1.0/epsi*sqrt(E*(1.0 - w0*g0)*(E - w0))*delta_tau);
}


// calculates the G+ function
__device__ utype G_plus_func(
        utype w0, 
        utype g0, 
        utype epsi,
        utype epsi2,
        utype mu_star,
        int   scat_corr,
        utype i2s_transition
){

    utype E = 1.0;
    
    // improved scattering correction disabled for the following terms -- at least for the moment   
    if(scat_corr==1){
        E = E_parameter(w0, g0, i2s_transition);
    }
    
    utype num = w0 * (E * (1.0 - w0 * g0) + g0 * epsi / epsi2);

    utype denom = E * pow(epsi,-2.0) * (E - w0) * (1.0 - w0 * g0) - pow(mu_star,-2.0);

    utype second_term = 1.0/epsi + 1.0/(mu_star * E * (1.0 - w0 * g0));
    
    utype third_term = epsi * w0 * g0 * mu_star / (epsi2 * E * (1.0 - w0 * g0));
            
    utype bracket = num/denom * second_term + third_term;

    utype result =  0.5 * bracket;

    return result;
}


// calculates the G- function
__device__ utype G_minus_func(
        utype w0, 
        utype g0, 
        utype epsi,
        utype epsi2,
        utype mu_star,
        int   scat_corr,
        utype i2s_transition
){
    
     utype E = 1.0;

    // improved scattering correction disabled for the following terms -- at least for the moment   
    if(scat_corr==1){
        E = E_parameter(w0, g0, i2s_transition);
    }
    
    utype num = w0 * (E * (1.0 - w0 * g0) + g0 * epsi / epsi2);

    utype denom = E * pow(epsi,-2.0) * (E - w0) * (1.0 - w0 * g0) - pow(mu_star,-2.0);

    utype second_term = 1.0/epsi - 1.0/(mu_star * E * (1.0 - w0 * g0));
    
    utype third_term = epsi * w0 * g0 * mu_star / (epsi2 * E * (1.0 - w0 * g0));
            
    utype bracket = num/denom * second_term - third_term;

    utype result =  0.5 * bracket;

    return result;
}


// limiting the values of the G_plus and G_minus coefficients to 1e8. 
// This value is somewhat ad hoc from visual analysis. To justify, results are quite insensitive to this value.
__device__ utype G_limiter(
    utype G, 
    int   debug){
    
    if(abs(G) < 1e8){
        return G;	
    }
    else{
        if(debug == 1){
            printf("WARNING: G_functions are being artificially limited!!! \n");
        }
        return 1e8 * G / abs(G);
    }
}


// calculates the power operation with a foor loop -- is allegedly faster than the implemented pow() function
__device__ utype power_int(utype x, int i){

    utype result = 1.0;
    int j = 1;
    
    while(j<=i){
        result *= x;
        j++;
    }
    return result;
}


// calculates the single scattering albedo w0
__device__ utype single_scat_alb(
        utype scat_cross, 
        utype abs_cross, 
        utype w_0_limit
){
    
    return min(scat_cross / (scat_cross + abs_cross), w_0_limit);
}


// calculates the two-stream coupling coefficient Zeta_minus with the scattering coefficient E
__device__ utype zeta_minus(
        utype w0, 
        utype g0,
        int   scat_corr,
        utype i2s_transition
){
    utype E = 1.0;
    
    if(scat_corr==1){
        E = E_parameter(w0, g0, i2s_transition);
    }
    
    return 0.5 * (1.0 - sqrt((E - w0)/(E*(1.0 - w0*g0))) );
}


// calculates the two-stream coupling coefficient Zeta_plus with the scattering coefficient E
__device__ utype zeta_plus(
        utype w0, 
        utype g0,
        int scat_corr,
        utype i2s_transition
){
    utype E = 1.0;
    
    if(scat_corr==1){
        E = E_parameter(w0, g0, i2s_transition);
    }
    
    return 0.5 * (1.0 + sqrt((E - w0)/(E*(1.0 - w0*g0))) );
}


// calculates the derivative of the Planck function regarding temperature
__device__ utype dB_dT(
        utype lambda, 
        utype T
){

    utype D = 2.0 * HCONST * power_int(CSPEED, 3) * HCONST / (power_int(lambda, 6) * KBOLTZMANN * (T*T));

    utype num =  exp(HCONST * CSPEED / (lambda * KBOLTZMANN * T));

    utype denom = (exp( HCONST * CSPEED / (lambda * KBOLTZMANN * T)) - 1.0) * (exp( HCONST * CSPEED / (lambda * KBOLTZMANN * T)) - 1.0);

    utype result = D * num / denom ;

    return result;
}


// calculates the integral of the Planck derivative over a wavelength interval
__device__ utype integrated_dB_dT(
        utype* kw, 
        utype* ky, 
        int 	ny, 
        utype 	lambda_bot, 
        utype 	lambda_top, 
        utype 	T
){

    utype result = 0;

    for (int y=0;y<ny;y++){
        utype x = (ky[y]-0.5)*2.0;
        utype arg = (lambda_top-lambda_bot)/2.0 * x + (lambda_top+lambda_bot)/2.0;
        result += (lambda_top-lambda_bot)/2.0 * kw[y]* dB_dT(arg,T);
    }
    return result;
}


// calculates the exponential integral of 1st kind
__device__ utype expint1(
        utype x
){

    utype a[] = {-0.57721566,0.99999193,-0.24991055,0.05519968,-0.00976004,0.00107857};
    utype b[] = {1,8.5733287401,18.059016973,8.6347608925,0.2677737343};
    utype c[] = {1,9.5733223454,25.6329561486,21.0996530827,3.9584969228};

    utype result;

    if(x < 1){
        result = -log(x);
        for(int j=0;j<6;j++){
            result += a[j] * power_int(x,j);
        }
    }
    else{
        utype num=0;
        utype denom=0;
        for(int j=0;j<5;j++){
            num += b[j] * power_int(x,4 - j);
            denom += c[j] * power_int(x,4 - j);
            result = 1/x*exp(-x)*num/denom;
        }
    }
    return result;
}

// constructing a table with Planck function values for given wavelengths and in a suitable temperature range
__global__ void plancktable(
        utype* planck_grid, 
        utype* lambda_edge, 
        utype* deltalambda,
        int 	nwave, 
        utype 	Tstar, 
        int 	p_iter,
        int     dim,
        int     step
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int t = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nwave && t < (dim/10+1)) {

        utype T;
        utype shifty;
        utype D;
        utype y_bot;
        utype y_top;

        // building flexible temperature grid from '1 K' to 'dim * step - 1 K' at 'step K' resolution
        // and Tstar
        if(t < (dim/10)){
                T = (t + p_iter * (dim/10)) * step + 1;
        }
        if(p_iter == 9){
            if(t == dim/10){
                T = Tstar;
            }
        }

        planck_grid[x + (t + p_iter * (dim/10)) * nwave] = 0.0;

        // analytical calculation, only for T > 0
        if(T > 0.01){
            D = 2.0 * (power_int(KBOLTZMANN / HCONST, 3) * KBOLTZMANN * power_int(T, 4)) / (CSPEED*CSPEED);
            y_top = HCONST * CSPEED / (lambda_edge[x+1] * KBOLTZMANN * T);
            y_bot = HCONST * CSPEED / (lambda_edge[x] * KBOLTZMANN * T);

            // rearranging so that y_top < y_bot (i.e. wavelengths are always increasing)
            if(y_bot < y_top){
                shifty = y_top;
                y_top = y_bot;
                y_bot = shifty;
            }

            for(int n=1;n<200;n++){
                planck_grid[x + (t + p_iter * (dim/10)) * nwave] += D * analyt_planck(n, y_bot, y_top);
            }
        }
        planck_grid[x + (t + p_iter * (dim/10)) * nwave] /= deltalambda[x];
    }
}


// adjust the incident flux to correspond to the correct brightness temperature
__global__ void corr_inc_energy(
        utype* 	planck_grid,
        utype* 	starflux,
        utype* 	deltalambda,
        int 	realstar,
        int 	nwave, 
        utype 	Tstar,
        int     dim
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < nwave){

        utype num_flux = 0;

        if(realstar == 1){

            for (int xl = 0; xl < nwave; xl++){

                num_flux += deltalambda[xl] * starflux[xl];
            }
        }
        else{
            for (int xl = 0; xl < nwave; xl++){
                
                num_flux += deltalambda[xl] * PI * planck_grid[xl + dim * nwave];
                
            }
        }
        
        utype theo_flux = STEFANBOLTZMANN * pow(Tstar, 4.0);
        
        utype corr_factor = theo_flux / num_flux;
        if(x==0){
            if(corr_factor > 1) printf("\nEnergy budget corrected (increased) by %.2f percent.\n", 100.0 * (corr_factor - 1.0));
            if(corr_factor < 1) printf("\nEnergy budget corrected (decreased) by %.2f percent.\n", 100.0 * (1.0 - corr_factor));
        }
        if(realstar == 1){
            
            starflux[x] *= corr_factor;
        }
        else{
            
            planck_grid[x + dim * nwave] *= corr_factor;
            
        }
    }
}


// calculate cloud absorption and scattering cross sections and also g_0 total                   
__global__ void calc_total_g_0_of_gas_and_clouds(
        utype* scat_cross_lay_or_int,
        utype* g_0_all_clouds_lay_or_int,
        utype* scat_cross_all_clouds_lay_or_int,
        utype* g_0_tot_lay_or_int,
        utype  g_0,
        int    nbin, 
        int    nlay_or_nint
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nbin && i < nlay_or_nint) {

        // calculating the total (weighted average) scattering asymmetry factor
        utype num = g_0 * scat_cross_lay_or_int[x + nbin * i] + g_0_all_clouds_lay_or_int[x + nbin * i] * scat_cross_all_clouds_lay_or_int[x + nbin * i];
        utype denom = scat_cross_lay_or_int[x + nbin * i] + scat_cross_all_clouds_lay_or_int[x + nbin * i];
        g_0_tot_lay_or_int[x + nbin * i] = num / denom;
    }
}


// temperature interpolation for the non-isothermal layers
__global__ void temp_inter(
        utype* tlay, 
        utype* tint, 
        int numinterfaces,
        int itervalue
){
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (0 < i && i < numinterfaces - 1) {
        tint[i] = tlay[i - 1] + 0.5 * (tlay[i] - tlay[i - 1]);
    }
    if (i == 0) {
        tint[i] = tlay[i] - 0.5 * (tlay[i + 1] - tlay[i]);
    }
    if (i == numinterfaces - 1) {
        tint[i] = tlay[i - 1] + 0.5 * (tlay[i - 1] - tlay[i - 2]);
    }
//     // uncomment following lines for debugging
//     if(itervalue % 100 == 0){
//         if(i==0){
//             printf("Tsurf: %f, Tint[0]: %f, Tlay[0]: %f \n", tlay[numinterfaces - 1], tint[0], tlay[0]); 
//         }
//     }
}


// interpolate layer and interface opacities from opacity table
__global__ void opac_interpol(
    utype*  temp, 
    utype*  opactemp, 
    utype*  press, 
    utype*  opacpress,
    utype*  ktable, 
    utype*  opac,
    utype*  crosstable,
    utype*  scat_cross,
    int 	npress, 
    int 	ntemp, 
    int 	ny,
    int 	nbin, 
    int 	nlay_or_nint
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < nbin && i < nlay_or_nint) {
        
        utype deltaopactemp = (opactemp[ntemp-1] - opactemp[0])/(ntemp-1.0);
        utype deltaopacpress = (log10(opacpress[npress -1]) - log10(opacpress[0])) / (npress-1.0);
        utype t = (temp[i] - opactemp[0]) / deltaopactemp;
        
        t = min(ntemp-1.001, max(0.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);
        
        utype p = (log10(press[i]) - log10(opacpress[0])) / deltaopacpress;
        
        p = min(npress-1.001, max(0.001, p));
        
        int pdown = floor(p);
        int pup = ceil(p);
        
        if(pdown != pup && tdown != tup){
            for(int y=0;y<ny;y++){
                opac[y+ny*x + ny*nbin*i] =
                ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown] * (pup - p) * (tup - t)
                + ktable[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown] * (p - pdown) * (tup - t)
                + ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup] * (pup - p) * (t -  tdown)
                + ktable[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup] * (p - pdown) * (t - tdown);
            }
            
            scat_cross[x + nbin * i] =
            crosstable[x + nbin* pdown + nbin*npress * tdown] * (pup - p) * (tup - t)
            + crosstable[x + nbin* pup + nbin*npress * tdown] * (p - pdown) * (tup - t)
            + crosstable[x + nbin* pdown + nbin*npress * tup] * (pup - p) * (t -  tdown)
            + crosstable[x + nbin* pup + nbin*npress * tup] * (p - pdown) * (t - tdown);
        }
        
        if(tdown == tup && pdown != pup){
            for(int y=0;y<ny;y++){
                opac[y+ny*x + ny*nbin*i] = 
                ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown] * (pup - p)
                + ktable[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown] * (p - pdown);
            }
            
            scat_cross[x + nbin * i] =
            crosstable[x + nbin* pdown + nbin*npress * tdown] * (pup - p)
            + crosstable[x + nbin* pup + nbin*npress * tdown] * (p - pdown);
        }
        
        if(pdown == pup && tdown != tup){
            for(int y=0;y<ny;y++){
                opac[y+ny*x + ny*nbin*i] = 
                ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown] * (tup - t)
                + ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup] * (t -  tdown);
            }
            
            scat_cross[x + nbin * i] = 
            crosstable[x + nbin* pdown + nbin*npress * tdown] * (tup - t) 
            + crosstable[x + nbin* pdown + nbin*npress * tup] * (t -  tdown);
        }
        
        if(tdown == tup && pdown == pup){
            for(int y=0;y<ny;y++){
                
                opac[y+ny*x + ny*nbin*i] = ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown]; 
            }
            
            scat_cross[x + nbin * i] = crosstable[x + nbin* pdown + nbin*npress * tdown];
        }
    }
}


__device__ utype bilin_interpol_func(
        utype pdowntdown,
        utype puptdown,
        utype pdowntup,
        utype puptup,
        utype p,
        utype t,
        int pdown,
        int pup,
        int tdown,
        int tup
){
    utype interpolated;
    
    if(pdown != pup && tdown != tup){
        interpolated = pdowntdown * (pup - p) * (tup - t)
                    + puptdown * (p - pdown) * (tup - t)
                    + pdowntup * (pup - p) * (t - tdown)
                    + puptup * (p - pdown) * (t - tdown);
    }
    if(tdown == tup && pdown != pup){
        interpolated = pdowntdown * (pup - p) 
                    + puptdown * (p - pdown);
    }
    if(pdown == pup && tdown != tup){
        interpolated = pdowntup * (t - tdown) 
                    + pdowntdown * (tup - t);
    }
    if(tdown == tup && pdown == pup){
        interpolated = pdowntdown;
    }	
    return interpolated;
}


// interpolate the mean molecular mass for each layer
__global__ void meanmolmass_interpol(
        utype* temp, 
        utype* opactemp, 
        utype* meanmolmass, 
        utype* opac_meanmass,
        utype* press, 
        utype* opacpress,
        int 	npress, 
        int 	ntemp, 
        int 	ninterface
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < ninterface){

        utype deltaopactemp = (opactemp[ntemp-1] - opactemp[0])/(ntemp-1.0);
        utype deltaopacpress = (log10(opacpress[npress -1]) - log10(opacpress[0])) / (npress-1.0);
        utype t = (temp[i] - opactemp[0]) / deltaopactemp;

        t = min(ntemp-1.001, max(0.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);

        utype p = (log10(press[i]) - log10(opacpress[0])) / deltaopacpress;

        p = min(npress-1.001, max(0.001, p));
        
        int pdown = floor(p);
        int pup = ceil(p);

        if(tdown != tup && pdown != pup){
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown] * (pup - p) * (tup - t)
                            + opac_meanmass[pup + npress * tdown] * (p - pdown) * (tup - t)
                            + opac_meanmass[pdown + npress * tup] * (pup - p) * (t -  tdown)
                            + opac_meanmass[pup + npress * tup] * (p - pdown) * (t - tdown);
        }
        if(tdown != tup && pdown == pup){
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown] * (tup - t)
                            + opac_meanmass[pdown + npress * tup] * (t -  tdown);
        }
        if(tdown == tup && pdown != pup){
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown] * (pup - p)
                            + opac_meanmass[pup + npress * tdown] * (p - pdown);
        }
        if(tdown == tup && pdown == pup){
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown];
        }
    }
}


// interpolate kappa for each layer
__global__ void kappa_interpol(
    utype*  temp, 
    utype*  entr_temp, 
    utype*  press, 
    utype*  entr_press,
    utype*  kappa, 
    utype*  entr_kappa,
    int     entr_npress,
    int 	entr_ntemp,
    int     nlay_or_nint
){
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i < nlay_or_nint){
        
        utype delta_temp;
        utype t;
        
        delta_temp = (entr_temp[entr_ntemp-1] - entr_temp[0]) / (entr_ntemp-1.0);
        t = (temp[i] - entr_temp[0]) / delta_temp;

        t = min(entr_ntemp-1.001, max(0.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);
        
        utype delta_press = (log10(entr_press[entr_npress-1]) - log10(entr_press[0])) / (entr_npress-1.0);
        
        utype p = (log10(press[i]) - log10(entr_press[0])) / delta_press;
        
        p = min(entr_npress-1.001, max(0.001, p));
        
        int pdown = floor(p);
        int pup = ceil(p);
        
        if(tdown != tup && pdown != pup){
            kappa[i] = entr_kappa[pdown + entr_npress * tdown] * (pup - p) * (tup - t)
            + entr_kappa[pup + entr_npress * tdown] * (p - pdown) * (tup - t)
            + entr_kappa[pdown + entr_npress * tup] * (pup - p) * (t -  tdown)
            + entr_kappa[pup + entr_npress * tup] * (p - pdown) * (t - tdown);
        }
        if(tdown != tup && pdown == pup){
            kappa[i] = entr_kappa[pdown + entr_npress * tdown] * (tup - t)
            + entr_kappa[pdown + entr_npress * tup] * (t -  tdown);
        }
        if(tdown == tup && pdown != pup){
            kappa[i] = entr_kappa[pdown + entr_npress * tdown] * (pup - p)
            + entr_kappa[pup + entr_npress * tdown] * (p - pdown);
        }
        if(tdown == tup && pdown == pup){
            kappa[i] = entr_kappa[pdown + entr_npress * tdown];
        }
    }
}


// interpolate heat capacity for each layer
__global__ void cp_interpol(
        utype* temp, 
        utype* entr_temp, 
        utype* press, 
        utype* entr_press,
        utype* cp_lay, 
        utype* entr_cp,
        int 	entr_npress, 
        int 	entr_ntemp, 
        int 	nlayer
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < nlayer){

        utype delta_temp = (log10(entr_temp[entr_ntemp-1]) - log10(entr_temp[0])) / (entr_ntemp-1.0);
        utype delta_press = (log10(entr_press[entr_npress-1]) - log10(entr_press[0])) / (entr_npress-1.0);
        utype t = (log10(temp[i]) - log10(entr_temp[0])) / delta_temp;

        t = min(entr_ntemp-1.001, max(0.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);

        utype p = (log10(press[i]) - log10(entr_press[0])) / delta_press;

        p = min(entr_npress-1.001, max(0.001, p));
        
        int pdown = floor(p);
        int pup = ceil(p);

        if(tdown != tup && pdown != pup){
            cp_lay[i] = entr_cp[pdown + entr_npress * tdown] * (pup - p) * (tup - t)
                            + entr_cp[pup + entr_npress * tdown] * (p - pdown) * (tup - t)
                            + entr_cp[pdown + entr_npress * tup] * (pup - p) * (t -  tdown)
                            + entr_cp[pup + entr_npress * tup] * (p - pdown) * (t - tdown);
        }
        if(tdown != tup && pdown == pup){
            cp_lay[i] = entr_cp[pdown + entr_npress * tdown] * (tup - t)
                            + entr_cp[pdown + entr_npress * tup] * (t -  tdown);
        }
        if(tdown == tup && pdown != pup){
            cp_lay[i] = entr_cp[pdown + entr_npress * tdown] * (pup - p)
                            + entr_cp[pup + entr_npress * tdown] * (p - pdown);
        }
        if(tdown == tup && pdown == pup){
            cp_lay[i] = entr_cp[pdown + entr_npress * tdown];
        }
    }
}


// interpolate entropy for each layer
__global__ void entropy_interpol(
        utype* temp, 
        utype* entr_temp, 
        utype* press, 
        utype* entr_press,
        utype* entropy, 
        utype* entr_entropy,
        int 	entr_npress, 
        int 	entr_ntemp, 
        int 	nlayer
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < nlayer){

        utype delta_temp = (log10(entr_temp[entr_ntemp-1]) - log10(entr_temp[0])) / (entr_ntemp-1.0);
        utype delta_press = (log10(entr_press[entr_npress-1]) - log10(entr_press[0])) / (entr_npress-1.0);
        utype t = (log10(temp[i]) - log10(entr_temp[0])) / delta_temp;

        t = min(entr_ntemp-1.001, max(0.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);

        utype p = (log10(press[i]) - log10(entr_press[0])) / delta_press;

        p = min(entr_npress-1.001, max(0.001, p));
        
        int pdown = floor(p);
        int pup = ceil(p);

        if(tdown != tup && pdown != pup){
            entropy[i] = entr_entropy[pdown + entr_npress * tdown] * (pup - p) * (tup - t)
                            + entr_entropy[pup + entr_npress * tdown] * (p - pdown) * (tup - t)
                            + entr_entropy[pdown + entr_npress * tup] * (pup - p) * (t -  tdown)
                            + entr_entropy[pup + entr_npress * tup] * (p - pdown) * (t - tdown);
        }
        if(tdown != tup && pdown == pup){
            entropy[i] = entr_entropy[pdown + entr_npress * tdown] * (tup - t)
                            + entr_entropy[pdown + entr_npress * tup] * (t -  tdown);
        }
        if(tdown == tup && pdown != pup){
            entropy[i] = entr_entropy[pdown + entr_npress * tdown] * (pup - p)
                            + entr_entropy[pup + entr_npress * tdown] * (p - pdown);
        }
        if(tdown == tup && pdown == pup){
            entropy[i] = entr_entropy[pdown + entr_npress * tdown];
        }
    }
}


// interpolate water phase state number for each layer
__global__ void phase_number_interpol(
        utype* temp, 
        utype* entr_temp, 
        utype* press, 
        utype* entr_press,
        utype* state, 
        utype* entr_state,
        int 	entr_npress, 
        int 	entr_ntemp, 
        int 	nlayer
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < nlayer){

        utype delta_temp = (entr_temp[entr_ntemp-1] - entr_temp[0]) / (entr_ntemp-1.0);
        utype delta_press = (log10(entr_press[entr_npress-1]) - log10(entr_press[0])) / (entr_npress-1.0);
        utype t = (temp[i] - entr_temp[0]) / delta_temp;

        t = min(entr_ntemp-1.001, max(0.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);

        utype p = (log10(press[i]) - log10(entr_press[0])) / delta_press;

        p = min(entr_npress-1.001, max(0.001, p));
        
        int pdown = floor(p);
        int pup = ceil(p);

        if(tdown != tup && pdown != pup){
            state[i] = entr_state[pdown + entr_npress * tdown] * (pup - p) * (tup - t)
                            + entr_state[pup + entr_npress * tdown] * (p - pdown) * (tup - t)
                            + entr_state[pdown + entr_npress * tup] * (pup - p) * (t -  tdown)
                            + entr_state[pup + entr_npress * tup] * (p - pdown) * (t - tdown);
        }
        if(tdown != tup && pdown == pup){
            state[i] = entr_state[pdown + entr_npress * tdown] * (tup - t)
                            + entr_state[pdown + entr_npress * tup] * (t -  tdown);
        }
        if(tdown == tup && pdown != pup){
            state[i] = entr_state[pdown + entr_npress * tdown] * (pup - p)
                            + entr_state[pup + entr_npress * tdown] * (p - pdown);
        }
        if(tdown == tup && pdown == pup){
            state[i] = entr_state[pdown + entr_npress * tdown];
        }
    }
}


// interpolates the Planck function for the layer temperatures from the pre-tabulated values
__global__ void planck_interpol_layer(
    utype* 	temp, 
    utype* 	planckband_lay,
    utype* 	planck_grid, 
    utype* 	starflux, 
    int 	realstar, 
    int 	numlayers, 
    int 	nwave,
    int     dim,
    int     step
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < nwave && i < numlayers + 2){
        
        planckband_lay[i + x * (numlayers + 2)] = 0.0;
        
        // getting the stellar flux --- is redundant to do it every interpolation, but probably has negligible costs ...
        if (i == numlayers){
            if(realstar==1){
                planckband_lay[i + x * (numlayers + 2)] = starflux[x]/PI;
            }
            else{
                planckband_lay[i + x * (numlayers + 2)] = planck_grid[x + dim * nwave];
            }
        }
        else{
            utype t;
            
            // interpolating for layer temperatures
            if (i < numlayers){
                t = (temp[i] - 1.0) / step;
            }
            // interpolating for below (surface/BOA) temperature
            if (i == numlayers + 1){
                t = (temp[numlayers] - 1.0) / step;
            }
            
            t = max(0.001, min(dim - 1.001, t));
            
            int tdown = floor(t);
            int tup = ceil(t);
            
            if(tdown != tup){
                planckband_lay[i + x * (numlayers + 2)] = planck_grid[x + tdown * nwave] * (tup - t)
                + planck_grid[x + tup * nwave] * (t-tdown);
            }
            if(tdown == tup){
                planckband_lay[i + x * (numlayers + 2)] = planck_grid[x + tdown * nwave];
            }
        }
    }
}


// interpolates the Planck function for the interface temperatures from the pre-tabulated values
__global__ void planck_interpol_interface(
    utype* temp, 
    utype* planckband_int, 
    utype* planck_grid, 
    int 	numinterfaces, 
    int 	nwave,
    int     dim,
    int     step
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < nwave && i < numinterfaces){
        
        utype t = (temp[i] - 1.0) / step;
        
        t = max(0.001, min(dim - 1.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);
        
        if(tdown != tup){
            planckband_int[i + x * numinterfaces] = planck_grid[x + tdown * nwave] * (tup - t)
            + planck_grid[x + tup * nwave] * (t - tdown);
        }
        if(tdown == tup){
            planckband_int[i + x * numinterfaces] = planck_grid[x + tdown * nwave];
        }
    }
}


// calculation of transmission, w0, zeta-functions, and capital letters for the layer centers in the isothermal case
__global__ void calc_trans_iso(
        utype* 	trans_wg,
        utype* 	delta_tau_wg,
        utype* 	M_term,
        utype* 	N_term,
        utype* 	P_term,
        utype* 	G_plus,
        utype* 	G_minus,
        utype* 	delta_colmass,
        utype* 	opac_wg_lay,
        utype* 	meanmolmass_lay,
        utype* 	scat_cross_lay,
        utype* 	abs_cross_all_clouds_lay,
        utype* 	scat_cross_all_clouds_lay,
        utype* 	delta_tau_all_clouds,
        utype*  w_0,
        utype* 	g_0_tot_lay,
        int*    scat_trigger,
        utype   g_0,
        utype 	epsi,
        utype   epsi2,
        utype 	mu_star,
        utype   w_0_limit,
        utype   w_0_scat_limit,
        int 	scat,
        int 	nbin,
        int 	ny,
        int 	nlayer,
        int 	clouds,
        int 	scat_corr,
        int     debug,
        utype   i2s_transition
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < nbin && y < ny && i < nlayer) {

        utype ray_scat_cross;
        utype cloud_scat_cross;
        
        utype g0 = g_0;

        if(clouds == 1){
            g0 = g_0_tot_lay[x + nbin*i];
        }

        if (scat == 1){
            ray_scat_cross = scat_cross_lay[x + nbin*i];
            cloud_scat_cross = scat_cross_all_clouds_lay[x + nbin*i];
        }
        else{
            ray_scat_cross = 0;
            cloud_scat_cross = 0;
        }

        utype cloud_abs_cross = abs_cross_all_clouds_lay[x + nbin*i];
                
        // single scattering albedo
        w_0[y+ny*x + ny*nbin*i] = single_scat_alb(ray_scat_cross + cloud_scat_cross, opac_wg_lay[y+ny*x + ny*nbin*i]*meanmolmass_lay[i] + cloud_abs_cross, w_0_limit);
        utype w0 = w_0[y+ny*x + ny*nbin*i];
        
        // optical depth
        delta_tau_wg[y+ny*x + ny*nbin*i] = delta_colmass[i] * (opac_wg_lay[y+ny*x + ny*nbin*i] + ray_scat_cross/meanmolmass_lay[i]);        
        
        delta_tau_all_clouds[x + nbin*i] = delta_colmass[i] * (cloud_abs_cross + cloud_scat_cross)/meanmolmass_lay[i];
        
        utype del_tau = delta_tau_wg[y+ny*x + ny*nbin*i] + delta_tau_all_clouds[x + nbin*i];
        
        // transmission function
        trans_wg[y+ny*x + ny*nbin*i] = trans_func(epsi, del_tau, w0, g0, scat_corr, i2s_transition);
        utype trans = trans_wg[y+ny*x + ny*nbin*i];

        // two-stream scattering coupling coefficients
        utype zeta_min = zeta_minus(w0, g0, scat_corr, i2s_transition);
        utype zeta_pl = zeta_plus(w0, g0, scat_corr, i2s_transition);

        M_term[y+ny*x + ny*nbin*i] = (zeta_min*zeta_min) * (trans*trans) - (zeta_pl*zeta_pl);
        N_term[y+ny*x + ny*nbin*i] = zeta_pl * zeta_min * (1.0 - (trans*trans));
        P_term[y+ny*x + ny*nbin*i] = ((zeta_min*zeta_min) - (zeta_pl*zeta_pl)) * trans;
                
        G_plus[y+ny*x + ny*nbin*i] = G_limiter(G_plus_func(w0, g0, epsi, epsi2, mu_star, scat_corr, i2s_transition), debug);
        G_minus[y+ny*x + ny*nbin*i] = G_limiter(G_minus_func(w0, g0, epsi, epsi2, mu_star, scat_corr, i2s_transition), debug);
        
        // determine whether scattering or pure absorption will be used in the flux calculation for that wavelength bin and Gaussian point
        if(w0 > w_0_scat_limit) scat_trigger[y+ny*x] = 1;
    }
}

// calculation of transmission, w0, zeta-functions, and capital letters for the non-isothermal case
__global__ void calc_trans_noniso(
        utype* trans_wg_upper,
        utype* trans_wg_lower,
        utype* delta_tau_wg_upper,
        utype* delta_tau_wg_lower,
        utype* M_upper,
        utype* M_lower,
        utype* N_upper,
        utype* N_lower,
        utype* P_upper,
        utype* P_lower,
        utype* G_plus_upper,
        utype* G_plus_lower,
        utype* G_minus_upper,
        utype* G_minus_lower,
        utype* delta_col_upper,
        utype* delta_col_lower,
        utype* opac_wg_lay,
        utype* opac_wg_int,	
        utype* meanmolmass_lay,
        utype* meanmolmass_int,
        utype* scat_cross_lay,
        utype* scat_cross_int,
        utype* abs_cross_all_clouds_lay,
        utype* abs_cross_all_clouds_int,
        utype* scat_cross_all_clouds_lay,
        utype* scat_cross_all_clouds_int,
        utype* delta_tau_all_clouds_upper,
        utype* delta_tau_all_clouds_lower,
        utype* w_0_upper,
        utype* w_0_lower,
        utype* 	g_0_tot_lay,
        utype* 	g_0_tot_int,
        int*    scat_trigger,
        utype	g_0,
        utype 	epsi,
        utype   epsi2,
        utype 	mu_star,
        utype   w_0_limit,
        utype   w_0_scat_limit,
        int 	scat,
        int 	nbin,
        int 	ny,
        int 	nlayer,
        int 	clouds,
        int 	scat_corr,
        int     debug,
        utype   i2s_transition
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < nbin && y < ny && i < nlayer){

        utype ray_scat_cross_up;
        utype ray_scat_cross_low;
        utype cloud_scat_cross_up;
        utype cloud_scat_cross_low;
        
        utype g0_up = g_0;
        utype g0_low = g_0;
        
        if(clouds == 1){
            g0_up = (g_0_tot_lay[x + nbin*i] + g_0_tot_int[x + nbin*(i+1)]) / 2.0;
            g0_low = (g_0_tot_int[x + nbin*i] + g_0_tot_lay[x + nbin*i]) / 2.0;
        }

        if (scat == 1){
            ray_scat_cross_up = (scat_cross_lay[x + nbin*i] + scat_cross_int[x + nbin*(i+1)]) / 2.0;
            ray_scat_cross_low = (scat_cross_int[x + nbin*i] + scat_cross_lay[x + nbin*i]) / 2.0;
            cloud_scat_cross_up = (scat_cross_all_clouds_lay[x + nbin*i] + scat_cross_all_clouds_int[x + nbin*(i+1)]) / 2.0;
            cloud_scat_cross_low = (scat_cross_all_clouds_int[x + nbin*i] + scat_cross_all_clouds_lay[x + nbin*i]) / 2.0;
        }
        else{
            ray_scat_cross_up = 0;
            ray_scat_cross_low = 0;
            cloud_scat_cross_up = 0;
            cloud_scat_cross_low = 0;
        }
        
        utype cloud_abs_cross_up = (abs_cross_all_clouds_lay[x + nbin*i] + abs_cross_all_clouds_int[x + nbin*(i+1)])/2.0;
        utype cloud_abs_cross_low = (abs_cross_all_clouds_int[x + nbin*i] + abs_cross_all_clouds_lay[x + nbin*i])/2.0;
    
        utype opac_up = (opac_wg_lay[y+ny*x + ny*nbin*i]+opac_wg_int[y+ny*x + ny*nbin*(i+1)]) / 2.0;
        utype opac_low = (opac_wg_int[y+ny*x + ny*nbin*i]+opac_wg_lay[y+ny*x + ny*nbin*i]) / 2.0;
        
        utype meanmolmass_up = (meanmolmass_lay[i] + meanmolmass_int[i+1]) / 2.0;
        utype meanmolmass_low = (meanmolmass_int[i] + meanmolmass_lay[i]) / 2.0;
        
        // single scattering albedo
        w_0_upper[y+ny*x + ny*nbin*i] = single_scat_alb(ray_scat_cross_up + cloud_scat_cross_up, opac_up*meanmolmass_up + cloud_abs_cross_up, w_0_limit);
        
        utype w_0_up = w_0_upper[y+ny*x + ny*nbin*i];
        w_0_lower[y+ny*x + ny*nbin*i] = single_scat_alb(ray_scat_cross_low + cloud_scat_cross_low, opac_low*meanmolmass_low + cloud_abs_cross_low, w_0_limit);
        utype w_0_low = w_0_lower[y+ny*x + ny*nbin*i];
        
        // optical depth
        delta_tau_wg_upper[y+ny*x + ny*nbin*i] = delta_col_upper[i] * (opac_up + ray_scat_cross_up/meanmolmass_up);
        delta_tau_wg_lower[y+ny*x + ny*nbin*i] = delta_col_lower[i] * (opac_low + ray_scat_cross_low/meanmolmass_low);
        
        delta_tau_all_clouds_upper[x + nbin*i] = delta_col_upper[i] * (cloud_abs_cross_up + cloud_scat_cross_up)/meanmolmass_up;
        delta_tau_all_clouds_lower[x + nbin*i] = delta_col_lower[i] * (cloud_abs_cross_low + cloud_scat_cross_low)/meanmolmass_low;
        
        utype del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*i] + delta_tau_all_clouds_upper[x + nbin*i];
        utype del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*i] + delta_tau_all_clouds_lower[x + nbin*i];
        
        // transmission function
        trans_wg_upper[y+ny*x + ny*nbin*i] = trans_func(epsi, del_tau_up, w_0_up, g0_up, scat_corr, i2s_transition);
        utype trans_up = trans_wg_upper[y+ny*x + ny*nbin*i];
        trans_wg_lower[y+ny*x + ny*nbin*i] = trans_func(epsi, del_tau_low, w_0_low, g0_low, scat_corr, i2s_transition);
        utype trans_low = trans_wg_lower[y+ny*x + ny*nbin*i];
        
        // two-stream scattering coupling coefficients
        utype zeta_min_up = zeta_minus(w_0_up, g0_up, scat_corr, i2s_transition);
        utype zeta_min_low = zeta_minus(w_0_low, g0_low, scat_corr, i2s_transition);
        utype zeta_pl_up = zeta_plus(w_0_up, g0_up, scat_corr, i2s_transition);		
        utype zeta_pl_low = zeta_plus(w_0_low, g0_low, scat_corr, i2s_transition);
        
        M_upper[y+ny*x + ny*nbin*i] = (zeta_min_up*zeta_min_up) * (trans_up*trans_up) - (zeta_pl_up*zeta_pl_up);
        M_lower[y+ny*x + ny*nbin*i] = (zeta_min_low*zeta_min_low) * (trans_low*trans_low) - (zeta_pl_low*zeta_pl_low);
        N_upper[y+ny*x + ny*nbin*i] = zeta_pl_up * zeta_min_up * (1.0 - (trans_up*trans_up));
        N_lower[y+ny*x + ny*nbin*i] = zeta_pl_low * zeta_min_low * (1.0 - (trans_low*trans_low));
        P_upper[y+ny*x + ny*nbin*i] = ((zeta_min_up*zeta_min_up) - (zeta_pl_up*zeta_pl_up)) * trans_up;
        P_lower[y+ny*x + ny*nbin*i] = ((zeta_min_low*zeta_min_low) - (zeta_pl_low*zeta_pl_low)) * trans_low;
        
        G_plus_upper[y+ny*x + ny*nbin*i] = G_limiter(G_plus_func(w_0_up, g0_up, epsi, epsi2, mu_star, scat_corr, i2s_transition), debug);
        G_plus_lower[y+ny*x + ny*nbin*i] = G_limiter(G_plus_func(w_0_low, g0_low, epsi, epsi2, mu_star, scat_corr, i2s_transition), debug);
        G_minus_upper[y+ny*x + ny*nbin*i] = G_limiter(G_minus_func(w_0_up, g0_up, epsi, epsi2, mu_star, scat_corr, i2s_transition), debug);
        G_minus_lower[y+ny*x + ny*nbin*i] = G_limiter(G_minus_func(w_0_low, g0_low, epsi, epsi2, mu_star, scat_corr, i2s_transition), debug);
        
        // determine whether scattering or pure absorption will be used in the flux calculation for that wavelength bin and Gaussian point
        if(w_0_up > w_0_scat_limit) scat_trigger[y+ny*x] = 1;
        if(w_0_low > w_0_scat_limit) scat_trigger[y+ny*x] = 1;
    }
}


// calculates the height of a layer
__global__ void calc_delta_z(
        utype* 	tlay,
        utype* 	pint,
        utype* 	play,
        utype* 	meanmolmass_lay,
        utype* 	delta_z_lay,
        utype 	g,
        int		nlayer
){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < nlayer){
        delta_z_lay[i] = KBOLTZMANN * tlay[i] / (meanmolmass_lay[i] * g) * log(pint[i] / pint[i+1]);
    }
}


// calculates the direct beam flux with geometric zenith angle correction, isothermal version
__global__ void fdir_iso(
        utype* 	F_dir_wg,
        utype* 	planckband_lay,
        utype* 	delta_tau_wg,
        utype* 	z_lay,
        utype 	mu_star,
        utype	R_planet,
        utype 	R_star, 
        utype 	a,
        int		dir_beam,
        int		geom_zenith_corr,
        int 	ninterface,
        int 	nbin,
        int 	ny
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < ninterface && x < nbin && y < ny) {

        // the stellar intensity at TOA
        utype I_dir = ((R_star / a)*(R_star / a)) * PI * planckband_lay[(ninterface - 1) + x * (ninterface-1+2)];

        // initialize each flux value
        F_dir_wg[y + ny * x + ny * nbin * i]  = -dir_beam * mu_star * I_dir;

        utype mu_star_layer_j;

        // flux values lower that TOA will now be attenuated depending on their location
        for(int j = ninterface - 2; j >= i; j--){
            
            if(geom_zenith_corr == 1){
            mu_star_layer_j  = - sqrt(1.0 - pow((R_planet + z_lay[i])/(R_planet+z_lay[j]), 2.0) * (1.0 - pow(mu_star, 2.0)));
            }
            else{
                mu_star_layer_j = mu_star;
            }

            // direct stellar flux	
            F_dir_wg[y+ny*x+ny*nbin*i] *= exp(delta_tau_wg[y+ny*x + ny*nbin*j] / mu_star_layer_j);
        }
    }
}


// calculates the direct beam flux with geometric zenith angle correction, non-isothermal version
__global__ void fdir_noniso(
        utype* 	F_dir_wg,
        utype* 	Fc_dir_wg,
        utype* 	planckband_lay,
        utype* 	delta_tau_wg_upper,
        utype* 	delta_tau_wg_lower,
        utype* 	z_lay,
        utype 	mu_star,
        utype	R_planet,
        utype 	R_star, 
        utype 	a,
        int		dir_beam,
        int		geom_zenith_corr,
        int 	ninterface,
        int 	nbin,
        int 	ny
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < ninterface && x < nbin && y < ny) {

        // the stellar intensity at TOA
        utype I_dir = ((R_star / a)*(R_star / a)) * PI * planckband_lay[(ninterface - 1) + x * (ninterface-1+2)];

        // initialize each flux value
        F_dir_wg[y + ny * x + ny * nbin * i]  = -dir_beam * mu_star * I_dir;

        utype mu_star_layer_j;

        // flux values lower that TOA will now be attenuated depending on their location
        for(int j = ninterface - 2; j >= i; j--){

            if(geom_zenith_corr == 1){
                mu_star_layer_j  = - sqrt(1.0 - pow((R_planet + z_lay[i])/(R_planet+z_lay[j]), 2.0) * (1.0 - pow(mu_star, 2.0)));
            }
            else{
                mu_star_layer_j = mu_star;
            }
            
            utype delta_tau = delta_tau_wg_upper[y+ny*x + ny*nbin*j] + delta_tau_wg_lower[y+ny*x + ny*nbin*j];
            
            // direct stellar flux
            Fc_dir_wg[y+ny*x+ny*nbin*i] = F_dir_wg[y+ny*x+ny*nbin*i] * exp(delta_tau_wg_upper[y+ny*x + ny*nbin*j] / mu_star_layer_j);
            F_dir_wg[y+ny*x+ny*nbin*i] *= exp(delta_tau / mu_star_layer_j);
        }
    }
}


// calculation of the spectral fluxes, isothermal case with emphasis on on-the-fly calculations
__global__ void fband_iso(
    utype* F_down_wg, 
    utype* F_up_wg, 
    utype* F_dir_wg, 
    utype* planckband_lay,
    utype* w_0,
    utype* M_term,
    utype* N_term,
    utype* P_term,
    utype* G_plus,
    utype* G_minus,
    utype* surf_albedo,
    utype* g_0_tot_lay,
    utype  g_0,
    int    singlewalk, 
    utype  Rstar, 
    utype  a, 
    int    numinterfaces, 
    int    nbin, 
    utype  f_factor, 
    utype  mu_star,
    int    ny, 
    utype  epsi,
    int    dir_beam,
    int    clouds,
    int    scat_corr,
    int    debug,
    utype  i2s_transition
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < nbin && y < ny) {
        
        utype w0;
        utype M;
        utype N;
        utype P;
        utype G_pl;
        utype G_min;
        utype g0;	
        
        utype E;
                        
        utype flux_terms;
        utype planck_terms;
        utype direct_terms;
        
        // calculation of downward fluxes from TOA to BOA
        for (int i = numinterfaces - 1; i >= 0; i--){
            
            // TOA boundary -- incoming stellar flux
            if (i == numinterfaces - 1) {
                F_down_wg[y + ny * x + ny * nbin * i] = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[i + x * (numinterfaces-1+2)];
            }
            else {
                w0 = w_0[y+ny*x + ny*nbin*i];
                M = M_term[y+ny*x + ny*nbin*i];
                N = N_term[y+ny*x + ny*nbin*i];
                P = P_term[y+ny*x + ny*nbin*i];
                G_pl = G_plus[y+ny*x + ny*nbin*i];
                G_min = G_minus[y+ny*x + ny*nbin*i];
                g0 = g_0;
                
                if(clouds == 1){
                    g0 = g_0_tot_lay[x + nbin * i];
                }
                
                // improved scattering correction factor E
                E = 1.0;
                
                if(scat_corr==1){
                    E = E_parameter(w0, g0, i2s_transition);
                }

                // isothermal solution
                flux_terms = P * F_down_wg[y+ny*x+ny*nbin*(i+1)] - N * F_up_wg[y+ny*x+ny*nbin*i];
                
                planck_terms = planckband_lay[i+x*(numinterfaces-1+2)] * (N + M - P);
                
                direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min * M + G_pl * N) - F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) * P * G_min;
                
                direct_terms = min(0.0, direct_terms);

                F_down_wg[y+ny*x+ny*nbin*i] = 1.0 / M * (flux_terms + 2.0 * PI * epsi *(1.0 - w0)/(E - w0) * planck_terms + direct_terms);
                
                // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                if(abs(F_down_wg[y+ny*x+ny*nbin*i]) < 1e-100) F_down_wg[y+ny*x+ny*nbin*i] = abs(F_down_wg[y+ny*x+ny*nbin*i]);

                //feedback if flux becomes negative
                if(debug == 1){
                    if(F_down_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- downward flux is negative at layer: %d, w-index: %d, y-index: %d, flux value: %.3e !!! \n", i, x, y, F_down_wg[y+ny*x+ny*nbin*i]);
                }
            }
        }
        
        // calculation of upward fluxes from BOA to TOA
        for (int i = 0; i < numinterfaces; i++){
            
            // BOA boundary -- surface emission and reflection
            if (i == 0){
                
                utype reflected_part = surf_albedo[x] * (F_dir_wg[y+ny*x+ny*nbin*i] + F_down_wg[y+ny*x+ny*nbin* i]);
                
                // this is the surface/BOA emission. it correctly considers the emissivity e = (1 - albedo)
                utype BOA_part = (1.0 - surf_albedo[x]) * PI * (1.0 - w0)/(E - w0) * planckband_lay[numinterfaces + x * (numinterfaces-1+2)]; // remember: numinterfaces = numlayers + 1
                
                F_up_wg[y+ny*x+ny*nbin* i] = reflected_part + BOA_part; // internal_part consists of the internal heat flux plus the surface/BOA emission
            }
            else {
                w0 = w_0[y+ny*x + ny*nbin*(i-1)];
                M = M_term[y+ny*x + ny*nbin*(i-1)];
                N = N_term[y+ny*x + ny*nbin*(i-1)];
                P = P_term[y+ny*x + ny*nbin*(i-1)];
                G_pl = G_plus[y+ny*x + ny*nbin*(i-1)];
                G_min = G_minus[y+ny*x + ny*nbin*(i-1)];
                g0 = g_0;
                
                if(clouds == 1){
                    g0 = g_0_tot_lay[x + nbin * (i-1)];
                }
                
                // improved scattering correction factor E
                E = 1.0;
                
                if(scat_corr==1){
                    E = E_parameter(w0, g0, i2s_transition);
                }

                // isothermal solution
                flux_terms = P * F_up_wg[y+ny*x+ny*nbin*(i-1)] - N * F_down_wg[y+ny*x+ny*nbin*i];
                
                planck_terms = planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (N + M - P);
                
                direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min * N + G_pl * M) - F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P * G_pl;
                
                direct_terms = min(0.0, direct_terms);
                
                F_up_wg[y+ny*x+ny*nbin*i] = 1.0 / M * (flux_terms + 2.0 * PI * epsi *(1.0 - w0)/(E - w0) * planck_terms + direct_terms);
                
                // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                if(abs(F_up_wg[y+ny*x+ny*nbin*i]) < 1e-100) F_up_wg[y+ny*x+ny*nbin*i] = abs(F_up_wg[y+ny*x+ny*nbin*i]);

                //feedback if flux becomes negative
                if(debug == 1){
                    if(F_up_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- upward flux is negative at layer: %d, w-index: %d, y-index: %d !!! \n", i, x, y);
                }
            }
        }
    }
}


// calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
__global__ void fband_noniso(
    utype*  F_down_wg, 
    utype*  F_up_wg, 
    utype*  Fc_down_wg, 
    utype*  Fc_up_wg,
    utype*  F_dir_wg,
    utype*  Fc_dir_wg,
    utype*  planckband_lay, 
    utype*  planckband_int,
    utype*  w_0_upper,
    utype*  w_0_lower,
    utype*  delta_tau_wg_upper,
    utype*  delta_tau_wg_lower,
    utype*  delta_tau_all_clouds_upper,
    utype*  delta_tau_all_clouds_lower,
    utype*  M_upper,
    utype*  M_lower,
    utype*  N_upper,
    utype*  N_lower,
    utype*  P_upper,
    utype*  P_lower,
    utype*  G_plus_upper,
    utype*  G_plus_lower,
    utype*  G_minus_upper,
    utype*  G_minus_lower,
    utype*  surf_albedo,
    utype*  g_0_tot_lay,
    utype*  g_0_tot_int,
    utype 	g_0,
    int 	singlewalk, 
    utype 	Rstar, 
    utype 	a, 
    int 	numinterfaces,
    int 	nbin, 
    utype 	f_factor,
    utype 	mu_star,
    int 	ny,
    utype 	epsi,
    utype 	delta_tau_limit,
    int 	dir_beam,
    int 	clouds,
    int     scat_corr,
    int     debug,
    utype   i2s_transition
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < nbin && y < ny) {
        
        utype w0_up;
        utype del_tau_up;
        utype M_up;
        utype N_up;
        utype P_up;
        utype G_pl_up;
        utype G_min_up;
        utype g0_up;
        utype E_up;
                
        utype w0_low;
        utype del_tau_low;
        utype M_low;
        utype N_low;
        utype P_low;
        utype G_pl_low;
        utype G_min_low;
        utype g0_low;
        utype E_low;
        
        utype flux_terms;
        utype planck_terms;
        utype direct_terms;
                
        // calculation of downward fluxes from TOA to BOA
        for (int i = numinterfaces - 1; i >= 0; i--){
            
            // TOA boundary -- incoming stellar flux
            if (i == numinterfaces - 1) {
                F_down_wg[y + ny * x + ny * nbin * i] = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[i + x * (numinterfaces-1+2)];
            }
            else {
                // upper part of layer quantities
                w0_up = w_0_upper[y+ny*x + ny*nbin*i];
                del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*i] + delta_tau_all_clouds_upper[x + nbin*i];
                M_up = M_upper[y+ny*x + ny*nbin*i];
                N_up = N_upper[y+ny*x + ny*nbin*i];
                P_up = P_upper[y+ny*x + ny*nbin*i];
                G_pl_up = G_plus_upper[y+ny*x + ny*nbin*i];
                G_min_up = G_minus_upper[y+ny*x + ny*nbin*i];
                g0_up = g_0;
                
                // lower part of layer quantities
                w0_low = w_0_lower[y+ny*x + ny*nbin*i];
                del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*i] + delta_tau_all_clouds_lower[x + nbin*i];
                M_low = M_lower[y+ny*x + ny*nbin*i];
                N_low = N_lower[y+ny*x + ny*nbin*i];
                P_low = P_lower[y+ny*x + ny*nbin*i];
                G_pl_low = G_plus_lower[y+ny*x + ny*nbin*i];
                G_min_low = G_minus_lower[y+ny*x + ny*nbin*i];
                g0_low = g_0;
        
                if(clouds == 1){
                    g0_up = (g_0_tot_lay[x + nbin * i] + g_0_tot_int[x + nbin * (i+1)]) / 2.0;
                    g0_low = (g_0_tot_int[x + nbin * i] + g_0_tot_lay[x + nbin * i]) / 2.0;
                }
                
                // improved scattering correction factor E
                E_up = 1.0;
                E_low = 1.0;
                
                // improved scattering correction disabled for the following terms -- at least for the moment   
                if(scat_corr==1){
                    E_up = E_parameter(w0_up, g0_up, i2s_transition);
                    E_low = E_parameter(w0_low, g0_low, i2s_transition);
                }

                // upper part of layer calculations
                if(del_tau_up < delta_tau_limit){
                    // the isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                    planck_terms = (planckband_int[(i+1)+x*numinterfaces] + planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_up + M_up - P_up);
                }
                else{
                    // the non-isothermal solution -- standard case
                    utype pgrad_up = (planckband_lay[i + x * (numinterfaces-1+2)] - planckband_int[(i + 1) + x * numinterfaces]) / del_tau_up;
                    
                    planck_terms = planckband_lay[i+x*(numinterfaces-1+2)] * (M_up + N_up) - planckband_int[(i+1)+x*numinterfaces] * P_up + epsi / (E_up * (1.0-w0_up*g0_up))  * (P_up - M_up + N_up) * pgrad_up;
                }
                flux_terms = P_up * F_down_wg[y+ny*x+ny*nbin*(i+1)] - N_up * Fc_up_wg[y+ny*x+ny*nbin*i];
                
                direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * M_up + G_pl_up * N_up) - F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) * G_min_up * P_up;
                
                direct_terms = min(0.0, direct_terms);
                
                Fc_down_wg[y+ny*x+ny*nbin*i] = 1.0 / M_up * (flux_terms + 2.0*PI*epsi*(1.0 - w0_up)/(E_up - w0_up)*planck_terms + direct_terms);
                
                // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                if(abs(Fc_down_wg[y+ny*x+ny*nbin*i]) < 1e-100) Fc_down_wg[y+ny*x+ny*nbin*i] = abs(Fc_down_wg[y+ny*x+ny*nbin*i]);

                //feedback if flux becomes negative
                if(debug == 1){
                    if(Fc_down_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: %d, w-index: %d, y-index: %d !!! \n", i, x, y);
                }
                
                // lower part of layer calculations
                if(del_tau_low < delta_tau_limit){
                    // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                    planck_terms = (planckband_int[i+x*numinterfaces] + planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_low + M_low - P_low);
                }
                else{
                    // non-isothermal solution -- standard case
                    utype pgrad_low = (planckband_int[i + x * numinterfaces] - planckband_lay[i + x * (numinterfaces-1+2)]) / del_tau_low;

                    planck_terms = planckband_int[i+x*numinterfaces] * (M_low + N_low) - planckband_lay[i+x*(numinterfaces-1+2)] * P_low + epsi / (E_low * (1.0-w0_low*g0_low)) * (P_low - M_low + N_low) * pgrad_low;
                }
                flux_terms = P_low * Fc_down_wg[y+ny*x+ny*nbin*i] - N_low * F_up_wg[y+ny*x+ny*nbin*i];
                
                direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_low * M_low + G_pl_low * N_low) - Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * P_low * G_min_low;
                
                direct_terms = min(0.0, direct_terms);
                
                F_down_wg[y+ny*x+ny*nbin*i] = 1.0 / M_low * (flux_terms + 2.0*PI*epsi*(1.0 - w0_low)/(E_low - w0_low)*planck_terms + direct_terms);

                // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                if(abs(F_down_wg[y+ny*x+ny*nbin*i]) < 1e-100) F_down_wg[y+ny*x+ny*nbin*i] = abs(F_down_wg[y+ny*x+ny*nbin*i]);
                
                //feedback if flux becomes negative
                if(debug == 1){
                    if(F_down_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: %d, w-index: %d, y-index: %d !!! \n", i, x, y);
                }
            }
        }
        
        // calculation of upward fluxes from BOA to TOA
        for (int i = 0; i < numinterfaces; i++){
            
            // BOA boundary -- surface emission and reflection
            if (i == 0){
                
                utype reflected_part = surf_albedo[x] * (F_dir_wg[y+ny*x+ny*nbin*i] + F_down_wg[y+ny*x+ny*nbin* i]);

                // this is the surface/BOA emission. it correctly includes the emissivity e = (1 - albedo)
                utype BOA_part = (1.0 - surf_albedo[x]) * PI * (1.0 - w0_low)/(E_low - w0_low) * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
                
                F_up_wg[y+ny*x+ny*nbin* i] = reflected_part + BOA_part; // internal_part consists of the internal heat flux plus the surface/BOA emission
            }
            else {
                // lower part of layer quantities
                w0_low = w_0_lower[y+ny*x + ny*nbin*(i-1)];
                del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*(i-1)] + delta_tau_all_clouds_lower[x + nbin*(i-1)];
                M_low = M_lower[y+ny*x + ny*nbin*(i-1)];
                N_low = N_lower[y+ny*x + ny*nbin*(i-1)];
                P_low = P_lower[y+ny*x + ny*nbin*(i-1)];
                G_pl_low = G_plus_lower[y+ny*x + ny*nbin*(i-1)];
                G_min_low = G_minus_lower[y+ny*x + ny*nbin*(i-1)];
                g0_low = g_0;
                
                // upper part of layer quantities
                w0_up = w_0_upper[y+ny*x + ny*nbin*(i-1)];
                del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*(i-1)] + delta_tau_all_clouds_upper[x + nbin*(i-1)];
                M_up = M_upper[y+ny*x + ny*nbin*(i-1)];
                N_up = N_upper[y+ny*x + ny*nbin*(i-1)];
                P_up = P_upper[y+ny*x + ny*nbin*(i-1)];
                G_pl_up = G_plus_upper[y+ny*x + ny*nbin*(i-1)];
                G_min_up = G_minus_upper[y+ny*x + ny*nbin*(i-1)];
                g0_up = g_0;
                
                if(clouds == 1){
                    g0_low = (g_0_tot_int[x + nbin * (i-1)] + g_0_tot_lay[x + nbin * (i-1)]) / 2.0;
                    g0_up = (g_0_tot_lay[x + nbin * (i-1)] + g_0_tot_int[x + nbin * i]) / 2.0;
                }
                
                // improved scattering correction factor E
                E_low = 1.0;
                E_up = 1.0;
                       
                if(scat_corr==1){
                    E_up = E_parameter(w0_up, g0_up, i2s_transition);
                    E_low = E_parameter(w0_low, g0_low, i2s_transition);
                }
                
                // lower part of layer calculations
                if(del_tau_low < delta_tau_limit){
                    // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                    planck_terms = (planckband_int[(i-1)+x*numinterfaces] + planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_low + M_low - P_low);
                }
                else{
                    // non-isothermal solution -- standard case
                    utype pgrad_low = (planckband_int[(i-1) + x * numinterfaces] - planckband_lay[(i-1) + x * (numinterfaces-1+2)]) / del_tau_low;
                    
                    planck_terms = planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (M_low + N_low) - planckband_int[(i-1)+x*numinterfaces] * P_low + epsi/(E_low*(1.0-w0_low*g0_low)) * pgrad_low * (M_low - P_low - N_low);
                }
                flux_terms = P_low * F_up_wg[y+ny*x+ny*nbin*(i-1)] - N_low * Fc_down_wg[y+ny*x+ny*nbin*(i-1)];
                
                direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * (G_min_low * N_low + G_pl_low * M_low) - F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_low * G_pl_low;
                
                direct_terms = min(0.0, direct_terms);
                
                Fc_up_wg[y+ny*x+ny*nbin*(i-1)] = 1.0 / M_low * (flux_terms + 2.0*PI*epsi*(1.0 - w0_low)/(E_low - w0_low)*planck_terms + direct_terms);
                
                // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                if(abs(Fc_up_wg[y+ny*x+ny*nbin*i]) < 1e-100) Fc_up_wg[y+ny*x+ny*nbin*i] = abs(Fc_up_wg[y+ny*x+ny*nbin*i]);

                //feedback if flux becomes negative
                if(debug == 1){
                    if(Fc_up_wg[y+ny*x+ny*nbin*(i-1)] < 0) printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: %d, w-index: %d, y-index: %d !!! \n", i-1, x, y);
                }
                
                // upper part of layer calculations
                if(del_tau_up < delta_tau_limit){
                    // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                    planck_terms = (planckband_int[i+x*numinterfaces] + planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_up + M_up - P_up);
                }
                else{
                    // non-isothermal solution -- standard case
                    utype pgrad_up = (planckband_lay[(i-1) + x * (numinterfaces-1+2)] - planckband_int[i + x * numinterfaces]) / del_tau_up;
                    
                    planck_terms = planckband_int[i+x*numinterfaces] * (M_up + N_up) - planckband_lay[(i-1)+x*(numinterfaces-1+2)] * P_up + epsi/(E_up*(1.0-w0_up*g0_up)) * pgrad_up * (M_up - P_up - N_up);
                }
                flux_terms = P_up * Fc_up_wg[y+ny*x+ny*nbin*(i-1)] - N_up * F_down_wg[y+ny*x+ny*nbin*i];
                
                direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * N_up + G_pl_up * M_up) - Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_up * G_pl_up;
                
                direct_terms = min(0.0, direct_terms);
                
                F_up_wg[y+ny*x+ny*nbin*i] = 1.0 / M_up * (flux_terms + 2.0*PI*epsi*(1.0 - w0_up)/(E_up - w0_up)*planck_terms + direct_terms);
                
                // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                if(abs(F_up_wg[y+ny*x+ny*nbin*i]) < 1e-100) F_up_wg[y+ny*x+ny*nbin*i] = abs(F_up_wg[y+ny*x+ny*nbin*i]);

                //feedback if flux becomes negative
                if(debug == 1){
                    if(F_up_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: %d, w-index: %d, y-index: %d !!! \n", i, x, y);
                }
            }
        }
    }
}


// calculation of the spectral fluxes, isothermal case via matrix method (Thomas algorithm)
__global__ void fband_matrix_iso(
    utype* F_down_wg, 
    utype* F_up_wg, 
    utype* F_dir_wg, 
    utype* planckband_lay,
    utype* w_0,
    utype* M_term,
    utype* N_term,
    utype* P_term,
    utype* G_plus,
    utype* G_minus,
    utype* g_0_tot_lay,
    utype* alpha,
    utype* beta,
    utype* source_term_down,
    utype* source_term_up,
    utype* c_prime, 
    utype* d_prime,
    int*   scat_trigger,
    utype* trans_wg,
    utype* surf_albedo,
    utype  g_0,
    int    singlewalk, 
    utype  Rstar, 
    utype  a, 
    int    numinterfaces, 
    int    nbin, 
    utype  f_factor, 
    utype  mu_star,
    int    ny, 
    utype  epsi,
    int    dir_beam,
    int    clouds,
    int    scat_corr,
    int    debug,
    utype  i2s_transition
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < nbin && y < ny) {
        
        if (scat_trigger[y + ny * x] == 1){ // do matrix approach with scattering
            
            utype source_term_TOA;
            utype source_term_BOA;
            utype planck_terms;
            utype direct_terms_down;
            utype direct_terms_up;
            
            utype E = 1.0;
            utype M;
            utype N;
            utype P;
            utype w0;
            utype g0 = g_0;
            utype G_min;
            utype G_pl;
            
            // first loop for setting flux coefficients
            for (int j = 0; j < numinterfaces-1; j++){ // index j for layers (numlayers = numinterfaces - 1)
                
                M = M_term[y+ny*x + ny*nbin*j];
                N = N_term[y+ny*x + ny*nbin*j];
                P = P_term[y+ny*x + ny*nbin*j];
                w0 = w_0[y+ny*x + ny*nbin*j];
                G_min = G_minus[y+ny*x + ny*nbin*j];
                G_pl = G_plus[y+ny*x + ny*nbin*j];
                
                if(clouds == 1){
                    g0 = g_0_tot_lay[x + nbin * j];
                }
                
                // improved scattering correction factor E
                if(scat_corr==1){
                    E = E_parameter(w0, g0, i2s_transition);
                }
                
                alpha[y+ny*x + ny*nbin*j] = P / M;
                beta[y+ny*x + ny*nbin*j] = - N / M;
                
                planck_terms = 2.0 * PI * epsi *(1.0 - w0)/(E - w0) * (N + M - P) * planckband_lay[j+x*(numinterfaces-1+2)];
                
                direct_terms_down = F_dir_wg[y+ny*x+ny*nbin*j]/(-mu_star) * (G_min * M + G_pl * N) - F_dir_wg[y+ny*x+ny*nbin*(j+1)]/(-mu_star) * P * G_min;
                direct_terms_down = min(0.0, direct_terms_down);
                
                source_term_down[y+ny*x + ny*nbin*j] = 1.0 / M * (planck_terms + direct_terms_down);
                
                direct_terms_up = F_dir_wg[y+ny*x+ny*nbin*(j+1)]/(-mu_star) * (G_min * N + G_pl * M) - F_dir_wg[y+ny*x+ny*nbin*j]/(-mu_star) * P * G_pl;
                direct_terms_up = min(0.0, direct_terms_up);
                
                source_term_up[y+ny*x + ny*nbin*j] = 1.0 / M * (planck_terms + direct_terms_up);
                
            }
            
            // top and bottom boundary conditions
            source_term_TOA = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[(numinterfaces - 1) + x * (numinterfaces-1+2)];
            
            w0 = w_0[y+ny*x + ny*nbin*0];

            if(clouds == 1){
                g0 = g_0_tot_lay[x + nbin * 0];
            }
            // improved scattering correction factor E
            if(scat_corr==1){
                E = E_parameter(w0, g0, i2s_transition);
            }
            
            // reflected direct beam plus surface (or interior) emission
            source_term_BOA = surf_albedo[x] * F_dir_wg[y+ny*x+ny*nbin*0] + (1.0 - surf_albedo[x]) * PI * (1.0 - w0)/(E - w0) * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
            
            // second loop for populating matrix coefficients for Thomas algorithm
            int n_matrix = 2 * numinterfaces;
            
            // pre-setting 0 index so we can start loop from 1
            utype b_i = -surf_albedo[x];
            utype c_i = 1.0;
            utype d_i = source_term_BOA;
            
            c_prime[y+ny*x+ny*nbin*0] = c_i/b_i;
            d_prime[y+ny*x+ny*nbin*0] = d_i/b_i;
            
            utype c_i_min_1;
            
            for (int i = 1; i < n_matrix-1; i++){ // index i for matrix
                
                c_i_min_1 = c_i;
                
                if(i % 2==0){
                    b_i = - beta[y+ny*x + ny*nbin*(i/2 - 1)];
                    c_i = 1.0;
                    d_i = source_term_up[y+ny*x + ny*nbin*(i/2 - 1)];
                }
                else{
                    b_i = - beta[y+ny*x + ny*nbin*((i - 1)/2)];
                    c_i = - alpha[y+ny*x + ny*nbin*((i - 1)/2)];
                    d_i = source_term_down[y+ny*x + ny*nbin*((i - 1)/2)];
                }
                c_prime[y+ny*x+ny*nbin*i] = c_i/(b_i - c_i_min_1 * c_prime[y+ny*x+ny*nbin*(i-1)]);
                d_prime[y+ny*x+ny*nbin*i] = (d_i - c_i_min_1 * d_prime[y+ny*x+ny*nbin*(i-1)]) / (b_i - c_i_min_1 * c_prime[y+ny*x+ny*nbin*(i-1)]);
            }
            
            b_i = 0;
            d_i = source_term_TOA;
            c_i_min_1 = c_i;
            
            d_prime[y+ny*x+ny*nbin*(n_matrix-1)] = (d_i - c_i_min_1 * d_prime[y+ny*x+ny*nbin*(n_matrix-2)]) / (b_i - c_i_min_1 * c_prime[y+ny*x+ny*nbin*(n_matrix-2)]);
            
            // third loop, this time backward, to populate all the fluxes
            utype x_i = d_prime[y+ny*x+ny*nbin*(n_matrix-1)];
            F_up_wg[y+ny*x+ny*nbin*((n_matrix-2)/2)] = x_i;
            
            for (int i = n_matrix-2; i >= 0; i--){ // index i for matrix
                
                x_i = d_prime[y+ny*x+ny*nbin*i] - c_prime[y+ny*x+ny*nbin*i] * x_i;
                
                // translate x to correct fluxes
                if(i%2==0){
                    F_down_wg[y+ny*x+ny*nbin*(i/2)] = x_i;
                }
                else{
                    F_up_wg[y+ny*x+ny*nbin*((i-1)/2)] = x_i;
                }
            }
        }
        else{ // do standard loop with pure absorption equations
            
            utype trans;
            
            // calculation of downward fluxes from TOA to BOA
            for (int i = numinterfaces - 1; i >= 0; i--){
                
                // TOA boundary -- incoming stellar flux
                if (i == numinterfaces - 1) {
                    F_down_wg[y + ny * x + ny * nbin * i] = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[i + x * (numinterfaces-1+2)];
                }
                else {
                    
                    trans = trans_wg[y+ny*x + ny*nbin*i];
                    
                    F_down_wg[y+ny*x+ny*nbin*i] = trans * F_down_wg[y+ny*x+ny*nbin*(i+1)] + 2.0 * PI * epsi * (1.0 - trans) * planckband_lay[i+x*(numinterfaces-1+2)];
                    
                    // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                    if(abs(F_down_wg[y+ny*x+ny*nbin*i]) < 1e-100) F_down_wg[y+ny*x+ny*nbin*i] = abs(F_down_wg[y+ny*x+ny*nbin*i]);
                    //feedback if flux becomes negative
                    if(debug == 1){
                        if(F_down_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- downward flux is negative at layer: %d, w-index: %d, y-index: %d, flux value: %.3e !!! \n", i, x, y, F_down_wg[y+ny*x+ny*nbin*i]);
                    }
                }
            }
            
            // calculation of upward fluxes from BOA to TOA
            for (int i = 0; i < numinterfaces; i++){
                
                // BOA boundary -- surface emission and reflection
                if (i == 0){
                    
                    utype reflected_part = surf_albedo[x] * (F_dir_wg[y+ny*x+ny*nbin*i] + F_down_wg[y+ny*x+ny*nbin* i]);
                    
                    // this is the surface/BOA emission. it correctly considers the emissivity e = (1 - albedo)
                    utype BOA_part = (1.0 - surf_albedo[x]) * PI * planckband_lay[numinterfaces + x * (numinterfaces-1+2)]; // remember: numinterfaces = numlayers + 1
                    
                    F_up_wg[y+ny*x+ny*nbin* i] = reflected_part + BOA_part; // internal_part consists of the internal heat flux plus the surface/BOA emission
                }
                else{
                    
                    trans = trans_wg[y+ny*x + ny*nbin*(i-1)];
                    
                    F_up_wg[y+ny*x+ny*nbin*i] = trans * F_up_wg[y+ny*x+ny*nbin*(i-1)] + 2.0 * PI * epsi * (1.0 - trans) * planckband_lay[(i-1)+x*(numinterfaces-1+2)];
                    
                    // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                    if(abs(F_up_wg[y+ny*x+ny*nbin*i]) < 1e-100) F_up_wg[y+ny*x+ny*nbin*i] = abs(F_up_wg[y+ny*x+ny*nbin*i]);
                    //feedback if flux becomes negative
                    if(debug == 1){
                        if(F_up_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- upward flux is negative at layer: %d, w-index: %d, y-index: %d !!! \n", i, x, y);
                    }
                }
            }
        }
    }
}


// calculation of the spectral fluxes, matrix method (Thomas algorithm), non-isothermal case
__global__ void fband_matrix_noniso(
    utype*  F_down_wg, 
    utype*  F_up_wg, 
    utype*  Fc_down_wg, 
    utype*  Fc_up_wg,
    utype*  F_dir_wg, 
    utype*  Fc_dir_wg,
    utype*  planckband_lay,
    utype*  planckband_int,
    utype*  w_0_upper,
    utype*  w_0_lower,
    utype*  delta_tau_wg_upper,
    utype*  delta_tau_wg_lower,
    utype*  delta_tau_all_clouds_upper,
    utype*  delta_tau_all_clouds_lower,
    utype*  M_upper,
    utype*  M_lower,
    utype*  N_upper,
    utype*  N_lower,
    utype*  P_upper,
    utype*  P_lower,
    utype*  G_plus_upper,
    utype*  G_plus_lower,
    utype*  G_minus_upper,
    utype*  G_minus_lower,
    utype*  g_0_tot_lay,
    utype*  g_0_tot_int,
    utype*  alpha,
    utype*  beta,
    utype*  source_term_down,
    utype*  source_term_up,
    utype*  c_prime, 
    utype*  d_prime,
    int*    scat_trigger,
    utype*  trans_wg_upper,
    utype*  trans_wg_lower,
    utype*  surf_albedo,
    utype   g_0,
    int     singlewalk, 
    utype   Rstar, 
    utype   a, 
    int     numinterfaces, 
    int     nbin, 
    utype   f_factor, 
    utype   mu_star,
    int     ny, 
    utype   epsi,
    utype 	delta_tau_limit,
    int     dir_beam,
    int     clouds,
    int     scat_corr,
    int     debug,
    utype   i2s_transition
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < nbin && y < ny) {
        
        if (scat_trigger[y + ny * x] == 1){ // do matrix approach with scattering
            
            utype source_term_TOA;
            utype source_term_BOA;
            utype planck_terms_down;
            utype planck_terms_up;
            utype direct_terms_down;
            utype direct_terms_up;
            
            utype E = 1.0;
            utype M;
            utype N;
            utype P;
            utype w0;
            utype g0 = g_0;
            utype G_min;
            utype G_pl;
            utype del_tau;
            utype pgrad;
            
            // first loop for setting flux coefficients
            for (int j = 0; j < 2*(numinterfaces-1); j++){ // matrix coefficients go over 2 * number of layers
                
                if(j % 2==0){
                    
                    M = M_lower[y+ny*x + ny*nbin*(j/2)];
                    N = N_lower[y+ny*x + ny*nbin*(j/2)];
                    P = P_lower[y+ny*x + ny*nbin*(j/2)];

                    w0 = w_0_lower[y+ny*x + ny*nbin*(j/2)];
                    G_min = G_minus_lower[y+ny*x + ny*nbin*(j/2)];
                    G_pl = G_plus_lower[y+ny*x + ny*nbin*(j/2)];
                    del_tau = delta_tau_wg_lower[y+ny*x + ny*nbin*(j/2)] + delta_tau_all_clouds_lower[x + nbin*(j/2)];
                    
                    if(clouds == 1){
                        g0 = (g_0_tot_int[x + nbin * (j/2)] + g_0_tot_lay[x + nbin * (j/2)]) / 2.0;
                    }
                    
                    // improved scattering correction factor E
                    if(scat_corr==1){
                        E = E_parameter(w0, g0, i2s_transition);
                    }
                    
                    if(del_tau < delta_tau_limit){ // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                        
                        planck_terms_up = (N + M - P) * (planckband_int[(j/2)+x*numinterfaces] + planckband_lay[(j/2)+x*(numinterfaces-1+2)])/2.0;
                        
                        planck_terms_down = planck_terms_up;
                    }
                    else{ // non-isothermal solution -- standard case
                        
                        pgrad = (planckband_int[(j/2) + x * numinterfaces] - planckband_lay[(j/2) + x * (numinterfaces-1+2)]) / del_tau;
                        
                        planck_terms_down = (M + N) * planckband_int[(j/2)+x*numinterfaces] - P * planckband_lay[(j/2)+x*(numinterfaces-1+2)] + epsi / (E * (1.0-w0*g0))  * (P - M + N) * pgrad;
                        
                        planck_terms_up = (M + N) * planckband_lay[(j/2)+x*(numinterfaces-1+2)] - P * planckband_int[(j/2)+x*numinterfaces] + epsi / (E * (1.0-w0*g0))  * (M - N - P) * pgrad;
                    }
                    
                    direct_terms_down = F_dir_wg[y+ny*x+ny*nbin*(j/2)]/(-mu_star) * (G_min * M + G_pl * N) - Fc_dir_wg[y+ny*x+ny*nbin*(j/2)]/(-mu_star) * P * G_min;
                    
                    direct_terms_up = Fc_dir_wg[y+ny*x+ny*nbin*(j/2)]/(-mu_star) * (G_min * N + G_pl * M) - F_dir_wg[y+ny*x+ny*nbin*(j/2)]/(-mu_star) * P * G_pl;
                }
                else{
                    
                    M = M_upper[y+ny*x + ny*nbin*((j-1)/2)];
                    N = N_upper[y+ny*x + ny*nbin*((j-1)/2)];
                    P = P_upper[y+ny*x + ny*nbin*((j-1)/2)];
                    w0 = w_0_upper[y+ny*x + ny*nbin*((j-1)/2)];
                    G_min = G_minus_upper[y+ny*x + ny*nbin*((j-1)/2)];
                    G_pl = G_plus_upper[y+ny*x + ny*nbin*((j-1)/2)];
                    del_tau = delta_tau_wg_upper[y+ny*x + ny*nbin*((j-1)/2)] + delta_tau_all_clouds_upper[x + nbin*((j-1)/2)];
                    
                    if(clouds == 1){
                        g0 = (g_0_tot_int[x + nbin * ((j+1)/2)] + g_0_tot_lay[x + nbin * ((j-1)/2)]) / 2.0;
                    }
                    
                    // improved scattering correction factor E
                    if(scat_corr==1){
                        E = E_parameter(w0, g0, i2s_transition);
                    }
                    
                    if(del_tau < delta_tau_limit){ // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                        
                        planck_terms_up = (N + M - P) * (planckband_lay[((j-1)/2) + x*(numinterfaces-1+2)] + planckband_int[((j+1)/2) + x*numinterfaces])/2.0;
                        
                        planck_terms_down = planck_terms_up;
                        
                    }
                    else{ // non-isothermal solution -- standard case
                        
                        pgrad = (planckband_lay[((j-1)/2) + x * (numinterfaces-1+2)] - planckband_int[((j+1)/2) + x * numinterfaces]) / del_tau;
                        
                        planck_terms_down = (M + N) * planckband_lay[((j-1)/2)+x*(numinterfaces-1+2)] - P * planckband_int[((j+1)/2)+x*numinterfaces] + epsi / (E * (1.0-w0*g0))  * (P - M + N) * pgrad;
                        
                        planck_terms_up = (M + N) * planckband_int[((j+1)/2)+x*numinterfaces] - P * planckband_lay[((j-1)/2)+x*(numinterfaces-1+2)] + epsi / (E * (1.0-w0*g0))  * (M - N - P) * pgrad;
                    }
                    
                    direct_terms_down = Fc_dir_wg[y+ny*x+ny*nbin*((j-1)/2)]/(-mu_star) * (G_min * M + G_pl * N) - F_dir_wg[y+ny*x+ny*nbin*((j+1)/2)]/(-mu_star) * P * G_min;
                    
                    direct_terms_up = F_dir_wg[y+ny*x+ny*nbin*((j+1)/2)]/(-mu_star) * (G_min * N + G_pl * M) - Fc_dir_wg[y+ny*x+ny*nbin*((j-1)/2)]/(-mu_star) * P * G_pl;
                }
                
                direct_terms_down = min(0.0, direct_terms_down);
                direct_terms_up = min(0.0, direct_terms_up);
                
                alpha[y+ny*x + ny*nbin*j] = P / M;
                beta[y+ny*x + ny*nbin*j] = - N / M;
                
                source_term_down[y+ny*x + ny*nbin*j] = 1.0 / M * (2.0 * PI * epsi *(1.0 - w0)/(E - w0) * planck_terms_down + direct_terms_down);
                source_term_up[y+ny*x + ny*nbin*j] = 1.0 / M * (2.0 * PI * epsi *(1.0 - w0)/(E - w0) * planck_terms_up + direct_terms_up);
            }
            
            // top and bottom boundary conditions
            source_term_TOA = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[(numinterfaces - 1) + x * (numinterfaces-1+2)];
            
            w0 = w_0_lower[y+ny*x + ny*nbin*0];
            
            if(clouds == 1){
                g0 = (g_0_tot_int[x + nbin * 0] + g_0_tot_lay[x + nbin * 0]) / 2.0;
            }
            
            // improved scattering correction factor E
            if(scat_corr==1){
                E = E_parameter(w0, g0, i2s_transition);
            }
            
            // reflected direct beam plus surface (or interior) emission
            source_term_BOA = surf_albedo[x] * F_dir_wg[y+ny*x+ny*nbin*0] + (1.0 - surf_albedo[x]) * PI * (1.0 - w0)/(E - w0) * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
            
            // second loop for populating matrix coefficients for Thomas algorithm
            int n_matrix = 4 * numinterfaces - 2;
            
            // pre-setting 0 index so we can start loop from 1
            utype b_i = -surf_albedo[x];
            utype c_i = 1.0;
            utype d_i = source_term_BOA;
            
            c_prime[y+ny*x+ny*nbin*0] = c_i/b_i;
            d_prime[y+ny*x+ny*nbin*0] = d_i/b_i;
            
            utype c_i_min_1;
            
            for (int i = 1; i < n_matrix-1; i++){ // index i for matrix
                
                c_i_min_1 = c_i;
                
                if(i % 2==0){
                    b_i = - beta[y+ny*x + ny*nbin*(i/2 - 1)];
                    c_i = 1.0;
                    d_i = source_term_up[y+ny*x + ny*nbin*(i/2 - 1)];
                }
                else{
                    b_i = - beta[y+ny*x + ny*nbin*((i - 1)/2)];
                    c_i = - alpha[y+ny*x + ny*nbin*((i - 1)/2)];
                    d_i = source_term_down[y+ny*x + ny*nbin*((i - 1)/2)];
                }
                c_prime[y+ny*x+ny*nbin*i] = c_i/(b_i - c_i_min_1 * c_prime[y+ny*x+ny*nbin*(i-1)]);
                d_prime[y+ny*x+ny*nbin*i] = (d_i - c_i_min_1 * d_prime[y+ny*x+ny*nbin*(i-1)]) / (b_i - c_i_min_1 * c_prime[y+ny*x+ny*nbin*(i-1)]);
            }
            
            b_i = 0;
            d_i = source_term_TOA;
            c_i_min_1 = c_i;
            
            d_prime[y+ny*x+ny*nbin*(n_matrix-1)] = (d_i - c_i_min_1 * d_prime[y+ny*x+ny*nbin*(n_matrix-2)]) / (b_i - c_i_min_1 * c_prime[y+ny*x+ny*nbin*(n_matrix-2)]);
            
            // third loop, this time backward, to populate all the fluxes using the matrix solution
            utype x_i = d_prime[y+ny*x+ny*nbin*(n_matrix-1)];
            
            F_up_wg[y+ny*x+ny*nbin*(numinterfaces - 1)] = x_i;
            
            for (int i = n_matrix-2; i >= 0; i--){ // index i for matrix
                
                x_i = d_prime[y+ny*x+ny*nbin*i] - c_prime[y+ny*x+ny*nbin*i] * x_i;

                // remove tiny negative fluxes caused by limited numerical precision
                if(x_i < 1e-100) x_i = abs(x_i);
                
                //feedback if flux is still negative (because that may be a bug then...)
                if(debug == 1){
                    if(x_i < 0) printf("WARNING WARNING WARNING WARNING -- negative flux found at matrix index: %d, wavelength index: %d, ypoint index: %d !!! \n", i, x, y);
                }
                
                // translate to numerical grid fluxes
                if(i%4==0){
                    F_down_wg[y+ny*x+ny*nbin*(i/4)] = x_i;
                }
                if(i%4==1){
                    F_up_wg[y+ny*x+ny*nbin*((i-1)/4)] = x_i;
                }
                if(i%4==2){
                    Fc_down_wg[y+ny*x+ny*nbin*((i-2)/4)] = x_i;
                }
                if(i%4==3){
                    Fc_up_wg[y+ny*x+ny*nbin*((i-3)/4)] = x_i;
                }
            }
        }
        else{ // do standard loop with pure absorption equations
            
            utype trans_up;
            utype trans_low;
            utype del_tau_up;
            utype del_tau_low;
            utype planck_terms;
            
            // calculation of downward fluxes from TOA to BOA
            for (int i = numinterfaces - 1; i >= 0; i--){
                
                // TOA boundary -- incoming stellar flux
                if (i == numinterfaces - 1) {
                    F_down_wg[y + ny * x + ny * nbin * i] = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[i + x * (numinterfaces-1+2)];
                }
                else {
                    // upper part of layer quantities
                    trans_up = trans_wg_upper[y+ny*x + ny*nbin*i];
                    del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*i] + delta_tau_all_clouds_upper[x + nbin*i];
                    
                    // lower part of layer quantities
                    trans_low = trans_wg_lower[y+ny*x + ny*nbin*i];
                    del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*i] + delta_tau_all_clouds_lower[x + nbin*i];
                    
                    // upper part of layer calculations
                    if(del_tau_up < delta_tau_limit){
                        
                        // the isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                        planck_terms = (planckband_int[(i+1)+x*numinterfaces] + planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (1.0 - trans_up);
                    }
                    else{
                        // the non-isothermal solution -- standard case
                        utype pgrad_up = (planckband_lay[i + x * (numinterfaces-1+2)] - planckband_int[(i + 1) + x * numinterfaces]) / del_tau_up;
                        
                        planck_terms = planckband_lay[i+x*(numinterfaces-1+2)] - trans_up * planckband_int[(i+1)+x*numinterfaces] + epsi * (trans_up - 1.0) * pgrad_up;
                    }
                    
                    Fc_down_wg[y+ny*x+ny*nbin*i] = trans_up * F_down_wg[y+ny*x+ny*nbin*(i+1)] + 2.0 * PI * epsi * planck_terms;
                    
                    // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                    if(abs(Fc_down_wg[y+ny*x+ny*nbin*i]) < 1e-100) Fc_down_wg[y+ny*x+ny*nbin*i] = abs(Fc_down_wg[y+ny*x+ny*nbin*i]);
                    //feedback if flux becomes negative
                    if(debug == 1){
                        if(Fc_down_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: %d, w-index: %d, y-index: %d !!! \n", i, x, y);
                    }
                    
                    // lower part of layer calculations
                    if(del_tau_low < delta_tau_limit){
                        
                        // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                        planck_terms = (planckband_int[i+x*numinterfaces] + planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (1.0 - trans_low);
                    }
                    else{
                        // non-isothermal solution -- standard case
                        utype pgrad_low = (planckband_int[i + x * numinterfaces] - planckband_lay[i + x * (numinterfaces-1+2)]) / del_tau_low;
                        
                        planck_terms = planckband_int[i+x*numinterfaces] - trans_low * planckband_lay[i+x*(numinterfaces-1+2)] + epsi * (trans_low - 1.0) * pgrad_low;
                    }
                    
                    F_down_wg[y+ny*x+ny*nbin*i] = trans_low * Fc_down_wg[y+ny*x+ny*nbin*i] + 2.0 * PI * epsi * planck_terms;
                    
                    // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                    if(abs(F_down_wg[y+ny*x+ny*nbin*i]) < 1e-100) F_down_wg[y+ny*x+ny*nbin*i] = abs(F_down_wg[y+ny*x+ny*nbin*i]);
                    //feedback if flux becomes negative
                    if(debug == 1){
                        if(F_down_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: %d, w-index: %d, y-index: %d !!! \n", i, x, y);
                    }
                }
            }
            
            // calculation of upward fluxes from BOA to TOA
            for (int i = 0; i < numinterfaces; i++){
                
                // BOA boundary -- surface emission and reflection
                if (i == 0){
                    
                    utype reflected_part = surf_albedo[x] * (F_dir_wg[y+ny*x+ny*nbin*i] + F_down_wg[y+ny*x+ny*nbin* i]);
                    
                    // this is the surface/BOA emission. it correctly includes the emissivity e = (1 - albedo)
                    utype BOA_part = (1.0 - surf_albedo[x]) * PI * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
                    
                    F_up_wg[y+ny*x+ny*nbin* i] = reflected_part + BOA_part; // internal_part consists of the internal heat flux plus the surface/BOA emission
                }
                else {
                    
                    // upper part of layer quantities
                    trans_up = trans_wg_upper[y+ny*x + ny*nbin*(i-1)];
                    del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*(i-1)] + delta_tau_all_clouds_upper[x + nbin*(i-1)];
                    
                    // lower part of layer quantities
                    trans_low = trans_wg_lower[y+ny*x + ny*nbin*(i-1)];
                    del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*(i-1)] + delta_tau_all_clouds_lower[x + nbin*(i-1)];
                    
                    // lower part of layer calculations
                    if(del_tau_low < delta_tau_limit){
                        // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                        planck_terms = (planckband_int[(i-1)+x*numinterfaces] + planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (1.0 - trans_low);
                    }
                    else{
                        // non-isothermal solution -- standard case
                        utype pgrad_low = (planckband_int[(i-1) + x * numinterfaces] - planckband_lay[(i-1) + x * (numinterfaces-1+2)]) / del_tau_low;
                        
                        planck_terms = planckband_lay[(i-1)+x*(numinterfaces-1+2)] - trans_low * planckband_int[(i-1)+x*numinterfaces] + epsi * pgrad_low * (1.0 - trans_low);
                    }
                    
                    Fc_up_wg[y+ny*x+ny*nbin*(i-1)] = trans_low* F_up_wg[y+ny*x+ny*nbin*(i-1)] + 2.0 * PI * epsi * planck_terms;
                    
                    // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                    if(abs(Fc_up_wg[y+ny*x+ny*nbin*i]) < 1e-100) Fc_up_wg[y+ny*x+ny*nbin*i] = abs(Fc_up_wg[y+ny*x+ny*nbin*i]);
                    //feedback if flux becomes negative
                    if(debug == 1){
                        if(Fc_up_wg[y+ny*x+ny*nbin*(i-1)] < 0) printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: %d, w-index: %d, y-index: %d !!! \n", i-1, x, y);
                    }
                    
                    // upper part of layer calculations
                    if(del_tau_up < delta_tau_limit){
                        // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                        planck_terms = (planckband_int[i+x*numinterfaces] + planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (1.0 - trans_up);
                    }
                    else{
                        // non-isothermal solution -- standard case
                        utype pgrad_up = (planckband_lay[(i-1) + x * (numinterfaces-1+2)] - planckband_int[i + x * numinterfaces]) / del_tau_up;
                        
                        planck_terms = planckband_int[i+x*numinterfaces] - trans_up * planckband_lay[(i-1)+x*(numinterfaces-1+2)] + epsi * pgrad_up * (1.0 - trans_up);
                    }
                    
                    F_up_wg[y+ny*x+ny*nbin*i] = trans_up * Fc_up_wg[y+ny*x+ny*nbin*(i-1)] + 2.0 * PI * epsi * planck_terms;
                    
                    // making infinitely small terms positive to avoid false positive Warnings about negative fluxes
                    if(abs(F_up_wg[y+ny*x+ny*nbin*i]) < 1e-100) F_up_wg[y+ny*x+ny*nbin*i] = abs(F_up_wg[y+ny*x+ny*nbin*i]);
                    //feedback if flux becomes negative
                    if(debug == 1){
                        if(F_up_wg[y+ny*x+ny*nbin*i] < 0) printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: %d, w-index: %d, y-index: %d !!! \n", i, x, y);
                    }
                }
            }
        }
    }
}


// calculates the integrated upwards and downwards fluxes -- double precision version
__global__ void integrate_flux_double(
        double* deltalambda, 
        double* F_down_tot, 
        double* F_up_tot, 
        double* F_net, 
        double* F_down_wg, 
        double* F_up_wg,
        double* F_dir_wg,
        double* F_down_band, 
        double* F_up_band, 
        double* F_dir_band,
        double* gauss_weight,
        int 	nbin, 
        int 	numinterfaces, 
        int 	ny
){

    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = threadIdx.z;
    
    if(y==0){
        while(i < numinterfaces){
            while(x < nbin){
                
                F_up_tot[i] = 0;
                F_down_tot[i] = 0;
                
                F_dir_band[x + nbin * i] = 0;
                F_up_band[x + nbin * i] = 0;
                F_down_band[x + nbin * i] = 0;
                
                x += blockDim.x;
            }
            x = threadIdx.x;
            i += blockDim.z;	
        }
    }
    __syncthreads();
    
    i = threadIdx.z;
    
    while(i < numinterfaces){
        while(y < ny){
            while(x < nbin){
                
                atomicAdd_double(&(F_dir_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_dir_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_up_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_up_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_down_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_down_wg[y + ny * x + ny * nbin * i]);
                
                x += blockDim.x;
            }
            x = threadIdx.x;
            y += blockDim.y;
        }
        y = threadIdx.y;
        i += blockDim.z;
    }
    __syncthreads();
    
    i = threadIdx.z;
    
    if(y == 0){
        while(i < numinterfaces){
            while(x < nbin){
                
                atomicAdd_double(&(F_up_tot[i]), F_up_band[x + nbin * i] * deltalambda[x]);
                atomicAdd_double(&(F_down_tot[i]), (F_dir_band[x + nbin * i] + F_down_band[x + nbin * i]) * deltalambda[x]);

                x += blockDim.x;
            }
            x = threadIdx.x;
            i += blockDim.z;
        }
    }
    __syncthreads();
    
    i = threadIdx.z;
    
    if(x == 0 && y == 0){
        while(i < numinterfaces){
            F_net[i] = F_up_tot[i] - F_down_tot[i];
            i += blockDim.z;
        }
    }
}


// calculates the integrated upwards and downwards fluxes -- single precision version
__global__ void integrate_flux_single(
        float* deltalambda, 
        float* F_down_tot, 
        float* F_up_tot, 
        float* F_net, 
        float* F_down_wg, 
        float* F_up_wg,
        float* F_dir_wg,
        float* F_down_band, 
        float* F_up_band, 
        float* F_dir_band,
        float* gauss_weight,
        int 	nbin, 
        int 	numinterfaces, 
        int 	ny
){

    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            F_up_tot[i] = 0;
            F_down_tot[i] = 0;

            F_dir_band[x + nbin * i] = 0;
            F_up_band[x + nbin * i] = 0;
            F_down_band[x + nbin * i] = 0;

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            while(x < nbin && y < ny && i < numinterfaces){

                atomicAdd_single(&(F_dir_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_dir_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_single(&(F_up_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_up_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_single(&(F_down_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_down_wg[y + ny * x + ny * nbin * i]);
                
                x += blockDim.x;
            }
            x = threadIdx.x;
            y += blockDim.y;
        }
        y = threadIdx.y;
        i += blockDim.z;	
    }
    __syncthreads();
    
    i = threadIdx.z;

    while(x < nbin && y == 0 && i < numinterfaces){
        
        while(x < nbin && y == 0 && i < numinterfaces){

            atomicAdd_single(&(F_up_tot[i]), F_up_band[x + nbin * i] * deltalambda[x]);
            atomicAdd_single(&(F_down_tot[i]), (F_dir_band[x + nbin * i] + F_down_band[x + nbin * i]) * deltalambda[x]);

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y == 0 && i < numinterfaces){

        F_net[i] = F_up_tot[i] - F_down_tot[i];
        
        i += blockDim.z;
    }
}


// calculates the net fluxes and advances the layer temperatures
__global__ void rad_temp_iter(
        utype*  F_down_tot, 
        utype*  F_up_tot, 
        utype*  F_net, 
        utype*  F_net_diff, 
        utype*  tlay, 
        utype*  play,
        utype*  tint, 
        utype*  pint,
        int* 	abrt, 
        utype*  T_store, 
        utype*  deltat_prefactor,
        utype*  F_add_heat_lay,
        utype* 	F_add_heat_sum,
        utype*  F_smooth,
        utype*  F_smooth_sum,
        utype*  c_p_lay,
        utype*  meanmolmass_lay,
        int 	itervalue, 
        utype 	f_factor, 
        int 	foreplay,
        utype 	g,
        int 	numlayers, 
        utype 	physical_tstep, 
        utype 	local_limit, 
        int 	adapt_interval,
        int		smooth,
        int     dim,
        int     step,
        utype   F_intern,
        int     no_atmo
){

    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if(i < numlayers+1){
        
        utype combined_F_net_diff;
        utype delta_t;
        utype delta_T = 0;
        
        // calculating the net flux divergence for each atmospheric layer and the surface or "ghost layer" at BOA
        if(i < numlayers){
            
            // net flux divergence for each layer
            F_net_diff[i] = F_net[i] - F_net[i+1] + F_add_heat_lay[i]; // now with additional heating term (if additional heating is disabled, the F_add_heat_lay array is zero and nothing gets added)
            
            // tweaking points for smoothing
            utype t_mid = tlay[i];
            
            if(smooth ==1){
                
                if(play[i] < 1e6 && i < numlayers -1 && i > 0){
                    t_mid = (tlay[i-1]+tlay[i+1])/2.0;
                }
                
                // temperature smoothing force (or "smoothing flux") -- dependent on the temperature displacement (power of 7 found to be best "middle" between cheating and energy conservation)
                F_smooth[i] = pow((t_mid - tlay[i]), 7.0);
                
                __syncthreads(); // syncing threads needed because all F_smooth[i] are summed up below

                // summing up smoothing fluxes -- necessary to be included in the radiative equilibrium criterion
                F_smooth_sum[i] = 0;
                for(int j=0; j<=i; j++) F_smooth_sum[i] += F_smooth[j];
            }
            
            // net flux gradient -- combination of pure radiative net flux and temperature smoothing term
            combined_F_net_diff = F_net_diff[i] + F_smooth[i];
        }
        else{ 
            // i = numlayers, i.e., surface/BOA "ghost layer" case
            combined_F_net_diff = F_intern - F_net[0];
            
            // use net flux of one layer above ground when not converged yet to avoid stuck convergence (ground layer and one layer above may become stuck in an circular loop otherwise)
            if(abs(F_intern - F_net[1])/(F_down_tot[numlayers] + F_intern) > 0.5 * local_limit){ 
                combined_F_net_diff = F_intern - F_net[1];
            }
        }

        // setting the numerical time step and advancing the temperature numerically with adaptive method
        if (physical_tstep == 0){
            if (itervalue == foreplay){
                deltat_prefactor[i] = 1e0; // 1e0 found to be most stable. earlier value was 1e2.
            }
            if (itervalue == 10000){
                deltat_prefactor[i] = 1e-1; // warning: hardcoded number. reset convergence iterations if for some reason stuck
            }
            
            if(combined_F_net_diff != 0){
                delta_t = deltat_prefactor[i] * play[0] / pow(abs(combined_F_net_diff), 0.9); // through tweaking 0.9 found to be most stable
            }
            
            delta_T = combined_F_net_diff / (pint[0] - pint[1]) * delta_t;
            
            //         // uncomment for debugging info
            //         if(i == numlayers){
            //             if(itervalue % 100 == 0) printf("F_intern: %.6e, F_TOA: %.6e, F_net[0]: %.6e, F_net[1]: %.6e, F_net[2]: %.6e \n", F_intern, F_down_tot[numlayers], F_net[0], F_net[1], F_net[2]);
            //         }
            
            // limits large temperature jumps for increased stability
            if(abs(delta_T) > 500.0){
                delta_T = 500.0 * combined_F_net_diff/abs(combined_F_net_diff);
            }
            
            // stores temperature array from 6 or 20 iterations ago
            if (itervalue % adapt_interval == 0){
                T_store[i] = tlay[i];
            }
            
            // adaptively increasing/decreasing the numerical timestep 
            // i.e., if temperature progresses monotonically, the timestep prefactor is increased. otherwise, if delta T is going back-and-forth, the prefactor is decreased.
            if (itervalue % adapt_interval == adapt_interval-1) {
                if(abs(tlay[i] - T_store[i]) < adapt_interval / 2.0 * abs(delta_T)){
                    deltat_prefactor[i] /= 1.5; // 1.5 was found to lead to fastest convergence
                }
                else{
                    deltat_prefactor[i] *= 1.1; // 1.1 was found to lead to fastest convergence
                }
            }
        }
        // advancing the temperature using a constant physical timestep
        else{
            delta_t = physical_tstep;
            if(i < numlayers){
                delta_T = g / (c_p_lay[i] / (meanmolmass_lay[i]/AMU)) * combined_F_net_diff / (pint[i] - pint[i+1]) * delta_t;
            }
            else{ // i = numlayers, i.e., surface/BOA "ghost layer" case. Taking parameters of the bottommost atmospheric layer for simplicity. WARNING: This is obviously wrong when modeling a solid surface.
                delta_T = g / (c_p_lay[0] / (meanmolmass_lay[0]/AMU)) * combined_F_net_diff / (pint[0] - pint[1]) * delta_t;
            }
        }
        
        // update layer temperatures
        tlay[i] = tlay[i] + delta_T;
        
        // for no-atmosphere case the atmosphere does not exist and hence it temperature is zero-ish everywhere apart from surface
        if(no_atmo==1 && i!=numlayers){
            tlay[i] = 1.001;
        }
        
        // prevent too low temperatures and too high temperatures
        // the temperature is limited to the maximum value of pre-tabulated BB values
        utype max_limit = dim * step - 1.001;
        tlay[i] = min(max(tlay[i],1.001), max_limit); 
        
        bool condition;
        if(i < numlayers) condition = abs(F_intern + F_add_heat_sum[i] + F_smooth_sum[i] - F_net[i+1])/(F_down_tot[numlayers] + F_intern) < local_limit;
        if(i == numlayers) condition = abs(F_intern - F_net[0])/(F_down_tot[numlayers] + F_intern) < local_limit;
        
        //if(itervalue % 10 == 0 && i == 70) printf("layer: %d, criterion: %.4e, limit: %.4e \n", i, abs(F_intern + F_add_heat_sum[i] + F_smooth_sum[i] - F_net[i+1])/(F_down_tot[numlayers] + F_intern), local_limit); // uncomment for criterion feedback
        
        // if condition is satisfied this layer signals its readiness to abort the iteration loop
        if (condition){
            abrt[i] = 1;
        }
        else {
            abrt[i] = 0;
        }
    }
}


// advances the layer temperatures in the convection loop
__global__ void conv_temp_iter(
        utype*  F_down_tot, 
        utype*  F_up_tot, 
        utype*  F_net, 
        utype*  F_net_diff, 
        utype*  tlay, 
        utype*  play,
        utype*  pint,
        utype*  T_store, 
        utype*  deltat_prefactor,
        int*    marked_red,
        utype*  F_add_heat_lay,
        utype*  F_smooth,
        utype*  F_smooth_sum,
        int 	numlayers,
        int 	itervalue,
        int 	adapt_interval,
        int 	smooth,
        utype   F_intern
){
    
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    
    if(i < numlayers+1){
        
        //set constant timestep value
        utype delta_t;
        
        utype combined_F_net_diff;
        
        if(i < numlayers){
            
            // net flux divergence for each layer
            F_net_diff[i] = F_net[i] - F_net[i+1] + F_add_heat_lay[i]; // now with additional heating term (if add heating is disabled, the F_add_heat_lay array is zero and nothing gets added)
            
            // tweaking points (completely analogous to the radiative iteration tweaking)
            utype t_mid = tlay[i];
            
            if(smooth ==1){
                
                if(play[i] < 1e6 && i < numlayers -1){
                    t_mid = (tlay[i-1]+tlay[i+1])/2.0;
                }
                
                // temperature smoothing force (or "smoothing flux") -- dependent on the temperature displacement (power of 7 found to be best "middle" between cheating and energy conservation)
                F_smooth[i] = pow((t_mid - tlay[i]), 7.0);
                
                __syncthreads(); // syncing threads needed because all F_smooth[i] are summed up below
                
                // summing up smoothing fluxes -- necessary to be included in the radiative equilibrium criterion
                F_smooth_sum[i] = 0;
                for(int j=0; j<=i; j++) F_smooth_sum[i] += F_smooth[j];
            }
            
            // net flux gradient -- combination of pure radiative net flux and temperature smoothing term
            combined_F_net_diff =  F_net_diff[i] + F_smooth[i];
        }
        else{ // i = numlayers is the surface/BOA "ghost layer"
            combined_F_net_diff = F_intern - F_net[0];
            
            for(int j=0; j<numlayers;j++){
                if(marked_red[j]==1){
                    
                    // uncomment for debugging info
                    // if (itervalue % 100 == 0) printf("Taking layer: %d for surface delta T \n", j);
                    
                    combined_F_net_diff = F_intern - F_net[j+1]; // avoiding taking convective layers as net flux driver for surface temperature
                    break;
                }
            }
            
        }
        // set initial timestep prefactor
        if (itervalue == 0){
            deltat_prefactor[i] = 1e-2; // 1e-2 found to be most stable.
        }
        if (itervalue == 6000){
            deltat_prefactor[i] = 1e-3; // warning: hardcoded number. cheap fix to unstuck calculations after convective stitching
        }
        
        if(combined_F_net_diff != 0){
            delta_t = deltat_prefactor[i] * play[0] / pow(abs(combined_F_net_diff), 0.5); // 0.5 was found to be most stable for the radiative-convective interplay
        }
        
        utype delta_T =  combined_F_net_diff / (pint[0] - pint[1]) * delta_t;
        
//          // uncomment for debugging info
//         if(i == numlayers){
//             if(itervalue % 100 == 0) printf("F_intern: %.6e, F_TOA: %.6e, F_net[0]: %.6e, F_net[1]: %.6e, F_net[2]: %.6e \n", F_intern, F_down_tot[numlayers], F_net[0], F_net[1], F_net[2]);
//         }
        
        // limits large temperature jumps for increased stability
        if(abs(delta_T) > 20.0){
            delta_T = 20.0 * combined_F_net_diff/abs(combined_F_net_diff);
        }

        // store last few entries of temperature change
        if (itervalue % adapt_interval == 0){
            T_store[i] = tlay[i];
        }

        if (itervalue % adapt_interval == adapt_interval-1) {
            if(abs(tlay[i] - T_store[i]) < adapt_interval / 2.0 * abs(delta_T)){
                deltat_prefactor[i] /= 1.5; // 1.5 was found to lead to fastest convergence
            }
            else{
                deltat_prefactor[i] *= 1.1; // 1.1 was found to lead to fastest convergence
            }
        }

        // update layer temperatures
        tlay[i] = tlay[i] + delta_T;
        
        // prevent too low temperatures
        tlay[i] = max(tlay[i],1.001);
    }
}


// integrates the transmission function for each wavelength bin
__global__ void integrate_optdepth_transmission_iso(
        utype* trans_wg,
        utype* trans_band,
        utype* delta_tau_wg,
        utype* delta_tau_band,
        utype* gauss_weight,  
        int     nbin, 
        int     nlayer, 
        int     ny
){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int i = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < nbin && i < nlayer){

        delta_tau_band[x+nbin*i] = 0;
        trans_band[x+nbin*i] = 0;

        for(int y=0;y<ny;y++){
            delta_tau_band[x+nbin*i] += 0.5 * gauss_weight[y] * delta_tau_wg[y+ny*x + ny*nbin*i];
            trans_band[x+nbin*i] += 0.5 * gauss_weight[y] * trans_wg[y+ny*x + ny*nbin*i];
        }
    }
}


// integrates the transmission function for each wavelength bin
__global__ void integrate_optdepth_transmission_noniso(
        utype* trans_wg_upper,
        utype* trans_wg_lower,
        utype* trans_band,
        utype* delta_tau_wg_upper,
        utype* delta_tau_wg_lower,
        utype* delta_tau_band,
        utype* gauss_weight,
        utype* delta_tau_all_clouds,
        utype* delta_tau_all_clouds_upper,
        utype* delta_tau_all_clouds_lower,
        int     nbin, 
        int     nlayer, 
        int     ny
){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int i = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < nbin && i < nlayer){

        delta_tau_band[x+nbin*i] = 0;
        trans_band[x+nbin*i] = 0;

        for(int y=0;y<ny;y++){
            delta_tau_band[x+nbin*i] += 0.5 * gauss_weight[y] * (delta_tau_wg_upper[y+ny*x + ny*nbin*i] + delta_tau_wg_lower[y+ny*x + ny*nbin*i]);
            trans_band[x+nbin*i] += 0.5 * gauss_weight[y] * (trans_wg_upper[y+ny*x + ny*nbin*i] * trans_wg_lower[y+ny*x + ny*nbin*i]);
        }
        
        delta_tau_all_clouds[x+nbin*i] = delta_tau_all_clouds_lower[x+nbin*i] + delta_tau_all_clouds_upper[x+nbin*i];
    }
}


// calculates the contribution function
__global__ void calc_contr_func_iso(
        utype* trans_wg,
        utype* trans_weight_band,
        utype* contr_func_band,
        utype* gauss_weight,
        utype* planckband_lay,
        utype 	epsi,
        int     nbin, 
        int     nlayer, 
        int     ny
){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int i = threadIdx.y + blockIdx.y*blockDim.y;
    
    if (x < nbin && i < nlayer){
        
        utype trans_to_top;
        
        for(int y=0;y<ny;y++){

            trans_to_top = 1.0;
            
            for (int j = i+1; j < nlayer; j++){
                trans_to_top = trans_to_top * trans_wg[y+ny*x+ny*nbin*j];
            }

            trans_weight_band[x+nbin*i] += 0.5 * gauss_weight[y] * (1.0 - trans_wg[y+ny*x+ny*nbin*i]) * trans_to_top;
        }
        
        contr_func_band[x+nbin*i] = 2.0 * PI * epsi * planckband_lay[i+x*(nlayer+2)] * trans_weight_band[x+nbin*i];
    }
}


// calculates the contribution function, non-isothermal version
__global__ void calc_contr_func_noniso(
        utype* trans_wg_upper,
        utype* trans_wg_lower,
        utype* trans_weight_band,
        utype* contr_func_band,
        utype* gauss_weight,
        utype* planckband_lay,
        utype 	epsi,
        int     nbin, 
        int     nlayer, 
        int     ny
){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int i = threadIdx.y + blockIdx.y*blockDim.y;
    
    if (x < nbin && i < nlayer){
        
        utype trans_to_top;
        
        for(int y=0;y<ny;y++){

            trans_to_top = 1.0;
            
            for (int j = i+1; j < nlayer; j++){
                trans_to_top = trans_to_top * trans_wg_upper[y+ny*x+ny*nbin*j] * trans_wg_lower[y+ny*x+ny*nbin*j];
            }

            trans_weight_band[x+nbin*i] += 0.5 * gauss_weight[y] * (1.0 - trans_wg_upper[y+ny*x+ny*nbin*i] * trans_wg_lower[y+ny*x+ny*nbin*i]) * trans_to_top;
        }
        
        contr_func_band[x+nbin*i] = 2.0 * PI * epsi * planckband_lay[i+x*(nlayer+2)] * trans_weight_band[x+nbin*i];
    }
}


// calculates the Planck and Rosseland mean opacities for each layer
__global__ void calc_mean_opacities(
        utype* planck_opac_T_pl, 
        utype* ross_opac_T_pl,
        utype* planck_opac_T_star, 
        utype* ross_opac_T_star, 
        utype* opac_wg_lay, 
        utype* abs_cross_all_clouds_lay,
        utype* meanmolmass_lay,
        utype* planckband_lay, 
        utype* opac_interwave, 
        utype* opac_deltawave, 
        utype* T_lay, 
        utype* gauss_weight, 
        utype* gauss_y,
        utype* opac_band_lay, 
        int 	nlayer, 
        int 	nbin, 
        int 	ny, 
        utype 	T_star
){

    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if(i < nlayer){

        utype num_planck_T_pl = 0;
        utype denom_planck_T_pl = 0;
        utype num_ross_T_pl = 0;
        utype denom_ross_T_pl = 0;
        utype num_planck_T_star = 0;
        utype denom_planck_T_star = 0;
        utype num_ross_T_star = 0;
        utype denom_ross_T_star = 0;
        
        // integrates opacity over each bin with Gaussian quadrature
        for(int x=0;x<nbin;x++){

            opac_band_lay[x+nbin*i] = 0;

            for (int y=0;y<ny;y++){
                opac_band_lay[x+nbin*i] += 0.5 * gauss_weight[y] * opac_wg_lay[y+ny*x+ny*nbin*i];
            }
        }

        for (int x = 0; x < nbin; x++) {

            // calculates Planck mean opacity with layer temperatures
            num_planck_T_pl += (opac_band_lay[x+nbin*i] + abs_cross_all_clouds_lay[x+nbin*i]/meanmolmass_lay[i]) * planckband_lay[i+x*(nlayer+2)]*opac_deltawave[x];
            denom_planck_T_pl += planckband_lay[i+x*(nlayer+2)]*opac_deltawave[x];

            // calculates Rosseland mean opacity with layer temperatures
            num_ross_T_pl += integrated_dB_dT(gauss_weight, gauss_y, ny, opac_interwave[x], opac_interwave[x+1], T_lay[i]);
            
            if ((opac_band_lay[x+nbin*i] + abs_cross_all_clouds_lay[x+nbin*i]/meanmolmass_lay[i]) > 0) {
                
                denom_ross_T_pl += integrated_dB_dT(gauss_weight, gauss_y, ny, opac_interwave[x], opac_interwave[x+1],T_lay[i]) / (opac_band_lay[x+nbin*i] + abs_cross_all_clouds_lay[x+nbin*i]/meanmolmass_lay[i]);
            
            }
            
            // calculates Planck mean opacity with stellar blackbody function
            num_planck_T_star += (opac_band_lay[x+nbin*i] + abs_cross_all_clouds_lay[x+nbin*i]/meanmolmass_lay[i]) * planckband_lay[nlayer+x*(nlayer+2)]*opac_deltawave[x];
            denom_planck_T_star += planckband_lay[nlayer+x*(nlayer+2)]*opac_deltawave[x];
            
            // calculates Rosseland mean opacity with stellar blackbody function
            num_ross_T_star += integrated_dB_dT(gauss_weight, gauss_y, ny, opac_interwave[x], opac_interwave[x+1], T_star);
            
            if ((opac_band_lay[x+nbin*i] + abs_cross_all_clouds_lay[x+nbin*i]/meanmolmass_lay[i]) > 0) {
                
                denom_ross_T_star += integrated_dB_dT(gauss_weight, gauss_y, ny, opac_interwave[x], opac_interwave[x+1], T_star) / (opac_band_lay[x+nbin*i] + abs_cross_all_clouds_lay[x+nbin*i]/meanmolmass_lay[i]);
            
            }
        }
        
        planck_opac_T_pl[i] = num_planck_T_pl / denom_planck_T_pl;
        ross_opac_T_pl[i] = num_ross_T_pl / denom_ross_T_pl;
                
                // for T < 70, K dB_dT is too small to be calculated numerically at short wavelengths
                if(T_lay[i] < 70){
                    ross_opac_T_pl[i] = -3;      
        }
        
        planck_opac_T_star[i] = num_planck_T_star / denom_planck_T_star;
        ross_opac_T_star[i] = num_ross_T_star / denom_ross_T_star;
        
        // same as above for star with 70 K. Does not make any sense anyway, such a cool star, 
        // hence also planck calculation prohibited.
        if (T_star < 70){
            planck_opac_T_star[i] = -3;
            ross_opac_T_star[i] = -3;         
        }
    }
}


// integrate the direct beam flux
__global__ void integrate_beamflux(
        utype* F_dir_tot, 
        utype* F_dir_band,
        utype* deltalambda, 
        utype* gauss_weight, 
        int 	nbin, 
        int 	numinterfaces
){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < numinterfaces){

        F_dir_tot[i] = 0;

        for (int x = 0; x < nbin; x++) {

            // sum the bin contributions to obtain the integrated flux
            F_dir_tot[i] += F_dir_band[x + nbin * i] * deltalambda[x];
        }
    }
}


// helper function for array sorting. it swaps two elements of an array.
__device__ void swap(utype* a, utype* b)  
{  
    utype t = *a;  
    *a = *b;  
    *b = t;  
}  


// function to sort an array. similar to the "insertion sort" method.
__device__ void sort_array(utype* opac_arr, utype* weight_arr){
    
    utype j = 0;
    
    while (j < 399){
        
        j = 0;
        
        for (int i = 1; i < 400; i++)
            
            if (opac_arr[i] < opac_arr[i-1]){
                
                swap(&opac_arr[i], &opac_arr[i-1]);
                swap(&weight_arr[i], &weight_arr[i-1]);
            }
            else{
                j++;
            }
    }
}


__device__ utype calc_index_h2o(
    utype wave, 
    utype press, 
    utype temp, 
    utype f_h2o,
    utype mass_h2o
){
    
    utype dens = f_h2o * press * mass_h2o / (KBOLTZMANN * temp);

    utype lamda = wave / 0.589e-4;
    utype delta = max(1.0, dens) / 1.0; // limiting to valid range, otherwise index may return NaN (this only affects atmospheric regions with at least around 1000 bar)
    utype theta = temp / 273.15;

    utype lamda_UV = 0.229202;
    utype lamda_IR = 5.432937;

    utype a0 = 0.244257733;
    utype a1 = 0.974634476e-2;
    utype a2 = -0.373234996e-2;
    utype a3 = 0.268678472e-3;
    utype a4 = 0.158920570e-2;
    utype a5 = 0.245934259e-2;
    utype a6 = 0.900704920;
    utype a7 = -0.166626219e-1;

    utype capital_a = delta * (a0 + a1*delta + a2*theta + a3*pow(1.0*lamda, 2.0)*theta + a4*pow(1.0*lamda, -2.0) + a5 / (pow(1.0*lamda, 2.0) - pow(1.0*lamda_UV, 2.0)) + a6 / (pow(1.0*lamda, 2.0) - pow(1.0*lamda_IR, 2.0)) + a7*pow(1.0*delta, 2.0));
    
    utype index = pow((2.0 * capital_a + 1.0)/(1.0 - capital_a), 0.5);
    
    return index;
}


// interpolates layer and interface opacities from opacity table
__global__ void opac_species_interpol(
		utype*  temp, 
		utype*  opactemp, 
		utype*  press, 
		utype*  opacpress,
		utype*  opac_opacity_pretab,
		utype*  opac_spec_wg_lay_or_int,
		int 	npress, 
		int 	ntemp, 
		int 	ny,
		int 	nbin,
		int 	nlay_or_nint
){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nbin && i < nlay_or_nint) {

		utype deltaopactemp = (opactemp[ntemp-1] - opactemp[0])/(ntemp-1.0);
		utype deltaopacpress = (log10(opacpress[npress -1]) - log10(opacpress[0])) / (npress-1.0);
		utype t = (temp[i] - opactemp[0]) / deltaopactemp;
		utype p = (log10(press[i]) - log10(opacpress[0])) / deltaopacpress;

		t = min(ntemp-1.0, max(0.0, t));
		
		int tdown = floor(t);
		int tup = ceil(t);

		p = min(npress-1.0, max(0.0, p));
		
		int pdown = floor(p);
		int pup = ceil(p);


		for(int y=0;y<ny;y++){

			opac_spec_wg_lay_or_int[y+ny*x + ny*nbin*i] = bilin_interpol_func(opac_opacity_pretab[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
                                                                              opac_opacity_pretab[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
                                                                              opac_opacity_pretab[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
                                                                              opac_opacity_pretab[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
                                                                              p, 
                                                                              t, 
                                                                              pdown, 
                                                                              pup, 
                                                                              tdown, 
                                                                              tup);
            
		}
	}
}


// adds the individual molecular opacity to the mixed opacity
__global__ void add_to_mixed_opac(
    utype*  vmr_lay_or_int,
    utype*  opac_spec_lay_or_int,
    utype*  opac_wg_lay_or_int,
    utype*  meanmolmass_lay_or_int,
    utype*  gauss_weight,
    utype*  gauss_y,
    utype   mass_spec,
    int     s,
    int     ro_method,
    int     ny,
    int     nbin,
    int     nlay_or_nint
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < nbin && i < nlay_or_nint){
        
        bool opac_mixing_negligible = false;
        bool condition_for_correlated_k;
        
        // pre-storing opacity arrays into local memory, because they are used so many times
        utype new_opac_wg[20];
        utype mixed_opac_wg[20];
        
        for(int y=0;y<ny;y++){
            
            mixed_opac_wg[y] = opac_wg_lay_or_int[y+ny*x+ny*nbin*i];
            new_opac_wg[y] = vmr_lay_or_int[i] * mass_spec/meanmolmass_lay_or_int[i] * opac_spec_lay_or_int[y+ny*x+ny*nbin*i];
        }
        
        // if the maximum of the newly added opacity is less than 1% of the minimum of the opacity we have already, there is really no point in doing RO. Hence we go for correlated-k.
        if ((0.01 * mixed_opac_wg[0] > new_opac_wg[ny-1]) || (0.01 * new_opac_wg[0] > mixed_opac_wg[ny-1])){
            
            opac_mixing_negligible = true;
        }
        
        condition_for_correlated_k = (ro_method == 0) || (s == 0) || opac_mixing_negligible || (ny == 1);
            
        if (condition_for_correlated_k){ // use correlated-k method
            
            for(int y=0;y<ny;y++){
                
                opac_wg_lay_or_int[y+ny*x+ny*nbin*i] += new_opac_wg[y];
            }
        }
        else{ // use random overlap method
            
            // calculate unsorted RO sum of opacities and corresponding Gaussian weights
            // currently only works with 20 gaussian points (400=20x20). TODO it would be nice to have this a flexible number at some point.
            utype opac_wg_square_tot[400];
            utype gaussian_weights_square_tot[400];
            utype yg_square_tot[400];
            
            // determine intersection between the two opacity curves (if there is any). y_intersect is first y index after(!) the intersection
            
            int y_intersect = ny; // if there is no intersection, the value y_intersect is such that the whole situation looks like before the intersection
            
            for (int y = 1; y < ny; y++){
                
                if((mixed_opac_wg[y] > new_opac_wg[y]) != (mixed_opac_wg[y-1] > new_opac_wg[y-1])){ 
                    
                    y_intersect = y;
                }
            }
            
            //if mixed opacity is stronger in the bottom part, it is on the outer loop there
            if (mixed_opac_wg[0] > new_opac_wg[0]){
                
                for(int y1 = 0; y1 < ny; y1++){
                    for(int y2 = 0; y2 < y_intersect; y2++){
                        
                        opac_wg_square_tot[y2 + y_intersect * y1] = mixed_opac_wg[y1] + new_opac_wg[y2];
                        gaussian_weights_square_tot[y2 + y_intersect * y1] = (0.5 * gauss_weight[y1]) * (0.5 * gauss_weight[y2]);
                    }
                }
                for(int y2 = y_intersect; y2 < ny; y2++){
                    for(int y1 = 0; y1 < ny; y1++){
                 
                        opac_wg_square_tot[y1 + ny * y2] = mixed_opac_wg[y1] + new_opac_wg[y2];
                        gaussian_weights_square_tot[y1 + ny * y2] = (0.5 * gauss_weight[y1]) * (0.5 * gauss_weight[y2]);
                    }
                }
            }
            else{ // if new opacity is stronger in the bottom part, it is on the outer loop there
                
                for(int y2 = 0; y2 < ny; y2++){
                    for(int y1 = 0; y1 < y_intersect; y1++){
                        
                        opac_wg_square_tot[y1 + y_intersect * y2] = mixed_opac_wg[y1] + new_opac_wg[y2];
                        gaussian_weights_square_tot[y1 + y_intersect * y2] = (0.5 * gauss_weight[y1]) * (0.5 * gauss_weight[y2]);
                    }
                }
                for(int y1 = y_intersect; y1 < ny; y1++){
                    for(int y2 = 0; y2 < ny; y2++){
                 
                        opac_wg_square_tot[y2 + ny * y1] = mixed_opac_wg[y1] + new_opac_wg[y2];
                        gaussian_weights_square_tot[y2 + ny * y1] = (0.5 * gauss_weight[y1]) * (0.5 * gauss_weight[y2]);
                    }
                }
            }
            
            // sort the RO sum of opacities and permutate the Gaussian weights accordingly
            sort_array(opac_wg_square_tot, gaussian_weights_square_tot);
            
            // calc. y-points (abscissa points) for the resorted k-values
            yg_square_tot[0] = 0.5 * gaussian_weights_square_tot[0];
            
            for (int w = 1; w < 400; w++){
                
                yg_square_tot[w] = yg_square_tot[w-1] + 0.5 * gaussian_weights_square_tot[w-1] + 0.5 * gaussian_weights_square_tot[w];
            }
            
            // rebinning of sorted k-function
            int y = 0;
            
            for(int w=1; w < 400; w++){
                
                // check that no y-point has been skipped
                if(yg_square_tot[w-1] > gauss_y[y]){
                    printf("ERROR ERROR ERROR: Rebinning algorithm in k-table RO method is malfunctioning. Please double-check source code!!! \n");
                    printf("yg_square_tot[%d]: %.3e, gauss_y[%d]: %.3e \n", w-1, yg_square_tot[w-1], y, gauss_y[y]);
                }
                
                if(yg_square_tot[w] > gauss_y[y]){
                    
                    opac_wg_lay_or_int[y+ny*x+ny*nbin*i] = (opac_wg_square_tot[w-1] * (yg_square_tot[w] - gauss_y[y]) + opac_wg_square_tot[w] * (gauss_y[y] - yg_square_tot[w-1])) / (yg_square_tot[w] - yg_square_tot[w-1]);
                    
                    if (y < 19) y++;
                    else break; // loop should stop once all y-points are through
                }
            }
        }
    }
}


// calculate h2o Rayleigh scattering cross section
// references: Murphy (1977), Schiebener et al. (1990), Wagner & Kretzschmar (2008) 
__global__ void calc_h2o_scat(
    utype*  temp,
    utype*  press,
    utype*  wave,
    utype*  scat_cross_lay_or_int,
    utype*  vmr_lay_or_int,
    utype   mass_h2o,
    int     nbin,
    int     nlay_or_nint
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < nbin && i < nlay_or_nint){
        
        utype scat_cross;
        
        utype index = calc_index_h2o(wave[x], press[i], temp[i], vmr_lay_or_int[i], mass_h2o);
        
        utype n_ref = vmr_lay_or_int[i] * press[i] / (KBOLTZMANN * temp[i]);
        
        utype King = (6.0 + 3.0 * 3e-4) / (6.0 - 7.0 * 3e-4);
        
        utype lamda_limit = 2.5e-4; // scattering expression only valid below this limit
        
        if(wave[x] < lamda_limit){
            
            scat_cross = 24.0 * pow(1.0*PI, 3.0) / (pow(1.0*n_ref, 2.0) * pow(1.0*wave[x], 4.0)) * pow((pow(1.0*index, 2.0) - 1.0) / (pow(1.0*index, 2.0) + 2.0), 2.0) * King;
        }
        else{
            scat_cross = 0.0;
        }
        
        scat_cross_lay_or_int[x + nbin * i] = scat_cross;
    }
}


// combine the individual molecular opacities to layer/interface opacities
__global__ void add_to_mixed_scat(
    utype*  vmr_lay_or_int,
    utype*  scat_cross_spec_lay_or_int,
    utype*  scat_cross_lay_or_int,
    int     nbin,
    int     nlay_or_nint
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < nbin && i < nlay_or_nint){
        
        scat_cross_lay_or_int[x+nbin*i] += vmr_lay_or_int[i] * scat_cross_spec_lay_or_int[x+nbin*i];
    }
}
