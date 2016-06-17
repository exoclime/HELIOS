// =================================================================================
// This file contains all the device functions and CUDA kernels.
// Copyright (C) 2016 Matej Malik
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


// physical constant:
const double pi = 3.14159265359;
const double hconst = 6.62606957e-27;
const double cspeed = 29979245800;
const double kBoltzmann = 1.3806488e-16;
const double stefanboltzmann = 5.6704e-5;


// calculates analytically the integral of the planck function
__device__ double analyt_planck(int n, double y1, double y2){

	double dn=n;

	return exp(-dn*y2) * (pow(y2,3.0)/dn + 3.0*pow(y2,2.0)/pow(dn,2.0) + 6.0*y2/pow(dn,3.0) + 6.0/pow(dn,4.0))
			- exp(-dn*y1) * (pow(y1,3.0)/dn + 3.0*pow(y1,2.0)/pow(dn,2.0) + 6.0*y1/pow(dn,3.0) + 6.0/pow(dn,4.0));
}


//  calculates the transmission function
__device__ double trans_func(double epsi, double delta_tau, double w0, double g0){

    return exp(-1.0/epsi*pow((1.0-w0*g0)*(1.0-w0),0.5)*delta_tau);
}


// calculates the single scattering albedo w0
__device__ double single_scat_alb(double cross_scat, double opac_abs, double meanmolmass){

	return cross_scat / (cross_scat + opac_abs*meanmolmass);
}


// calculates the two-stream coupling coefficient Zeta_minus
__device__ double	zeta_minus(double w0, double g0){

	return 0.5 * (1.0 - pow((1.0-w0)/(1.0-w0*g0),0.5) );
}


// calculates the two-stream coupling coefficient Zeta_plus
__device__ double	zeta_plus(double w0, double g0){

	return 0.5 * (1.0 + pow((1.0-w0)/(1.0-w0*g0),0.5) );
}


// calculates the derivative of the Planck function regarding temperature
__device__ double dB_dT(double lambda, double T){

	double D = 2.0 * pow(hconst, 2.0) * pow(cspeed, 3.0) / (pow(lambda, 6.0) * kBoltzmann * pow(T, 2.0));

    double num =  exp(hconst * cspeed / (lambda * kBoltzmann * T));

    double denom = pow(exp( hconst * cspeed / (lambda * kBoltzmann * T)) - 1.0, 2.0);

    double result = D * num / denom ;

	return result;
}


// calculates the integral of the Planck derivative over a wavelength interval
__device__ double integrated_dB_dT(double* kw, double* ky, int ny, double lambda_bot, double lambda_top, double T){

	double result = 0;

	for (int y=0;y<ny;y++){
		double x = (ky[y]-0.5)*2.0;
		double arg = (lambda_top-lambda_bot)/2.0 * x + (lambda_top+lambda_bot)/2.0;
		result += (lambda_top-lambda_bot)/2.0 * kw[y]* dB_dT(arg,T);
	}
	return result;
}


// calculates the exponential integral of 1st kind
__device__ double expint1(double x){

	double a[] = {-0.57721566,0.99999193,-0.24991055,0.05519968,-0.00976004,0.00107857};
	double b[] = {1,8.5733287401,18.059016973,8.6347608925,0.2677737343};
	double c[] = {1,9.5733223454,25.6329561486,21.0996530827,3.9584969228};

	double result;

	if(x < 1){
		result = -log(x);
		for(int j=0;j<6;j++){
			result += a[j] * pow(x,j*1.0);
		}
	}
	else{
		double num=0;
		double denom=0;
		for(int j=0;j<5;j++){
			num += b[j] * pow(x,4.0-j);
			denom += c[j] * pow(x,4.0-j);
			result = 1/x*exp(-x)*num/denom;
		}
	}
	return result;
}


// constructing a table with Planck function values for given wavelengths and in a suitable temperature range
__global__ void plancktable(double* planck_grid, double* lambda_edge, double* deltalambda,
		int nwave, double Tstar, double Tint){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int t = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nwave && t < 402) {

		double T;
		double shifty;
		double D;
		double y_bot;
		double y_top;

			// building temperature grid from 1 to 4000 K and Tstar and Tint a 10 K resolution
			T = (t+1) * 10;

			if(t == 400){
				T = Tstar;
			}

			if(t == 401){

				T = Tint;
			}

			planck_grid[x + t * nwave] = 0.0;

			// analytical calculation, only for T > 0
			if(T > 0.01){
				D = 2.0 * pow(kBoltzmann * T,4.0)/(pow(hconst,3.0)*pow(cspeed,2.0));
				y_top = hconst * cspeed / (lambda_edge[x+1] * kBoltzmann * T);
				y_bot = hconst * cspeed / (lambda_edge[x] * kBoltzmann * T);

				// rearranging so that y_top < y_bot (i.e. wavelengths are always increasing)
				if(y_bot < y_top){
					shifty = y_top;
					y_top = y_bot;
					y_bot = shifty;
				}

				for(int n=1;n<100;n++){
					planck_grid[x + t * nwave] += D * analyt_planck(n, y_bot, y_top);
				}
			}

			planck_grid[x + t * nwave] /= deltalambda[x];

	}
}


// constructing the atmospheric grid with pressure & column mass
__global__ void gridkernel(double* play, double* pint, double* tlay, double* deltacolumnm,
		double* deltacolupper, double* deltacollower, double P_Boa, double P_Toa, int numlayers, double g) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < numlayers) {

		play[i] = P_Boa * exp(log(P_Toa / P_Boa) * i / (numlayers - 1.0));
		pint[i] = play[i] * pow(P_Boa / P_Toa, 1.0 / (2.0 * (numlayers - 1)));
		if (i == numlayers - 1) {
			pint[i + 1] = play[i] * pow(P_Toa / P_Boa, 1.0 / (2.0 * (numlayers - 1)));
		}
	}
	__syncthreads();

	if (i < numlayers) {
		deltacolumnm[i] = (pint[i] - pint[i + 1]) / g;
		deltacolupper[i] = (play[i] - pint[i + 1]) / g;
		deltacollower[i] = (pint[i] - play[i]) / g;
	}
}


// construction of a table for the capital letter terms in opacity, wavelength and layer
__global__ void captable(double* Mterm_grid, double* Nterm_grid, double* Pterm_grid, double* Qterm_grid,
		double* deltacolmass, double* cross_scat, double epsi, int numlayers, int nbin, int scat,
		double meanmolmass, double g0){

	int o = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int i = threadIdx.z + blockIdx.z * blockDim.z;

	if (o < 190 && x < nbin && i < numlayers) {

		// logarithmic grid in opacity (log delta_opac = 0.1) and taking the grid's column mass and wavelength
		double opac=pow(10.0,0.1*o-15);
		double w0;
		double trans;
		double zeta_min;
		double zeta_pl;

		if (scat == 1){
			w0 = single_scat_alb(cross_scat[x], opac, meanmolmass);
			trans = trans_func(epsi, deltacolmass[i] * (opac + cross_scat[x]/meanmolmass) , w0, g0);
			zeta_min=zeta_minus(w0,g0);
			zeta_pl=zeta_plus(w0,g0);
		}
		else{
			trans = trans_func(epsi, deltacolmass[i] * opac , 0, 0);
			zeta_min=0.0;
			zeta_pl=1.0;
		}

		Mterm_grid[o + 190 * x + 190 * nbin * i] = pow( zeta_min ,2.0) * pow(trans, 2.0) - pow( zeta_pl ,2.0);
		Pterm_grid[o + 190 * x + 190 * nbin * i] = (pow( zeta_min ,2.0) - pow( zeta_pl ,2.0)) * trans;
		Nterm_grid[o + 190 * x + 190 * nbin * i] = zeta_pl * zeta_min * (1.0 - pow(trans,2.0));
		Qterm_grid[o + 190 * x + 190 * nbin * i] = (pow( zeta_min ,2.0) * trans + pow( zeta_pl ,2.0)) * (1.0 - trans);

	}
}


// temperature interpolation for the non-isothermal layers
__global__ void temp_inter(double* tlay, double* tint, int numinterfaces) {

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
}


// interpolate layer and interface opacities from opacity table
__global__ void opac_interpol(double* temp, double* opactemp, double* press, double* opacpress,
		double* ktable, double* opac, double* ky, int npress, int ntemp, int ny,
		int nbin, double opaclimit, int nlay_or_nint) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nbin && i < nlay_or_nint) {

		int x_1micron = lrint(nbin * 2.0 / 3.0);

		double deltaopactemp = (opactemp[ntemp-1] - opactemp[0])/(ntemp-1.0);
		double deltaopacpress = (log10(opacpress[npress -1]) - log10(opacpress[0])) / (npress-1.0);
		double t = (temp[i] - opactemp[0]) / deltaopactemp;

		if (t > ntemp-1) {
			t = ntemp-1.0001;
		}
		if (t < 0) {
			t = 0.0001;
		}
		int tdown = floor(t);
		int tup = ceil(t);

		double p = (log10(press[i]) - log10(opacpress[0])) / deltaopacpress;

		if (p > npress-1) {
			p = npress-1.0001;
		}
		if (p < 0) {
			p = 0.0001;
		}
		int pdown = floor(p);
		int pup = ceil(p);

		if(pdown != pup && tdown != tup){
			for(int y=0;y<ny;y++){
				double interpolated_value =
					ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown] * (pup - p) * (tup - t)
					+ ktable[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown] * (p - pdown) * (tup - t)
					+ ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup] * (pup - p) * (t -  tdown)
					+ ktable[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup] * (p - pdown) * (t - tdown);

				if (x < x_1micron) {
					opac[y+ny*x + ny*nbin*i] = max(interpolated_value, opaclimit);
				}
				else {
					opac[y+ny*x + ny*nbin*i] = interpolated_value;
				}
			}
		}

		if(tdown == tup && pdown != pup){
			for(int y=0;y<ny;y++){
				double interpolated_value = ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown] * (pup - p)
										    + ktable[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown] * (p - pdown);
				if (x < x_1micron) {
					opac[y+ny*x + ny*nbin*i] = max(interpolated_value, opaclimit);
				}
				else {
					opac[y+ny*x + ny*nbin*i] = interpolated_value;
				}
			}
		}

		if(pdown == pup && tdown != tup){
			for(int y=0;y<ny;y++){
				double interpolated_value = ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown] * (tup - t)
										    + ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup] * (t -  tdown);
				if (x < x_1micron) {
					opac[y+ny*x + ny*nbin*i] = max(interpolated_value, opaclimit);
				}
				else {
					opac[y+ny*x + ny*nbin*i] = interpolated_value;
				}
			}
		}

		if(tdown == tup && pdown == pup){
			for(int y=0;y<ny;y++){

				double interpolated_value = ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown];

				if (x < x_1micron) {
					opac[y+ny*x + ny*nbin*i] = max(interpolated_value, opaclimit);
				}
				else {
					opac[y+ny*x + ny*nbin*i] = interpolated_value;
				}
			}
		}
	}
}


// interpolate capital letter to grid (isothermal layers) from capital letter table
__global__ void cap_interpol_iso(double* Mterm_grid, double* Nterm_grid, double* Pterm_grid, double* Qterm_grid,
		double* Mterm, double* Nterm, double* Pterm, double* Qterm, double* kopac_lay,
		int numlayers, int nbin, int ny){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nbin && i < numlayers){

		for(int y=0;y<ny;y++){

			double opac = kopac_lay[y+ny*x + ny*nbin*i];

				double o = (log10(opac) - log10(1e-15)) / 0.1;

				if (o > 189) {
					o = 190-1.0001;
				}
				if (o < 0) {
					o = 0.0001;
				}
				int odown = floor(o);
				int oup = ceil(o);

				if(odown != oup){
					Mterm[y+ny*x+ny*nbin*i] = Mterm_grid[odown + 190 * x + 190 * nbin * i] * (oup - o)
										+ Mterm_grid[oup + 190 * x + 190 * nbin * i] * (o - odown);

					Nterm[y+ny*x+ny*nbin*i] = Nterm_grid[odown + 190 * x + 190 * nbin * i] * (oup - o)
										+ Nterm_grid[oup + 190 * x + 190 * nbin * i] * (o - odown);

					Pterm[y+ny*x+ny*nbin*i] = Pterm_grid[odown + 190 * x + 190 * nbin * i] * (oup - o)
										+ Pterm_grid[oup + 190 * x + 190 * nbin * i] * (o - odown);

					Qterm[y+ny*x+ny*nbin*i] = Qterm_grid[odown + 190 * x + 190 * nbin * i] * (oup - o)
										+ Qterm_grid[oup + 190 * x + 190 * nbin * i] * (o - odown);
				}

				if(odown == oup){
					Mterm[y+ny*x+ny*nbin*i] = Mterm_grid[odown + 190 * x + 190 * nbin * i];

					Nterm[y+ny*x+ny*nbin*i] = Nterm_grid[odown + 190 * x + 190 * nbin * i];

					Pterm[y+ny*x+ny*nbin*i] = Pterm_grid[odown + 190 * x + 190 * nbin * i];

					Qterm[y+ny*x+ny*nbin*i] = Qterm_grid[odown + 190 * x + 190 * nbin * i];
				}
		}
	}
}


// interpolate capital letter to grid (non-isothermal layers) from capital letter table
__global__ void cap_interpol_noniso(double* Mterm_uppergrid, double* Nterm_uppergrid, double* Pterm_uppergrid,
		double* Qterm_uppergrid, double* Mterm_lowergrid, double* Nterm_lowergrid, double* Pterm_lowergrid,
		double* Qterm_lowergrid, double* Mterm_upper, double* Nterm_upper, double* Pterm_upper, double* Qterm_upper,
		double* Mterm_lower, double* Nterm_lower, double* Pterm_lower, double* Qterm_lower,
		double* kopac_lay, double* kopac_int, double* cross_scat, int numlayers, int nbin, int ny){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x < nbin && i < numlayers){

		for(int y=0;y<ny;y++){

			double opac[] = {
					(kopac_lay[y+ny*x + ny*nbin*i]+kopac_int[y+ny*x + ny*nbin*(i+1)])/2.0,
					(kopac_int[y+ny*x + ny*nbin*i]+kopac_lay[y+ny*x + ny*nbin*i])/2.0
			};

			double o = (log10(opac[z]) - log10(1e-15)) / 0.1;

			if (o > 189) {
				o = 190-1.0001;
			}
			if (o < 0) {
				o = 0.0001;
			}
			int odown = floor(o);
			int oup = ceil(o);

			if(z==0){

				if(odown != oup){
					Mterm_upper[y+ny*x+ny*nbin*i] = Mterm_uppergrid[odown + 190 * x + 190 * nbin * i] * (oup - o)
										+ Mterm_uppergrid[oup + 190 * x + 190 * nbin * i] * (o - odown);

					Nterm_upper[y+ny*x+ny*nbin*i] = Nterm_uppergrid[odown + 190 * x + 190 * nbin * i] * (oup - o)
										+ Nterm_uppergrid[oup + 190 * x + 190 * nbin * i] * (o - odown);

					Pterm_upper[y+ny*x+ny*nbin*i] = Pterm_uppergrid[odown + 190 * x + 190 * nbin * i] * (oup - o)
										+ Pterm_uppergrid[oup + 190 * x + 190 * nbin * i] * (o - odown);

					Qterm_upper[y+ny*x+ny*nbin*i] = Qterm_uppergrid[odown + 190 * x + 190 * nbin * i] * (oup - o)
										+ Qterm_uppergrid[oup + 190 * x + 190 * nbin * i] * (o - odown);
				}

				if(odown == oup){
					Mterm_upper[y+ny*x+ny*nbin*i] = Mterm_uppergrid[odown + 190 * x + 190 * nbin * i];

					Nterm_upper[y+ny*x+ny*nbin*i] = Nterm_uppergrid[odown + 190 * x + 190 * nbin * i];

					Pterm_upper[y+ny*x+ny*nbin*i] = Pterm_uppergrid[odown + 190 * x + 190 * nbin * i];

					Qterm_upper[y+ny*x+ny*nbin*i] = Qterm_uppergrid[odown + 190 * x + 190 * nbin * i];
				}
			}

			if(z==1){

				if(odown != oup){
					Mterm_lower[y+ny*x+ny*nbin*i] = Mterm_lowergrid[odown + 190 * x + 190 * nbin * i] * (oup - o)
													+ Mterm_lowergrid[oup + 190 * x + 190 * nbin * i] * (o - odown);

					Nterm_lower[y+ny*x+ny*nbin*i] = Nterm_lowergrid[odown + 190 * x + 190 * nbin * i] * (oup - o)
													+ Nterm_lowergrid[oup + 190 * x + 190 * nbin * i] * (o - odown);

					Pterm_lower[y+ny*x+ny*nbin*i] = Pterm_lowergrid[odown + 190 * x + 190 * nbin * i] * (oup - o)
													+ Pterm_lowergrid[oup + 190 * x + 190 * nbin * i] * (o - odown);

					Qterm_lower[y+ny*x+ny*nbin*i] = Qterm_lowergrid[odown + 190 * x + 190 * nbin * i] * (oup - o)
													+ Qterm_lowergrid[oup + 190 * x + 190 * nbin * i] * (o - odown);
				}

				if(odown == oup){
					Mterm_lower[y+ny*x+ny*nbin*i] = Mterm_lowergrid[odown + 190 * x + 190 * nbin * i];

					Nterm_lower[y+ny*x+ny*nbin*i] = Nterm_lowergrid[odown + 190 * x + 190 * nbin * i];

					Pterm_lower[y+ny*x+ny*nbin*i] = Pterm_lowergrid[odown + 190 * x + 190 * nbin * i];

					Qterm_lower[y+ny*x+ny*nbin*i] = Qterm_lowergrid[odown + 190 * x + 190 * nbin * i];
				}
			}
		}
	}
}


// interpolates the Planck function for the layer temperatures from the pre-tabulated values
__global__ void planck_interpol_layer(double* lambda, double* lambda_edge,
		double* deltalambda, double* tempcell, double* planckband_cell,
		double* planck_grid, double* starflux, int realstar, int numlayers, int nwave){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nwave && i < (numlayers+2)){

		planckband_cell[i + x * (numlayers + 2)] = 0.0;

		double t = (tempcell[i] - 10.0) / 10.0;
		if (t > 399) {
			t = 399-0.0001;
		}
		if (t < 0) {
			t = 0.0001;
		}
		int tdown = floor(t);
		int tup = ceil(t);

		if(tdown != tup){
			planckband_cell[i + x * (numlayers + 2)] = planck_grid[x + tdown * nwave] * (tup - t)
															+ planck_grid[x + tup * nwave] * (t-tdown);
		}
		if(tdown == tup){
			planckband_cell[i + x * (numlayers + 2)] = planck_grid[x + tdown * nwave];
		}

		// taking stellar and internal temperatures
		if (i == numlayers) {
			if(realstar==1){
				planckband_cell[i + x * (numlayers + 2)] = starflux[x]/pi;
			}
			else{
				planckband_cell[i + x * (numlayers + 2)] = planck_grid[x + 400 * nwave];
			}
		}

		if (i == numlayers+1) {
			planckband_cell[i + x * (numlayers + 2)] = planck_grid[x + 401 * nwave];
		}
	}
}


// interpolates the Planck function for the interface temperatures from the pre-tabulated values
__global__ void planck_interpol_interface(double* lambda, double* lambda_edge, double* deltalambda,
		double* temp, double* planckband_int, double* planck_grid, int numinterfaces, int nwave){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nwave && i < numinterfaces){

		planckband_int[i + x * numinterfaces] = 0.0;

		double t = (temp[i] - 10.0) / 10.0;
		if (t > 399) {
			t = 399-0.0001;
		}
		if (t < 0) {
			t = 0.0001;
		}
		int tdown = floor(t);
		int tup = ceil(t);

		if(tdown != tup){
			planckband_int[i + x * numinterfaces] = planck_grid[x + tdown * nwave] * (tup - t)
													+ planck_grid[x + tup * nwave] * (t-tdown);
		}
		if(tdown == tup){
			planckband_int[i + x * numinterfaces] = planck_grid[x + tdown * nwave];
		}
	}
}


// initialize empty flux arrays
__global__ void flux_init(double* F_down_wk, double* F_up_wk, double* Fc_down_wk, double* Fc_up_wk,
		int ny, int nbin, int numinterfaces){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int i = threadIdx.z + blockIdx.z * blockDim.z;

	if(x < nbin && y < ny && i < numinterfaces){

		F_down_wk[y + ny * x + ny * nbin * i] = 0;
		F_up_wk[y + ny * x + ny * nbin * i] = 0;

		if(i < numinterfaces-1){
			Fc_down_wk[y + ny * x + ny * nbin * i] = 0;
			Fc_up_wk[y + ny * x + ny * nbin * i] = 0;
		}
	}
}


// calculation of the spectral fluxes, isothermal case with pre-tabulated quantities
__global__ void fband_iso_tabu(double* F_down_wk, double* F_up_wk,
		double* planckband_lay, double* Mterm, double* Nterm, double* Pterm, double* Qterm,
		double* deltacolmass, double* kopac_lay, double* cross_scat,
		int scat, int singlewalk, double meanmolmass, double Rstar,
		double a, double g0, int numinterfaces, int nbin,
		double ffactor, int ny, double epsi) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nbin && y < ny) {

		// where to switch to pure scattering equations
		double w0_limit = 1.0-1e-6;

		for (int i = numinterfaces - 1; i >= 0; i--) {

			if (i == numinterfaces - 1) {
				F_down_wk[y + ny * x + ny * nbin * i] =
						ffactor * pow((Rstar / a),2.0) * pi * planckband_lay[i + x * (numinterfaces-1+2)];
			}
			else {
				double w0 = single_scat_alb(cross_scat[x], kopac_lay[y+ny*x + ny*nbin*i], meanmolmass);

				if(w0 < w0_limit || scat == 0){

					double M = Mterm[y + ny * x + ny * nbin * i];
					double P = Pterm[y + ny * x + ny * nbin * i];
					double N = Nterm[y + ny * x + ny * nbin * i];
					double Q = Qterm[y + ny * x + ny * nbin * i];

					F_down_wk[y+ny*x+ny*nbin*i] =
							1.0 / M * (P * F_down_wk[y+ny*x+ny*nbin*(i+1)] - N * F_up_wk[y+ny*x+ny*nbin*i]
							+ 2.0 * pi * epsi * planckband_lay[i+x*(numinterfaces-1+2)] * (N - Q));
				}
				else{
					double delta_tau = deltacolmass[i] * (kopac_lay[y+ny*x + ny*nbin*i] + cross_scat[x]/meanmolmass);

					F_down_wk[y+ny*x+ny*nbin*i] =
							F_down_wk[y+ny*x+ny*nbin*(i+1)]
							+ (F_up_wk[y+ny*x+ny*nbin*i] - F_down_wk[y+ny*x+ny*nbin*(i+1)])
							/ (2.0*epsi/((1.0-g0)*delta_tau) + 1.0);
				}
			}
		}

		for (int i = 0; i < numinterfaces; i++) {

			if (i == 0) {
				if(scat == 0 && singlewalk == 1){
					// without scattering there is no downward flux contribution to the upward stream
					F_up_wk[y + ny * x + ny * nbin * i] =
							0 + pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
				else{
					// the usual expression
					F_up_wk[y + ny * x + ny * nbin * i] =
							F_down_wk[y + ny * x + ny * nbin * i]
							+ pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
			}
			else {
				double w0 = single_scat_alb(cross_scat[x], kopac_lay[y+ny*x + ny*nbin*(i-1)], meanmolmass);

				if(w0 < w0_limit || scat == 0){

					double M = Mterm[y + ny * x + ny * nbin * (i-1)];
					double P = Pterm[y + ny * x + ny * nbin * (i-1)];
					double N = Nterm[y + ny * x + ny * nbin * (i-1)];
					double Q = Qterm[y + ny * x + ny * nbin * (i-1)];

					F_up_wk[y+ny*x+ny*nbin*i] =
							1.0 / M * (P * F_up_wk[y+ny*x+ny*nbin*(i-1)] - N * F_down_wk[y+ny*x+ny*nbin*i]
							+ 2.0 * pi * epsi * planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (N - Q));
				}
				else{
					double delta_tau = deltacolmass[i-1]
					                   * (kopac_lay[y+ny*x + ny*nbin*(i-1)] + cross_scat[x]/meanmolmass);

					F_up_wk[y+ny*x+ny*nbin*i] =
							F_up_wk[y+ny*x+ny*nbin*(i-1)]
							- (F_up_wk[y+ny*x+ny*nbin*(i-1)] - F_down_wk[y+ny*x+ny*nbin*i])
							/ (2.0*epsi/((1.0-g0)*delta_tau) + 1.0);
				}
			}
		}
	}
}


// calculation of the spectral fluxes, isothermal case with emphasis on on-the-fly calculations
__global__ void fband_iso_notabu(double* F_down_wk, double* F_up_wk, double* planckband_lay, double* deltacolmass,
		double* kopac_lay, double* cross_scat, int scat, int singlewalk, double meanmolmass,
		double Rstar, double a, double g0, int numinterfaces, int nbin, double ffactor, int ny, double epsi){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nbin && y < ny) {

		// where to switch to pure scattering equations
		double w0_limit = 1.0-1e-6;

		for (int i = numinterfaces - 1; i >= 0; i--){

			if (i == numinterfaces - 1) {
				F_down_wk[y + ny * x + ny * nbin * i] =
						ffactor * pow((Rstar / a),2.0) * pi * planckband_lay[i + x * (numinterfaces-1+2)];
			}
			else {
				double w0 = single_scat_alb(cross_scat[x], kopac_lay[y+ny*x + ny*nbin*i], meanmolmass);

				if(w0 < w0_limit || scat == 0){

					double trans;
					double zeta_min;
					double zeta_pl;

					if (scat == 1){
						double delta_tau = deltacolmass[i]
						                   * (kopac_lay[y+ny*x + ny*nbin*i] + cross_scat[x]/meanmolmass);
						trans = trans_func(epsi, delta_tau, w0, g0);
						zeta_min=zeta_minus(w0,g0);
						zeta_pl=zeta_plus(w0,g0);
					}
					else{
						trans = trans_func(epsi, deltacolmass[i] * kopac_lay[y+ny*x + ny*nbin*i] , 0, 0);
						zeta_min=0.0;
						zeta_pl=1.0;
					}

					double M = pow( zeta_min ,2.0) * pow(trans, 2.0) - pow( zeta_pl ,2.0);
					double P = (pow( zeta_min ,2.0) - pow( zeta_pl ,2.0)) * trans;
					double N = zeta_pl * zeta_min * (1.0 - pow(trans,2.0));
					double Q = (pow( zeta_min ,2.0) * trans + pow( zeta_pl ,2.0)) * (1.0 - trans);

					F_down_wk[y+ny*x+ny*nbin*i] =
							1.0 / M * (P * F_down_wk[y+ny*x+ny*nbin*(i+1)] - N * F_up_wk[y+ny*x+ny*nbin*i]
							+ 2.0 * pi * epsi * planckband_lay[i+x*(numinterfaces-1+2)] * (N - Q));
				}
				else{
					double delta_tau = deltacolmass[i]
					                   * (kopac_lay[y+ny*x + ny*nbin*i] + cross_scat[x]/meanmolmass);

					F_down_wk[y+ny*x+ny*nbin*i] =
							F_down_wk[y+ny*x+ny*nbin*(i+1)]
							+ (F_up_wk[y+ny*x+ny*nbin*i] - F_down_wk[y+ny*x+ny*nbin*(i+1)])
							/ (2.0*epsi/((1.0-g0)*delta_tau) + 1.0);
				}
			}
		}

		for (int i = 0; i < numinterfaces; i++){

			if (i == 0) {
				if(scat == 0 && singlewalk == 1){
					// without scattering there is no downward flux contribution to the upward stream
					F_up_wk[y + ny * x + ny * nbin * i] =
							0 + pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
				else{
					// the usual expression
					F_up_wk[y + ny * x + ny * nbin * i] =
							F_down_wk[y + ny * x + ny * nbin * i]
							+ pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
			}
			else {
				double w0 = single_scat_alb(cross_scat[x], kopac_lay[y+ny*x + ny*nbin*(i-1)], meanmolmass);

				if(w0 < w0_limit || scat == 0){

					double trans;
					double zeta_min;
					double zeta_pl;

					if (scat == 1){
						double delta_tau = deltacolmass[i-1]
						                   * (kopac_lay[y+ny*x + ny*nbin*(i-1)] + cross_scat[x]/meanmolmass);
						trans = trans_func(epsi, delta_tau, w0, g0);
						zeta_min=zeta_minus(w0,g0);
						zeta_pl=zeta_plus(w0,g0);
					}
					else{
						trans = trans_func(epsi, deltacolmass[i-1] * kopac_lay[y+ny*x + ny*nbin*(i-1)] , 0, 0);
						zeta_min=0.0;
						zeta_pl=1.0;
					}

					double M = pow( zeta_min ,2.0) * pow(trans, 2.0) - pow( zeta_pl ,2.0);
					double P = (pow( zeta_min ,2.0) - pow( zeta_pl ,2.0)) * trans;
					double N = zeta_pl * zeta_min * (1.0 - pow(trans,2.0));
					double Q = (pow( zeta_min ,2.0) * trans + pow( zeta_pl ,2.0)) * (1.0 - trans);

					F_up_wk[y+ny*x+ny*nbin*i] =
							1.0 / M * (P * F_up_wk[y+ny*x+ny*nbin*(i-1)] - N * F_down_wk[y+ny*x+ny*nbin*i]
							+ 2.0 * pi * epsi * planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (N-Q));
				}
				else{
					double delta_tau = deltacolmass[i-1]
					                   * (kopac_lay[y+ny*x + ny*nbin*(i-1)] + cross_scat[x]/meanmolmass);

					F_up_wk[y+ny*x+ny*nbin*i] =
							F_up_wk[y+ny*x+ny*nbin*(i-1)]
							        - (F_up_wk[y+ny*x+ny*nbin*(i-1)] - F_down_wk[y+ny*x+ny*nbin*i])
							        / (2.0*epsi/((1.0-g0)*delta_tau) + 1.0);
				}
			}
		}
	}
}


// calculation of the spectral fluxes, non-isothermal case with pre-tabulated quantities
__global__ void fband_noniso_tabu(double* F_down_wk, double* F_up_wk, double* Fc_down_wk, double* Fc_up_wk,
		double* planckband_lay, double* planckband_int, double* Mterm_upper, double* Nterm_upper,
		double* Pterm_upper, double* Qterm_upper, double* Mterm_lower, double* Nterm_lower, double* Pterm_lower,
		double* Qterm_lower, double* deltacolupper, double* deltacollower, double* kopac_lay, double* kopac_int,
		double* cross_scat, int scat, int singlewalk, double meanmolmass, double Rstar, double a, double g0, double epsi,
		int numinterfaces, int nbin, double ffactor, int ny){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nbin && y < ny) {

		// where to switch to pure scattering equations
		double w0_limit = 1.0-1e-6;

		for (int i = numinterfaces - 1; i >= 0; i--){

			if (i == numinterfaces - 1) {
				F_down_wk[y + ny * x + ny * nbin * i] =
						ffactor * pow((Rstar / a),2.0) * pi * planckband_lay[i + x * (numinterfaces-1+2)];
			}
			else {
				double kopac_up = (kopac_lay[y+ny*x + ny*nbin*i] + kopac_int[y+ny*x + ny*nbin*(i+1)])/2.0;

				double w0_up;
				double delta_tau_up;

				if(scat==1){
					w0_up = single_scat_alb(cross_scat[x], kopac_up, meanmolmass);
					delta_tau_up = deltacolupper[i] * (kopac_up + cross_scat[x]/meanmolmass);
				}
				if(scat==0){
					w0_up = 0;
					delta_tau_up = deltacolupper[i] * kopac_up;
				}

				if(w0_up < w0_limit || scat == 0){

					double pgrad_upper = (planckband_lay[i + x * (numinterfaces-1+2)]
					                     - planckband_int[(i + 1) + x * numinterfaces]) / delta_tau_up;

					double M_up = Mterm_upper[y + ny * x + ny * nbin * i];
					double N_up = Nterm_upper[y + ny * x + ny * nbin * i];
					double P_up = Pterm_upper[y + ny * x + ny * nbin * i];
					double Q_up = Qterm_upper[y + ny * x + ny * nbin * i];

					Fc_down_wk[y+ny*x+ny*nbin*i] =
							1.0 / M_up * (P_up * F_down_wk[y+ny*x+ny*nbin*(i+1)] - N_up * Fc_up_wk[y+ny*x+ny*nbin*i]
					        + 2.0*pi*epsi*(planckband_lay[i+x*(numinterfaces-1+2)] * (M_up + N_up)
					        - planckband_int[(i+1)+x*numinterfaces] * P_up
					        + epsi/(1.0+w0_up*g0) * pgrad_upper * (P_up - M_up + N_up) ) );

					// to prevent instability for very small optical depths
					if( delta_tau_up < 1e-4 ){
						Fc_down_wk[y+ny*x+ny*nbin*i] =
								1.0 / M_up * (P_up * F_down_wk[y+ny*x+ny*nbin*(i+1)]
								- N_up * Fc_up_wk[y+ny*x+ny*nbin*i]
								+ 2.0*pi*epsi*( (planckband_int[(i+1)+x*numinterfaces]
								+ planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_up - Q_up) ) );
					}
				}
				else{
					Fc_down_wk[y+ny*x+ny*nbin*i] =
							F_down_wk[y+ny*x+ny*nbin*(i+1)]
							+ (Fc_up_wk[y+ny*x+ny*nbin*i] - F_down_wk[y+ny*x+ny*nbin*(i+1)])
							/ (2.0*epsi / ((1.0-g0)*delta_tau_up) + 1.0);
				}

				double kopac_low = (kopac_int[y+ny*x + ny*nbin*i] + kopac_lay[y+ny*x + ny*nbin*i])/2.0;
				double w0_low;
				double delta_tau_low;

				if(scat==1){
					w0_low = single_scat_alb(cross_scat[x], kopac_low, meanmolmass);
					delta_tau_low = deltacollower[i] * (kopac_low + cross_scat[x]/meanmolmass);
				}
				if(scat==0){
					w0_low = 0;
					delta_tau_low = deltacollower[i] * kopac_low;
				}

				if(w0_low < w0_limit || scat == 0){

					double pgrad_lower = (planckband_int[i + x * numinterfaces]
					                     - planckband_lay[i + x * (numinterfaces-1+2)]) / delta_tau_low;

					double M_low = Mterm_lower[y + ny * x + ny * nbin * i];
					double N_low = Nterm_lower[y + ny * x + ny * nbin * i];
					double P_low = Pterm_lower[y + ny * x + ny * nbin * i];
					double Q_low = Qterm_lower[y + ny * x + ny * nbin * i];

					F_down_wk[y+ny*x+ny*nbin*i] =
							1.0 / M_low * (P_low * Fc_down_wk[y+ny*x+ny*nbin*i] - N_low * F_up_wk[y+ny*x+ny*nbin*i]
					        + 2.0*pi*epsi* (planckband_int[i+x*numinterfaces] * (M_low + N_low)
					        - planckband_lay[i+x*(numinterfaces-1+2)] * P_low
					        + epsi/(1.0+w0_low*g0) * pgrad_lower * (P_low - M_low + N_low) ) );

					// to prevent instability for very small optical depths
					if(delta_tau_low < 1e-4){
						F_down_wk[y+ny*x+ny*nbin*i] =
								1.0 / M_low * (P_low * Fc_down_wk[y+ny*x+ny*nbin*i]
								- N_low * F_up_wk[y+ny*x+ny*nbin*i]
								+ 2.0*pi*epsi*( (planckband_int[i+x*numinterfaces]
								+ planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_low - Q_low) ) );
					}
				}
				else{
					F_down_wk[y+ny*x+ny*nbin*i] =
							Fc_down_wk[y+ny*x+ny*nbin*i]
							+ (F_up_wk[y+ny*x+ny*nbin*i] - Fc_down_wk[y+ny*x+ny*nbin*i])
							/ (2.0*epsi / ((1.0-g0)*delta_tau_low) + 1.0);
				}
			}
		}

		for (int i = 0; i < numinterfaces; i++) {

			if (i == 0) {
				if(scat == 0 && singlewalk == 1){
					// without scattering there is no downward flux contribution to the upward stream
					F_up_wk[y + ny * x + ny * nbin * i] =
							0 + pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
				else{
					// the usual expression
					F_up_wk[y + ny * x + ny * nbin * i] =
							F_down_wk[y + ny * x + ny * nbin * i]
							+ pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
			}
			else {
				double kopac_low = (kopac_int[y+ny*x + ny*nbin*(i-1)] + kopac_lay[y+ny*x + ny*nbin*(i-1)])/2.0;
				double w0_low;
				double delta_tau_low;

				if(scat==1){
					w0_low = single_scat_alb(cross_scat[x], kopac_low, meanmolmass);
					delta_tau_low = deltacollower[i-1] * (kopac_low + cross_scat[x]/meanmolmass);
				}
				if(scat==0){
					w0_low = 0;
					delta_tau_low = deltacollower[i-1] * kopac_low;
				}
				if(w0_low < w0_limit || scat == 0){

					double pgrad_lower = (planckband_int[(i-1) + x * numinterfaces]
					                     - planckband_lay[(i-1) + x * (numinterfaces-1+2)]) / delta_tau_low;

					double M_low = Mterm_lower[y + ny * x + ny * nbin * (i-1)];
					double N_low = Nterm_lower[y + ny * x + ny * nbin * (i-1)];
					double P_low = Pterm_lower[y + ny * x + ny * nbin * (i-1)];
					double Q_low = Qterm_lower[y + ny * x + ny * nbin * (i-1)];

					Fc_up_wk[y+ny*x+ny*nbin*(i-1)] =
							1.0 / M_low * (P_low * F_up_wk[y+ny*x+ny*nbin*(i-1)]
							- N_low * Fc_down_wk[y+ny*x+ny*nbin*(i-1)]
							+ 2.0*pi*epsi* (planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (M_low + N_low)
							- planckband_int[(i-1)+x*numinterfaces] * P_low
							+ epsi/(1.0+w0_low*g0) * pgrad_lower * (M_low - P_low - N_low) ) );

					// to prevent instability for very small optical depths
					if(delta_tau_low < 1e-4){
						Fc_up_wk[y+ny*x+ny*nbin*(i-1)] =
								1.0 / M_low * (P_low * F_up_wk[y+ny*x+ny*nbin*(i-1)]
								- N_low * Fc_down_wk[y+ny*x+ny*nbin*(i-1)]
								+ 2.0*pi*epsi*( (planckband_int[(i-1)+x*numinterfaces]
								+ planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_low - Q_low) ) );
					}
				}
				else{
					Fc_up_wk[y+ny*x+ny*nbin*(i-1)] =
							F_up_wk[y+ny*x+ny*nbin*(i-1)]
							- (F_up_wk[y+ny*x+ny*nbin*(i-1)] - Fc_down_wk[y+ny*x+ny*nbin*(i-1)])
							/ (2.0*epsi / ((1.0-g0)*delta_tau_low) + 1.0);
				}

				double kopac_up = (kopac_lay[y+ny*x + ny*nbin*(i-1)] + kopac_int[y+ny*x + ny*nbin*i])/2.0;
				double w0_up;
				double delta_tau_up;

				if(scat==1){
					w0_up = single_scat_alb(cross_scat[x], kopac_up, meanmolmass);
					delta_tau_up = deltacolupper[i-1] * (kopac_up + cross_scat[x]/meanmolmass);
				}
				if(scat==0){
					w0_up = 0;
					delta_tau_up = deltacolupper[i-1] * kopac_up;
				}

				if(w0_up < w0_limit || scat == 0){

					double pgrad_upper = (planckband_lay[(i-1) + x * (numinterfaces-1+2)]
					                     - planckband_int[i + x * numinterfaces]) / delta_tau_up;

					double M_up = Mterm_upper[y + ny * x + ny * nbin * (i-1)];
					double N_up = Nterm_upper[y + ny * x + ny * nbin * (i-1)];
					double P_up = Pterm_upper[y + ny * x + ny * nbin * (i-1)];
					double Q_up = Qterm_upper[y + ny * x + ny * nbin * (i-1)];

					F_up_wk[y+ny*x+ny*nbin*i] =
							1.0 / M_up * (P_up * Fc_up_wk[y+ny*x+ny*nbin*(i-1)]
							- N_up * F_down_wk[y+ny*x+ny*nbin*i]
							+ 2.0*pi*epsi* (planckband_int[i+x*numinterfaces] * (M_up + N_up)
						    - planckband_lay[(i-1)+x*(numinterfaces-1+2)] * P_up
						    + epsi/(1.0+w0_up*g0) * pgrad_upper * (M_up - P_up - N_up) ) );

					// to prevent instability for very small optical depths
					if( delta_tau_up < 1e-4 ){
						F_up_wk[y+ny*x+ny*nbin*i] =
								1.0 / M_up * (P_up * Fc_up_wk[y+ny*x+ny*nbin*(i-1)]
								- N_up * F_down_wk[y+ny*x+ny*nbin*i]
								+ 2.0*pi*epsi*( (planckband_int[i+x*numinterfaces]
								+ planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_up - Q_up) ) );
					}
				}
				else{
					F_up_wk[y+ny*x+ny*nbin*i] =
							Fc_up_wk[y+ny*x+ny*nbin*(i-1)]
							- (Fc_up_wk[y+ny*x+ny*nbin*(i-1)] - F_down_wk[y+ny*x+ny*nbin*i])
							/ (2.0*epsi / ((1.0-g0)*delta_tau_up) + 1.0);
				}
			}
		}
	}
}


// calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
__global__ void fband_noniso_notabu(double* F_down_wk, double* F_up_wk, double* Fc_down_wk, double* Fc_up_wk,
		double* planckband_lay, double* planckband_int, double* deltacolupper, double* deltacollower,
		double* kopac_lay, double* kopac_int, double* cross_scat, int scat, int singlewalk, double meanmolmass,
		double Rstar, double a, double g0, double epsi, int numinterfaces, int nbin, double ffactor, int ny){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nbin && y < ny) {

		// where to switch to pure scattering equations
		double w0_limit = 1.0-1e-6;

		for (int i = numinterfaces - 1; i >= 0; i--){

			if (i == numinterfaces - 1) {
				F_down_wk[y + ny * x + ny * nbin * i] =
						ffactor * pow((Rstar / a),2.0) * pi * planckband_lay[i + x * (numinterfaces-1+2)];
			}
			else {

				double kopac_up = (kopac_lay[y+ny*x + ny*nbin*i] + kopac_int[y+ny*x + ny*nbin*(i+1)])/2.0;
				double trans;
				double zeta_min;
				double zeta_pl;
				double w0_up;
				double delta_tau_up;

				if(scat==1){
					w0_up = single_scat_alb(cross_scat[x], kopac_up, meanmolmass);
					delta_tau_up = deltacolupper[i] * (kopac_up + cross_scat[x]/meanmolmass);
					trans = trans_func(epsi, delta_tau_up , w0_up, g0);
					zeta_min=zeta_minus(w0_up,g0);
					zeta_pl=zeta_plus(w0_up,g0);
				}
				if(scat==0){
					w0_up = 0;
					delta_tau_up = deltacolupper[i] * kopac_up;
					trans = trans_func(epsi, delta_tau_up, 0, 0);
					zeta_min=0.0;
					zeta_pl=1.0;
				}

				if(w0_up < w0_limit || scat == 0){

					double pgrad_upper = (planckband_lay[i + x * (numinterfaces-1+2)]
					                     - planckband_int[(i + 1) + x * numinterfaces]) / delta_tau_up;

					double M_up = pow( zeta_min ,2.0) * pow(trans, 2.0) - pow( zeta_pl ,2.0);
					double P_up = (pow( zeta_min ,2.0) - pow( zeta_pl ,2.0)) * trans;
					double N_up = zeta_pl * zeta_min * (1.0 - pow(trans,2.0));
					double Q_up = (pow( zeta_min ,2.0) * trans + pow( zeta_pl ,2.0)) * (1.0 - trans);

					Fc_down_wk[y+ny*x+ny*nbin*i] =
							1.0 / M_up * (P_up * F_down_wk[y+ny*x+ny*nbin*(i+1)]
							- N_up * Fc_up_wk[y+ny*x+ny*nbin*i]
							+ 2.0*pi*epsi* (planckband_lay[i+x*(numinterfaces-1+2)] * (M_up + N_up)
					        - planckband_int[(i+1)+x*numinterfaces] * P_up
					        + epsi/(1.0+w0_up*g0) * pgrad_upper * (P_up - M_up + N_up) ) );

					// to prevent instability from too small optical depths
					if(delta_tau_up < 1e-4){
						Fc_down_wk[y+ny*x+ny*nbin*i] =
								1.0 / M_up * (P_up * F_down_wk[y+ny*x+ny*nbin*(i+1)]
								- N_up * Fc_up_wk[y+ny*x+ny*nbin*i]
								+ 2.0*pi*epsi*( (planckband_int[(i+1)+x*numinterfaces]
								+ planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_up - Q_up) ) );
					}
				}
				else{
					Fc_down_wk[y+ny*x+ny*nbin*i] =
							F_down_wk[y+ny*x+ny*nbin*(i+1)]
							+ (Fc_up_wk[y+ny*x+ny*nbin*i] - F_down_wk[y+ny*x+ny*nbin*(i+1)])
							/ (2.0*epsi/((1.0-g0)*delta_tau_up) + 1.0);
				}

				double kopac_low = (kopac_int[y+ny*x + ny*nbin*i] + kopac_lay[y+ny*x + ny*nbin*i])/2.0;
				double w0_low;
				double delta_tau_low;

				if(scat==1){
					w0_low = single_scat_alb(cross_scat[x], kopac_low, meanmolmass);
					delta_tau_low = deltacollower[i] * (kopac_low + cross_scat[x]/meanmolmass);
					trans = trans_func(epsi, delta_tau_low , w0_low, g0);
					zeta_min=zeta_minus(w0_low,g0);
					zeta_pl=zeta_plus(w0_low,g0);
				}
				if(scat==0){
					w0_low = 0;
					delta_tau_low = deltacollower[i] * kopac_low;
					trans = trans_func(epsi, delta_tau_low, 0, 0);
					zeta_min=0.0;
					zeta_pl=1.0;
				}

				if(w0_low < w0_limit || scat == 0){

					double pgrad_lower = (planckband_int[i + x * numinterfaces]
					                     - planckband_lay[i + x * (numinterfaces-1+2)]) / delta_tau_low;

					double M_low = pow( zeta_min ,2.0) * pow(trans, 2.0) - pow( zeta_pl ,2.0);
					double P_low = (pow( zeta_min ,2.0) - pow( zeta_pl ,2.0)) * trans;
					double N_low = zeta_pl * zeta_min * (1.0 - pow(trans,2.0));
					double Q_low = (pow( zeta_min ,2.0) * trans + pow( zeta_pl ,2.0)) * (1.0 - trans);

					F_down_wk[y+ny*x+ny*nbin*i] =
							1.0 / M_low * (P_low * Fc_down_wk[y+ny*x+ny*nbin*i]
							- N_low * F_up_wk[y+ny*x+ny*nbin*i]
							+ 2.0*pi*epsi* (planckband_int[i+x*numinterfaces] * (M_low + N_low)
							- planckband_lay[i+x*(numinterfaces-1+2)] * P_low
							+ epsi/(1.0+w0_low*g0) * pgrad_lower * (P_low - M_low + N_low) ) );

					// to prevent instability from too small optical depths
					if(delta_tau_low < 1e-4){
						F_down_wk[y+ny*x+ny*nbin*i] =
								1.0 / M_low * (P_low * Fc_down_wk[y+ny*x+ny*nbin*i]
								- N_low * F_up_wk[y+ny*x+ny*nbin*i]
								+ 2.0*pi*epsi*( (planckband_int[i+x*numinterfaces]
								+ planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_low - Q_low) ) );
					}
				}
				else{
					F_down_wk[y+ny*x+ny*nbin*i] =
							Fc_down_wk[y+ny*x+ny*nbin*i]
							+ (F_up_wk[y+ny*x+ny*nbin*i] - Fc_down_wk[y+ny*x+ny*nbin*i])
							/ (2.0*epsi/((1.0-g0)*delta_tau_low) + 1.0);
				}
			}
		}

		for (int i = 0; i < numinterfaces; i++){

			if (i == 0) {
				if(scat == 0 && singlewalk == 1){
					// without scattering there is no downward flux contribution to the upward stream
					F_up_wk[y + ny * x + ny * nbin * i] =
							0 + pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
				else{
					// usual expression
					F_up_wk[y + ny * x + ny * nbin * i] =
							F_down_wk[y + ny * x + ny * nbin * i]
							+ pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
			}
			else {

				double kopac_low = (kopac_int[y+ny*x + ny*nbin*(i-1)] + kopac_lay[y+ny*x + ny*nbin*(i-1)])/2.0;
				double w0_low;
				double delta_tau_low;
				double trans;
				double zeta_min;
				double zeta_pl;

				if(scat==1){
					w0_low = single_scat_alb(cross_scat[x], kopac_low, meanmolmass);
					delta_tau_low = deltacollower[i-1] * (kopac_low + cross_scat[x]/meanmolmass);
					trans = trans_func(epsi, delta_tau_low , w0_low, g0);
					zeta_min=zeta_minus(w0_low,g0);
					zeta_pl=zeta_plus(w0_low,g0);
				}
				if(scat==0){
					w0_low = 0;
					delta_tau_low = deltacollower[i-1] * kopac_low;
					trans = trans_func(epsi, delta_tau_low, 0, 0);
					zeta_min=0.0;
					zeta_pl=1.0;
				}

				if(w0_low < w0_limit || scat == 0){

					double pgrad_lower = (planckband_int[(i-1) + x * numinterfaces]
					                     - planckband_lay[(i-1) + x * (numinterfaces-1+2)]) / delta_tau_low;

					double M_low = pow( zeta_min ,2.0) * pow(trans, 2.0) - pow( zeta_pl ,2.0);
					double P_low = (pow( zeta_min ,2.0) - pow( zeta_pl ,2.0)) * trans;
					double N_low = zeta_pl * zeta_min * (1.0 - pow(trans,2.0));
					double Q_low = (pow( zeta_min ,2.0) * trans + pow( zeta_pl ,2.0)) * (1.0 - trans);

					Fc_up_wk[y+ny*x+ny*nbin*(i-1)] =
							1.0 / M_low * (P_low * F_up_wk[y+ny*x+ny*nbin*(i-1)]
							- N_low * Fc_down_wk[y+ny*x+ny*nbin*(i-1)]
							+ 2.0*pi*epsi* (planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (M_low + N_low)
						    - planckband_int[(i-1)+x*numinterfaces] * P_low
						    + epsi/(1.0+w0_low*g0) * pgrad_lower * (M_low - P_low - N_low) ) );

					// to prevent instability for very small optical depths
					if(delta_tau_low < 1e-4){
						Fc_up_wk[y+ny*x+ny*nbin*(i-1)] =
								1.0 / M_low * (P_low * F_up_wk[y+ny*x+ny*nbin*(i-1)]
								- N_low * Fc_down_wk[y+ny*x+ny*nbin*(i-1)]
								+ 2.0*pi*epsi*( (planckband_int[(i-1)+x*numinterfaces]
								+ planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_low - Q_low) ) );
					}
				}
				else{
					Fc_up_wk[y+ny*x+ny*nbin*(i-1)] =
							F_up_wk[y+ny*x+ny*nbin*(i-1)]
							- (F_up_wk[y+ny*x+ny*nbin*(i-1)] - Fc_down_wk[y+ny*x+ny*nbin*(i-1)])
							/ (2.0*epsi/((1.0-g0)*delta_tau_low) + 1.0);
				}

				double kopac_up = (kopac_lay[y+ny*x + ny*nbin*(i-1)] + kopac_int[y+ny*x + ny*nbin*i])/2.0;
				double w0_up;
				double delta_tau_up;

				if(scat==1){
					w0_up = single_scat_alb(cross_scat[x], kopac_up, meanmolmass);
					delta_tau_up = deltacolupper[i-1] * (kopac_up + cross_scat[x]/meanmolmass);
					trans = trans_func(epsi, delta_tau_up , w0_up, g0);
					zeta_min=zeta_minus(w0_up,g0);
					zeta_pl=zeta_plus(w0_up,g0);
				}
				if(scat==0){
					w0_up = 0;
					delta_tau_up = deltacolupper[i-1] * kopac_up;
					trans = trans_func(epsi, delta_tau_up, 0, 0);
					zeta_min=0.0;
					zeta_pl=1.0;
				}

				if(w0_up < w0_limit || scat == 0){

					double pgrad_upper = (planckband_lay[(i-1) + x * (numinterfaces-1+2)]
					                     - planckband_int[i + x * numinterfaces]) / delta_tau_up;

					double M_up = pow( zeta_min ,2.0) * pow(trans, 2.0) - pow( zeta_pl ,2.0);
					double P_up = (pow( zeta_min ,2.0) - pow( zeta_pl ,2.0)) * trans;
					double N_up = zeta_pl * zeta_min * (1.0 - pow(trans,2.0));
					double Q_up = (pow( zeta_min ,2.0) * trans + pow( zeta_pl ,2.0)) * (1.0 - trans);

					F_up_wk[y+ny*x+ny*nbin*i] =
							1.0 / M_up * (P_up * Fc_up_wk[y+ny*x+ny*nbin*(i-1)]
							- N_up * F_down_wk[y+ny*x+ny*nbin*i]
							+ 2.0*pi*epsi* (planckband_int[i+x*numinterfaces] * (M_up + N_up)
							- planckband_lay[(i-1)+x*(numinterfaces-1+2)] * P_up
							+ epsi/(1.0+w0_up*g0) * pgrad_upper * (M_up - P_up - N_up) ) );

					// to prevent instabilities for too small optical depths
					if(delta_tau_up < 1e-4){
						F_up_wk[y+ny*x+ny*nbin*i] =
								1.0 / M_up * (P_up * Fc_up_wk[y+ny*x+ny*nbin*(i-1)]
								- N_up * F_down_wk[y+ny*x+ny*nbin*i]
								+ 2.0*pi*epsi*( (planckband_int[i+x*numinterfaces]
								+ planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_up - Q_up) ) );
					}
				}
				else{
					F_up_wk[y+ny*x+ny*nbin*i] =
							Fc_up_wk[y+ny*x+ny*nbin*(i-1)]
							- (Fc_up_wk[y+ny*x+ny*nbin*(i-1)] - F_down_wk[y+ny*x+ny*nbin*i])
							/ (2.0*epsi/((1.0-g0)*delta_tau_up) + 1.0);
				}
			}
		}
	}
}


// calculation of the spectral fluxes, isothermal case with direct (exact) method
__global__ void fband_iso_direct(double* F_down_wk, double* F_up_wk, double* planckband_lay,
		double* deltacolmass, double* kopac_lay, double Rstar, double a, int singlewalk,
		int numinterfaces, int nbin, double ffactor, int ny){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nbin && y < ny) {

		for (int i = numinterfaces - 1; i >= 0; i--){

			if (i == numinterfaces - 1) {
				F_down_wk[y + ny * x + ny * nbin * i] =
						ffactor * pow((Rstar / a),2.0) * pi * planckband_lay[i + x * (numinterfaces-1+2)];
			}
			else {
				double delta_tau = deltacolmass[i] * kopac_lay[y+ny*x + ny*nbin*i];
				double trans = (1.0 - delta_tau) * exp(-delta_tau) + pow(delta_tau,2.0) * expint1(delta_tau);

				F_down_wk[y+ny*x+ny*nbin*i] =
						F_down_wk[y+ny*x+ny*nbin*(i+1)]*trans
						+ pi*planckband_lay[i+x*(numinterfaces-1+2)] * (1.0 - trans);
			}
		}

		for (int i = 0; i < numinterfaces; i++){

			if (i == 0) {
				if(singlewalk == 1){
					// without scattering there is no downward flux contribution to the upward stream
					F_up_wk[y + ny * x + ny * nbin * i] =
							0 + pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
				else{
					// usual expression
					F_up_wk[y + ny * x + ny * nbin * i] =
							F_down_wk[y + ny * x + ny * nbin * i]
							+ pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
			}
			else {
				double delta_tau = deltacolmass[i-1] * kopac_lay[y+ny*x + ny*nbin*(i-1)];
				double trans = (1.0 - delta_tau) * exp(-delta_tau) + pow(delta_tau,2.0) * expint1(delta_tau);

				F_up_wk[y+ny*x+ny*nbin*i] =
						F_up_wk[y+ny*x+ny*nbin*(i-1)]*trans
						+ pi*planckband_lay[(i-1) + x * (numinterfaces-1+2)]*(1.0-trans);
			}
		}
	}
}


// calculation of the spectral fluxes, non-isothermal case with direct (exact) method
__global__ void fband_noniso_direct(double* F_down_wk, double* F_up_wk, double* Fc_down_wk, double* Fc_up_wk,
		double* planckband_lay, double* planckband_int, double* deltacolupper, double* deltacollower,
		double* kopac_lay, double* kopac_int, double Rstar, double a, int singlewalk, int numinterfaces,
		int nbin, double ffactor, int ny){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nbin && y < ny) {

		for (int i = numinterfaces - 1; i >= 0; i--){

			if (i == numinterfaces - 1) {
				F_down_wk[y + ny * x + ny * nbin * i] =
						ffactor * pow((Rstar / a),2.0) * pi * planckband_lay[i + x * (numinterfaces-1+2)];
			}
			else {

				double kopac_up = (kopac_lay[y+ny*x + ny*nbin*i] + kopac_int[y+ny*x + ny*nbin*(i+1)])/2.0;
				double delta_tau_up = deltacolupper[i] * kopac_up;
				double trans_up = (1.0 - delta_tau_up) * exp(-delta_tau_up)
								  + pow(delta_tau_up,2.0) * expint1(delta_tau_up);
				double pgrad_upper = (planckband_lay[i + x * (numinterfaces-1+2)]
				                     - planckband_int[(i + 1) + x * numinterfaces]) / delta_tau_up;

				Fc_down_wk[y+ny*x+ny*nbin*i] =
						F_down_wk[y+ny*x+ny*nbin*(i+1)]*trans_up
						+ pi*planckband_int[(i+1)+x*numinterfaces]*(1.0-trans_up)
						+ pi*pgrad_upper*(-2.0/3.0*(1.0-exp(-delta_tau_up)) + delta_tau_up*(1.0-1.0/3.0*trans_up));

				// to prevent instability from too small optical depths
				if(delta_tau_up < 1e-4){
					Fc_down_wk[y+ny*x+ny*nbin*i] =
							F_down_wk[y+ny*x+ny*nbin*(i+1)]*trans_up
							+ pi*planckband_lay[i+x*(numinterfaces-1+2)]*(1.0-trans_up);
				}

				double kopac_low = (kopac_int[y+ny*x + ny*nbin*i] + kopac_lay[y+ny*x + ny*nbin*i])/2.0;
				double delta_tau_low = deltacollower[i] * kopac_low;
				double trans_low = (1.0 - delta_tau_low) * exp(-delta_tau_low)
								   + pow(delta_tau_low,2.0) * expint1(delta_tau_low);
				double pgrad_lower = (planckband_int[i + x * numinterfaces]
				                     - planckband_lay[i + x * (numinterfaces-1+2)]) / delta_tau_low;

				F_down_wk[y+ny*x+ny*nbin*i] =
						Fc_down_wk[y+ny*x+ny*nbin*i]*trans_low
						+ pi*planckband_lay[i+x*(numinterfaces-1+2)]*(1.0-trans_low)
						+ pi*pgrad_lower*(-2.0/3.0*(1.0-exp(-delta_tau_low))
						+ delta_tau_low*(1.0-1.0/3.0*trans_low));

				// to prevent instability from too small optical depths
				if(delta_tau_low < 1e-4){
					F_down_wk[y+ny*x+ny*nbin*i] =
							Fc_down_wk[y+ny*x+ny*nbin*i]*trans_low
							+ pi*planckband_lay[i+x*(numinterfaces-1+2)]*(1.0-trans_low);
				}
			}
		}

		for (int i = 0; i < numinterfaces; i++){

			if (i == 0) {
				if(singlewalk == 1){
					// without scattering there is no downward flux contribution to the upward stream
					F_up_wk[y + ny * x + ny * nbin * i] =
							0 + pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
				else{
					// usual expression
					F_up_wk[y + ny * x + ny * nbin * i] =
							F_down_wk[y + ny * x + ny * nbin * i]
							+ pi * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
				}
			}
			else {

				double kopac_low = (kopac_int[y+ny*x + ny*nbin*(i-1)] + kopac_lay[y+ny*x + ny*nbin*(i-1)])/2.0;
				double delta_tau_low = deltacollower[i-1] * kopac_low;
				double trans_low = (1.0 - delta_tau_low) * exp(-delta_tau_low)
								   + pow(delta_tau_low,2.0) * expint1(delta_tau_low);
				double pgrad_lower = (planckband_int[(i-1) + x * numinterfaces]
				                     - planckband_lay[(i-1) + x * (numinterfaces-1+2)]) / delta_tau_low;

				Fc_up_wk[y+ny*x+ny*nbin*(i-1)] =
						F_up_wk[y+ny*x+ny*nbin*(i-1)]*trans_low
						+ pi*planckband_int[(i-1) + x * numinterfaces]*(1.0-trans_low)
						+ pi*pgrad_lower*(2.0/3.0*(1.0-exp(-delta_tau_low)) - delta_tau_low*(1.0-1.0/3.0*trans_low));

				// to prevent instabilities from very small optical depths
				if(delta_tau_low < 1e-4){
					Fc_up_wk[y+ny*x+ny*nbin*(i-1)] =
							F_up_wk[y+ny*x+ny*nbin*(i-1)]*trans_low
							+ pi*planckband_lay[(i-1) + x * (numinterfaces-1+2)]*(1.0-trans_low);
				}

				double kopac_up = (kopac_lay[y+ny*x + ny*nbin*(i-1)] + kopac_int[y+ny*x + ny*nbin*i])/2.0;
				double delta_tau_up = deltacolupper[i-1] * kopac_up;;
				double trans_up = (1.0 - delta_tau_up) * exp(-delta_tau_up)
								  + pow(delta_tau_up,2.0) * expint1(delta_tau_up);
				double pgrad_upper = (planckband_lay[(i-1) + x * (numinterfaces-1+2)]
				                     - planckband_int[i + x * numinterfaces]) / delta_tau_up;

				F_up_wk[y+ny*x+ny*nbin*i] =
						Fc_up_wk[y+ny*x+ny*nbin*(i-1)]*trans_up
						+ pi*planckband_lay[(i-1) + x * (numinterfaces-1+2)]*(1.0-trans_up)
						+ pi*pgrad_upper*(2.0/3.0*(1.0-exp(-delta_tau_up)) - delta_tau_up*(1.0-1.0/3.0*trans_up));

				// to prevent instabilities from very small optical depths
				if(delta_tau_up < 1e-4){
						F_up_wk[y+ny*x+ny*nbin*i] =
								Fc_up_wk[y+ny*x+ny*nbin*(i-1)]*trans_up
								+ pi*planckband_lay[(i-1) + x * (numinterfaces-1+2)]*(1.0-trans_up);
				}
			}
		}
	}
}


// calculates the integrated upwards and downwards fluxes
__global__ void flux_integrate(double* deltalambda, double* F_down_tot, double* F_up_tot, double* F_down_wk, double* F_up_wk,
		double* F_down_band, double* F_up_band, double* kw, int nbin, int numinterfaces, int ny){

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i < numinterfaces){

		F_up_tot[i] = 0;
		F_down_tot[i] = 0;

		for (int x = 0; x < nbin; x++) {

			F_up_band[x + nbin * i] = 0;
			F_down_band[x + nbin * i] = 0;

			// Gauss - Legendre integration over each bin
			for(int y=0;y<ny;y++){
				F_up_band[x + nbin * i] += 0.5 * kw[y] * F_up_wk[y + ny * x + ny * nbin * i];
				F_down_band[x + nbin * i] += 0.5 * kw[y] * F_down_wk[y + ny * x + ny * nbin * i];
			}

			// sum the bin contributions to obtain the integrated flux
			F_up_tot[i] += F_up_band[x + nbin * i] * deltalambda[x];
			F_down_tot[i] += F_down_band[x + nbin * i] * deltalambda[x];
		}
	}
}


// calculates the net fluxes and advances the layer temperatures
__global__ void netfluxandtempiter(double* F_down_tot, double* F_up_tot, double* F_net, double* tlay, double* play,
		double* tint, double* pint, int* abrt, double* deltatlay, double* delta_T_store, double* deltat_prefactor,
		int itervalue, double ffactor, int prequel, double meanmolmass, double Rstar, double Tstar, double g,
		double a, double deltatime, int numlayers, int varydelta, double c_p){

	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if(i < numlayers){

		// obtain constant timestep value
		deltatlay[i] = deltatime;

		// net flux for each layer
		F_net[i] = (F_up_tot[i] - F_down_tot[i])
							- (F_up_tot[i + 1] - F_down_tot[i + 1]);

		// approximately effective temperature of the atmosphere
		double T_eff = pow(ffactor,0.25) * pow((Rstar / a), 0.5) * Tstar;

		// layer density
		double rholay = play[i] * meanmolmass / (kBoltzmann * tlay[i]);

		// radiative timescale approximation
		double t_rad = c_p * play[i] / (stefanboltzmann * g * pow(T_eff, 3.0));

		// vertical thickness of layer
		double delta_z = kBoltzmann * tlay[i] / (meanmolmass * g) * log(pint[i] / pint[i+1]);

		// if using varying timestep
		if(varydelta == 1){
			if (itervalue == prequel){
				deltat_prefactor[i] = 1e5;
			}

			if(F_net[i] != 0){
				deltatlay[i] = deltat_prefactor[i] * t_rad / pow(abs(F_net[i]), 0.9);
			}
		}

		double delta_T = 1.0 / (rholay * c_p) * F_net[i] / delta_z * deltatlay[i];

		// limit to large temperature jumps
		if(abs(delta_T) > 100.0){
			delta_T = 100.0 * F_net[i]/abs(F_net[i]);
		}

		// adaptive timestepping method
		if (varydelta == 1) {
			if (itervalue % 6 == 0) {
				// delete the last 6 stored temperature step entries
				for(int n = 0; n < 6; n++){
					delta_T_store[n + i * 6] = 0;
				}
			}
			// store always last 6 entries
			int n = itervalue % 6;
			delta_T_store[n + i * 6] = delta_T;

			// after every 6th storage, decrease or increase timestep
			if (itervalue % 6 == 5) {
				double sum = 0;
				for(int n = 0; n < 6; n++){
					sum += delta_T_store[n + i * 6];
				}
				if(abs(sum) <= abs(delta_T)){
					deltat_prefactor[i] /= 1.5;
				}
				int more_than_0 = 0;
				int less_than_0 = 0;
				for(int n = 0; n < 6; n++){
					if(delta_T_store[n + i * 6] < 0){
						less_than_0 += 1;
					}
					else{
						more_than_0 += 1;
					}
				}
				if(less_than_0==6 || more_than_0==6){
					deltat_prefactor[i] *= 1.1;
				}
			}
		}

		// update layer temperatures
		tlay[i] = tlay[i] + delta_T;

		// abort condition
		if (abs(F_net[i]/(stefanboltzmann*pow(tlay[i],4.0))) < 1e-7){
			abrt[i] = 1;
		}
		else {
			abrt[i] = 0;
		}
	}
}


// calculates the Planck and Rosseland mean opacities for each layer
__global__ void mean_opacities(double* planck_opac_planet, double* ross_opac_planet,
		double* planck_opac_star, double* ross_opac_star, double* kopac_lay, double* planckband_lay,
		double* lambda, double* lambda_edge, double* deltalambda, double* temp, double* kw, double* ky,
		double* opac_lay, int numlayers, int nbin, int ny, double Tstar) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if(i < numlayers){

		double numerator_planck_planet = 0;
		double denominator_planck_planet = 0;
		double numerator_ross_planet = 0;
		double denominator_ross_planet = 0;
		double numerator_planck_star = 0;
		double denominator_planck_star = 0;
		double numerator_ross_star = 0;
		double denominator_ross_star = 0;

		// integrate opacity over each bin with Gauss - Legendre
		for(int x=0;x<nbin;x++){

			opac_lay[i+numlayers*x]=0;

			for (int y=0;y<ny;y++){
				opac_lay[i+numlayers*x]+= 0.5 * kw[y] * kopac_lay[y+ny*x+ny*nbin*i];
			}
		}

		for (int x = 0; x < nbin; x++) {

			// calculate Planck mean opacity with layer temperatures
			numerator_planck_planet +=
					opac_lay[i+numlayers*x] * planckband_lay[i+x*(numlayers+2)]*deltalambda[x];
			denominator_planck_planet +=
					planckband_lay[i+x*(numlayers+2)]*deltalambda[x];

			// calculate Rosseland mean opacity with layer temperatures
			numerator_ross_planet +=
					integrated_dB_dT(kw,ky,ny,lambda_edge[x],lambda_edge[x+1],temp[i]);
			denominator_ross_planet +=
					integrated_dB_dT(kw,ky,ny,lambda_edge[x],lambda_edge[x+1],temp[i]) / opac_lay[i+numlayers*x];

			// calculate Planck mean opacity with stellar blackbody function
			numerator_planck_star +=
					opac_lay[i+numlayers*x] * planckband_lay[numlayers+x*(numlayers+2)]*deltalambda[x];
			denominator_planck_star +=
					planckband_lay[numlayers+x*(numlayers+2)]*deltalambda[x];

			// calculate Rosseland mean opacity with stellar blackbody function
			numerator_ross_star +=
					integrated_dB_dT(kw,ky,ny, lambda_edge[x],lambda_edge[x+1],Tstar);
			denominator_ross_star +=
					integrated_dB_dT(kw,ky,ny, lambda_edge[x],lambda_edge[x+1],Tstar) / opac_lay[i+numlayers*x];
		}

		planck_opac_planet[i] = numerator_planck_planet / denominator_planck_planet;
		ross_opac_planet[i] = numerator_ross_planet / denominator_ross_planet;

		planck_opac_star[i] = numerator_planck_star / denominator_planck_star;
		ross_opac_star[i] = numerator_ross_star / denominator_ross_star;
	}
}


// calculates the transmission function for each layer - for output purposes only
__global__ void transmission(double* transmission, double* kopac_lay, double* kw, double* colmass,
		double* cross_scat, int scat, double epsi, int nbin, int numlayers, int ny, double meanmolmass, double g0){

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int i = threadIdx.y + blockIdx.y*blockDim.y;

	if (x < nbin && i < numlayers){

		double w0;
		double delta_tau;

		transmission[x+nbin*i] = 0;

		for(int y=0;y<ny;y++){

			if(scat == 0){
				w0 = 0;
				delta_tau = colmass[i] * kopac_lay[y+ny*x+ny*nbin*i];
			}
			if(scat == 1){
				w0 = single_scat_alb(cross_scat[x], kopac_lay[y+ny*x+ny*nbin*i], meanmolmass);
				delta_tau = colmass[i] * (kopac_lay[y+ny*x+ny*nbin*i] + cross_scat[x]/meanmolmass);
			}

			transmission[x+nbin*i] += 0.5 * kw[y] * trans_func(epsi, delta_tau, w0, g0);
		}
	}
}
