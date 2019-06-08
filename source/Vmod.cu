// =================================================================================
// This file contains the device functions and CUDA kernels used in the VULCAN mod.
// Copyright (C) 2018 Matej Malik
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

// switch between utype and single precision
/***
#define USE_SINGLE
***/
#ifdef USE_SINGLE
typedef float utype;
#else
typedef double utype;
#endif

// physical constant:
const utype AMU = 1.660539e-24;
const utype M_H2O = 18.0153 * AMU;
const utype M_CO2 = 44.01 * AMU;
const utype M_CO = 28.01 * AMU;
const utype M_CH4 = 16.04 * AMU;
const utype M_NH3 = 17.031 * AMU;
const utype M_HCN = 27.0253 * AMU;
const utype M_C2H2 = 26.04 * AMU;
const utype M_TIO = 63.866 * AMU;
const utype M_VO = 66.9409 * AMU;
const utype M_H2 = 2.01588 * AMU;
const utype M_HE = 4.0026 * AMU;


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


// temperature interpolation for the non-isothermal layers
__global__ void f_mol_and_meanmass_inter(
		utype* 	f_h2o_lay, 
		utype* 	f_co2_lay,
		utype* 	f_co_lay,
		utype* 	f_ch4_lay, 
		utype* 	f_nh3_lay, 
		utype* 	f_hcn_lay,
		utype* 	f_c2h2_lay,
        utype* 	f_tio_lay,
        utype* 	f_vo_lay,
		utype* 	f_h2_lay, 
		utype* 	f_he_lay,
		utype* 	f_h2o_int, 
		utype* 	f_co2_int,
		utype* 	f_co_int,
		utype* 	f_ch4_int, 
		utype* 	f_nh3_int, 
		utype* 	f_hcn_int,
		utype* 	f_c2h2_int,
        utype* 	f_tio_int,
        utype* 	f_vo_int,
		utype* 	f_h2_int, 
		utype* 	f_he_int,
		utype*	meanmolmass_lay,
		utype*	meanmolmass_int,
		int numinterfaces
		){

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (0 < i && i < numinterfaces - 1) {
		f_h2o_int[i] = 0.5 * (f_h2o_lay[i - 1] + f_h2o_lay[i]);
		f_co2_int[i] = 0.5 * (f_co2_lay[i - 1] + f_co2_lay[i]);
		f_co_int[i] = 0.5 * (f_co_lay[i - 1] + f_co_lay[i]);
		f_ch4_int[i] = 0.5 * (f_ch4_lay[i - 1] + f_ch4_lay[i]);
		f_nh3_int[i] = 0.5 * (f_nh3_lay[i - 1] + f_nh3_lay[i]);
		f_hcn_int[i] = 0.5 * (f_hcn_lay[i - 1] + f_hcn_lay[i]);
		f_c2h2_int[i] = 0.5 * (f_c2h2_lay[i - 1] + f_c2h2_lay[i]);
        f_tio_int[i] = 0.5 * (f_tio_lay[i - 1] + f_tio_lay[i]);
        f_vo_int[i] = 0.5 * (f_vo_lay[i - 1] + f_vo_lay[i]);
		f_h2_int[i] = 0.5 * (f_h2_lay[i - 1] + f_h2_lay[i]);
		f_he_int[i] = 0.5 * (f_he_lay[i - 1] + f_he_lay[i]);
		meanmolmass_int[i] = 0.5 * (meanmolmass_lay[i - 1] + meanmolmass_lay[i]);
	}
	if (i == 0) {
		f_h2o_int[i] = f_h2o_lay[i];
		f_co2_int[i] = f_co2_lay[i];
		f_co_int[i] = f_co_lay[i];
		f_ch4_int[i] = f_h2o_lay[i];
		f_nh3_int[i] = f_nh3_lay[i];
		f_hcn_int[i] = f_hcn_lay[i];
		f_c2h2_int[i] = f_c2h2_lay[i];
        f_tio_int[i] = f_tio_lay[i];
        f_vo_int[i] = f_vo_lay[i];
		f_h2_int[i] = f_h2_lay[i];
		f_he_int[i] = f_he_lay[i];
		meanmolmass_int[i] = meanmolmass_lay[i];
	}
	if (i == numinterfaces - 1) {
		f_h2o_int[i] = f_h2o_lay[i-1];
				f_co2_int[i] = f_co2_lay[i-1];
				f_co_int[i] = f_co_lay[i-1];
				f_ch4_int[i] = f_h2o_lay[i-1];
				f_nh3_int[i] = f_nh3_lay[i-1];
				f_hcn_int[i] = f_hcn_lay[i-1];
				f_c2h2_int[i] = f_c2h2_lay[i-1];
                f_tio_int[i] = f_tio_lay[i-1];
                f_vo_int[i] = f_vo_lay[i-1];
				f_h2_int[i] = f_h2_lay[i-1];
				f_he_int[i] = f_he_lay[i-1];
				meanmolmass_int[i] = meanmolmass_lay[i-1];
	}
}


// interpolate layer and interface opacities from opacity table
__global__ void opac_mol_mixed_interpol(
		utype*  temp, 
		utype*  opactemp, 
		utype*  press, 
		utype*  opacpress,
		utype*  opac_k,
		utype*  k_h2o,
		utype*  k_co2,
		utype*  k_co,
		utype*  k_ch4,
		utype*  k_nh3,
		utype*  k_hcn,
		utype*  k_c2h2,
        utype*  k_tio,
        utype*  k_vo,
		utype*  k_cia_h2h2,
		utype*  k_cia_h2he,
		utype*  opac_h2o,
		utype*  opac_co2,
		utype*  opac_co,
		utype*  opac_ch4,
		utype*  opac_nh3,
		utype*  opac_hcn,
		utype*  opac_c2h2,
        utype*  opac_tio,
        utype*  opac_vo,
		utype*  opac_cia_h2h2,
		utype*  opac_cia_h2he,
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

			opac_h2o[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_h2o[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_h2o[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_h2o[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_h2o[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
            
			opac_co2[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_co2[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_co2[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_co2[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_co2[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
			opac_co[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_co[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_co[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_co[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_co[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
			opac_ch4[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_ch4[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_ch4[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_ch4[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_ch4[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
			opac_nh3[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_nh3[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_nh3[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_nh3[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_nh3[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
			opac_hcn[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_hcn[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_hcn[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_hcn[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_hcn[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
			opac_c2h2[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_c2h2[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_c2h2[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_c2h2[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_c2h2[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
            opac_tio[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_tio[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_tio[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_tio[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_tio[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
            opac_vo[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_vo[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_vo[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_vo[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_vo[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
			opac_cia_h2h2[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_cia_h2h2[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_cia_h2h2[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_cia_h2h2[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_cia_h2h2[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
			opac_cia_h2he[y+ny*x + ny*nbin*i] = bilin_interpol_func(
					k_cia_h2he[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown], 
					k_cia_h2he[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown], 
					k_cia_h2he[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup],
					k_cia_h2he[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup],
					p, t, pdown, pup, tdown, tup
			);
		}
	}
}


// combine the individual molecular opacities to layer/interface opacities
__global__ void comb_opac(
    utype*  f_h2o_lay_or_int, 
    utype*  f_co2_lay_or_int,
    utype*  f_co_lay_or_int,
    utype*  f_ch4_lay_or_int, 
    utype*  f_nh3_lay_or_int, 
    utype*  f_hcn_lay_or_int,
    utype*  f_c2h2_lay_or_int,
    utype*  f_tio_lay_or_int,
    utype*  f_vo_lay_or_int,
    utype*  f_h2_lay_or_int, 
    utype*  f_he_lay_or_int,
    utype*  opac_h2o_lay_or_int,
    utype*  opac_co2_lay_or_int,
    utype*  opac_co_lay_or_int,
    utype*  opac_ch4_lay_or_int,
    utype*  opac_nh3_lay_or_int,
    utype*  opac_hcn_lay_or_int,
    utype*  opac_c2h2_lay_or_int,
    utype*  opac_tio_lay_or_int,
    utype*  opac_vo_lay_or_int,
    utype*  opac_cia_h2h2_lay_or_int,
    utype*  opac_cia_h2he_lay_or_int,
    utype*  opac_wg_lay_or_int,
    utype*  meanmolmass_lay_or_int,
    int     ny,
    int     nbin,
    int     nlay_or_nint
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < nbin && i < nlay_or_nint){
        
        for(int y=0;y<ny;y++){
            
            opac_wg_lay_or_int[y+ny*x+ny*nbin*i] = f_h2o_lay_or_int[i] * M_H2O/meanmolmass_lay_or_int[i] * opac_h2o_lay_or_int[y+ny*x+ny*nbin*i]
            + f_co2_lay_or_int[i] * M_CO2/meanmolmass_lay_or_int[i] * opac_co2_lay_or_int[y+ny*x+ny*nbin*i]
            + f_co_lay_or_int[i] * M_CO/meanmolmass_lay_or_int[i] * opac_co_lay_or_int[y+ny*x+ny*nbin*i]
            + f_ch4_lay_or_int[i] * M_CH4/meanmolmass_lay_or_int[i] * opac_ch4_lay_or_int[y+ny*x+ny*nbin*i]
            + f_nh3_lay_or_int[i] * M_NH3/meanmolmass_lay_or_int[i] * opac_nh3_lay_or_int[y+ny*x+ny*nbin*i]
            + f_hcn_lay_or_int[i] * M_HCN/meanmolmass_lay_or_int[i] * opac_hcn_lay_or_int[y+ny*x+ny*nbin*i]
            + f_c2h2_lay_or_int[i] * M_C2H2/meanmolmass_lay_or_int[i] * opac_c2h2_lay_or_int[y+ny*x+ny*nbin*i]
            + f_tio_lay_or_int[i] * M_TIO/meanmolmass_lay_or_int[i] * opac_tio_lay_or_int[y+ny*x+ny*nbin*i]  
            + f_vo_lay_or_int[i] * M_VO/meanmolmass_lay_or_int[i] * opac_vo_lay_or_int[y+ny*x+ny*nbin*i]  
            + f_h2_lay_or_int[i] * f_h2_lay_or_int[i] * M_H2/meanmolmass_lay_or_int[i] * opac_cia_h2h2_lay_or_int[y+ny*x+ny*nbin*i]
            + f_h2_lay_or_int[i] * f_he_lay_or_int[i] * M_HE/meanmolmass_lay_or_int[i] * opac_cia_h2he_lay_or_int[y+ny*x+ny*nbin*i];

        }
    }
}

// combine the individual molecular opacities to layer/interface opacities
__global__ void comb_scat_cross(
    utype* f_h2_lay_or_int,
    utype* k_scat_cross,
    utype* scat_cross_lay_or_int,
    int nbin,
    int nlay_or_nint
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < nbin && i < nlay_or_nint){
        
        scat_cross_lay_or_int[x+nbin*i] = f_h2_lay_or_int[i] * k_scat_cross[x];
    }
}
