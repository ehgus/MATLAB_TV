/**
 * @file pctdemo_life_mex_texture.cpp
 * @brief MEX gateway for a stencil operation.
 * Copyright 2013 The MathWorks, Inc.
 *
 */

#include "tmwtypes.h"
#include "mex.h"
#include <cstring>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include "file.h"
#include "histo.h"
#include "pi.h"
#include "getqual.h"
#include "congruen.h"
#include "maskfat.h"
#include "util.h"
#include "getqual.h"
#include "lpnorm.h"
#define BORDER   0x20
#define DEFAULT_NUM_ITER  30 //20
#define DEFAULT_PCG_ITER  30  //20
#define DEFAULT_E0        0.0001 //0.001


#define BORDER (0x20)
 /**
  * MEX gateway
  */
mwSize get_1D_array_size(const mwSize dim_size, const mwSize* dims) {

	if (dim_size == 1) {
		return dims[0];
	}
	else if (dim_size == 2) {
		if (dims[0] == 1) {
			return dims[1];
		}
		else if (dims[1] == 1) {
			return dims[0];
		}
		else {
			return 0;
		}
	}
	else {
		return 0;
	}
}

void mexFunction(int  nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "lapjv:InvalidInput";
	char const * const errMsg = "Provide -input matrix- as type -single- to MEX file.";
	char const * const errMsg2 = "All array must be 2D or 3D";
	char const * const errMsg3 = "Could not return";
	char const * const errMsg4 = "Phase and quality maps must be same sizes";
	char const * const errMsg5 = "Both inputs must be real";

	int k;

	if (nrhs != 1) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}

	// retrieve the inputs

	mxArray * const Phase = mxDuplicateArray(prhs[0]);
	if (!mxIsSingle(Phase)) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}
	if (mxIsComplex(Phase)) {
		mexErrMsgIdAndTxt(errId, errMsg5);
	}
	float * const pPhase = static_cast<float *>(mxGetData(Phase));
	
	const mwSize *dim_array_Phase; mwSize dims_Phase;
	dims_Phase = mxGetNumberOfDimensions(Phase);
	dim_array_Phase = mxGetDimensions(Phase);
	if (!(dims_Phase == 2 || dims_Phase == 3)) {
		mexErrMsgIdAndTxt(errId, errMsg2);
	}


	int  Phase_sizes[3] = { (int)dim_array_Phase[0], (int)dim_array_Phase[1], (int)1 };
	if (dims_Phase == 3) {
		Phase_sizes[2] = (int)dim_array_Phase[2];
	}

	//unwrapp
	
	for (int matt_num = 0; matt_num < Phase_sizes[2]; ++matt_num) {

		int xsize = Phase_sizes[0];
		int ysize = Phase_sizes[1];
		int            i, j, k, n;
		FILE           *ifp, *ofp, *mfp = 0, *qfp = 0;
		float          *phase;     //inited //free
		float          *soln;      //inited //free
		float          *qual_map;  //inited //free
		float          *rarray;    //inited //free
		float          *zarray;    //inited //free
		float          *parray;    //inited //free
		float          *dxwts;     //inited //free
		float          *dywts;     //inited //free
		unsigned char  *bitflags;  //inited //free
		double         *xcos, *ycos;
		char           buffer[200], tempstr[200];
		char           infile[200], outfile[200];
		char           bmaskfile[200], qualfile[200];
		char           format[200], modekey[200];
		int            in_format, debug_flag;
		//int            xsize, ysize;   /* defined out of the loop */
		int            xsize_actual, ysize_actual, xsize_dct, ysize_dct;
		int            avoid_code, thresh_flag, fatten, tsize;
		int            num_iter = DEFAULT_NUM_ITER;
		int            pcg_iter = DEFAULT_PCG_ITER;
		double         rmin, rmax, rscale, e0 = DEFAULT_E0;
		double         one_over_twopi = 1.0 / TWOPI;
		UnwrapMode     mode;

		strcpy(modekey, "none");
		strcpy(qualfile, "none");
		tsize = 1;
		strcpy(bmaskfile, "none");
		debug_flag = 1;
		thresh_flag = 0;
		fatten = 0;
		//num_iter = 1;
		//pcg_iter = 1;
		strcpy(qualfile, "none");
		mode = (UnwrapMode)SetQualityMode(modekey, qualfile, 1);

		/* Increase dimensions to power of two (plus one) */
		xsize_actual = xsize;
		ysize_actual = ysize;
		for (xsize_dct = 1; xsize_dct + 1 < xsize_actual; xsize_dct *= 2)
			;
		xsize_dct += 1;
		for (ysize_dct = 1; ysize_dct + 1 < ysize_actual; ysize_dct *= 2)
			;
		ysize_dct += 1;
		if (xsize_dct != xsize_actual || ysize_dct != ysize_actual) {
			mexPrintf("Dimensions increase from %dx%d to %dx%d for FFT's\n",
				xsize_actual, ysize_actual, xsize_dct, ysize_dct);
		}
		xsize = xsize_dct;
		ysize = ysize_dct;

		// iterate on each given phase 
		phase = (float*)malloc(xsize*ysize * sizeof(float));
		soln = (float*)malloc(xsize*ysize * sizeof(float));
		qual_map = (float*)malloc(xsize*ysize * sizeof(float));
		bitflags = (unsigned char*)malloc(xsize*ysize * sizeof(unsigned char));
		//define soln and bitflags
		for (k = 0; k < xsize*ysize; k++)
			qual_map[k] = 255;
		for (k = 0; k < xsize*ysize; k++)
			phase[k] = 0;
		for (k = 0; k < xsize*ysize; k++)
			bitflags[k] = 0;
		for (k = 0; k < xsize*ysize; k++)
			soln[k] = 0;

		//set the phase
		for (j = ysize_dct - 1; j >= 0; j--) {
			for (i = xsize_dct - 1; i >= 0; i--) {
				if (i < xsize_actual && j < ysize_actual)
					phase[j*xsize_dct + i] = pPhase[j*xsize_actual + i + matt_num * xsize_actual*ysize_actual] *one_over_twopi;
				else phase[j*xsize_dct + i] = 0.0;
			}
		}
		for (j = ysize_dct - 1; j >= 0; j--) {
			for (i = xsize_dct - 1; i >= 0; i--) {
				if (i < xsize_actual && j < ysize_actual)
					bitflags[j*xsize_dct + i] = bitflags[j*xsize_actual + i];
				else bitflags[j*xsize_dct + i] = BORDER;
			}
		}
		if (qual_map) {
			for (j = ysize_dct - 1; j >= 0; j--) {
				for (i = xsize_dct - 1; i >= 0; i--) {
					if (i < xsize_actual && j < ysize_actual)
						qual_map[j*xsize_dct + i] = qual_map[j*xsize_actual + i];
					else
						qual_map[j*xsize_dct + i] = 0.0;
				}
			}
		}
		xsize = xsize_dct;
		ysize = ysize_dct;
		/* Allocate more memory */
		rarray = (float*)malloc(xsize*ysize * sizeof(float));
		zarray = (float*)malloc(xsize*ysize * sizeof(float));
		parray = (float*)malloc(xsize*ysize * sizeof(float));
		dxwts = (float*)malloc(xsize*ysize * sizeof(float));
		dywts = (float*)malloc(xsize*ysize * sizeof(float));
		for (k = 0; k < xsize*ysize; k++) {
			rarray[k] = 0;
			zarray[k] = 0;
			parray[k] = 0;
			dxwts[k] = 0;
			dywts[k] = 0;
		}
		//unwrapp

		mexPrintf("Unwrapping...\n");
		mexPrintf("max_iter %d pcg_max_iter %d \n", num_iter, pcg_iter);
		mexEvalString("drawnow;");
		LpNormUnwrap(soln, phase, dxwts, dywts, bitflags, qual_map,
			rarray, zarray, parray, num_iter, pcg_iter, e0, xsize, ysize);
		mexPrintf("\nFinished\n");

		//copy result 
		xsize = xsize_actual;
		ysize = ysize_actual;
		for (j = 0; j < ysize; j++) {
			for (i = 0; i < xsize; i++) {
				pPhase[j*xsize + i + matt_num * xsize_actual*ysize_actual] = TWOPI * soln[j*xsize_dct + i];
			}
		}
		//free unused
		free(phase);
		free(soln);
		free(qual_map);
		free(bitflags);

		free(rarray);
		free(zarray);
		free(parray);
		free(dxwts);
		free(dywts);
	}
	//return
	if (nlhs == 1) {
		plhs[0] = Phase;
	}
	else {
		mexErrMsgIdAndTxt(errId, errMsg3);
	}

}

