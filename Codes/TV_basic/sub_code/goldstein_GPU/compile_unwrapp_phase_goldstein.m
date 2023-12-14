clc;
cd(fileparts(matlab.desktop.editor.getActiveFilename));
mexcuda -output unwrapp_phase_goldstein_gpu unwrapp_phase_goldstein_cuda.cu unwrapp_phase_goldstein_mex.cpp