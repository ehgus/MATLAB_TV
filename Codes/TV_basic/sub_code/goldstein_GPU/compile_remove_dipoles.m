clc;
cd(fileparts(matlab.desktop.editor.getActiveFilename));
mexcuda -output remove_dipoles_gpu remove_dipoles_cuda.cu remove_dipoles_mex.cpp