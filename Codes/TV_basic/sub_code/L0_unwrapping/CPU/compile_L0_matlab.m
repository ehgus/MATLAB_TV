clc;
cd('C:\Users\Administrator\Desktop\HERVE\L0_unwrapping\CPU');
%mex -output quality_guided_unwrapp matlab_quality_guided.cpp dxdygrad.c extract.c getqual.c grad.c list.c mainqual.c maskfat.c quality.c qualgrad.c qualpseu.c qualvar.c util.c ;

mex -output L0_unwrapp_cpu  congruen.cpp dct.cpp dxdygrad.cpp getqual.cpp grad.cpp histo.cpp laplace.cpp lpnorm.cpp maskfat.cpp matlab_L0_unwrapp.cpp pcg.cpp qualgrad.cpp qualpseu.cpp qualvar.cpp raster.cpp residues.cpp solncos.cpp util.cpp  ;

