% {
bg =single( (loadTIFF('C:\Users\Administrator\Desktop\HERVE\__MULTIPLE_scattering\datas\raw_bg.tiff')));
%bg=bg(:,:,1:300);
sp =single( (loadTIFF('C:\Users\Administrator\Desktop\HERVE\__MULTIPLE_scattering\datas\raw_tomo78.tiff')));
%sp=sp(:,:,1:300);
%}
%%
% {
imsize=size(bg,1);
lambda=0.457; % wavelength
pixel_size=5.5;
M=55.555;
NA=0.75;
res1 = pixel_size/M;%(=camera pixel size in um);
% spatial frequency resolution by field of view
kres1 = 1/(imsize*pixel_size/M);%(=Fourier space pixel size in um^(-1));
NAlimit_radius = NA/lambda/kres1;
NA_mask =  single(~mk_ellipse(NAlimit_radius, NAlimit_radius, imsize, imsize));

field=QPI_GPU(sp,bg,NA_mask,NAlimit_radius);

wrapped_phase=angle(single(gpuArray(field)));

%}
%%
randge=200:300;
plan=Unwrap_Fourier_GPU_plan(size(field(:,:,randge)));
phase=Unwrap_Fourier_GPU(single(gpuArray(field(:,:,randge))),plan);




