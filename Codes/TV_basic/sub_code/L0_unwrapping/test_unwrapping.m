%{
%field_file='Q:\shin_cell_phantom\shin_cell_phantom\shin_pol_setup_cell_phantom_data.mat';
field_file='Q:\shin_test\retFields_sample2_Cam1s.mat';
load(field_file);
%}
%%

%{
wrapped_phase=single(gpuArray(retPhase));
wrapped_phase=mod(wrapped_phase+pi,2*pi)-pi;
%}
%wrapped_phase=100*single(gpuArray(rand(500,500,500)));

%close all
%%
% {
base='D:\OLD DATA\big_tissue\R4';%'Q:\BLOOD_CLOT\RAW\Raw_2';
tomo_num=32;%22;
bg =single( (loadTIFF([ base '\raw_bg.tiff'])));
bg=bg(:,:,1:10);
sp =single( (loadTIFF([ base '\raw_tomo' num2str(tomo_num) '.tiff'])));
sp=sp(:,:,1:10);
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

wait(gpuDevice());tic; 
%wrapped_phase(:,:,90)=1;
%for ii=1:100
unwrapped_phase = unwrapp2_gpu(wrapped_phase);
%end
wait(gpuDevice());toc;


%[residue_idx,residue_value] = get_residu_position_values(residue_map);
%%
plan = fdct2_prep(size(wrapped_phase),true);
%profile on;
wait(gpuDevice());tic;
pcg_phase=PCG_unwrap((wrapped_phase),ones(size(wrapped_phase),'single','gpuArray'), 30, 0.0001,plan);
wait(gpuDevice());toc;
%profile off;
%profile viewer;
display_vid_fun_simple(gather(pcg_phase))
%%
wait(gpuDevice());tic;
l0_phase=L0_unwrap(wrapped_phase );
wait(gpuDevice());toc;

display_vid_fun_simple(gather(l0_phase));

%%
%{
cpu_wrapped_phase=gather(wrapped_phase);
tic;
cpu_phase=L0_unwrapp_cpu((cpu_wrapped_phase(:,:,:)));
toc;
display_vid_fun_simple(gather(cpu_phase))

%tic;
%cpu_phase2=L0_unwrapp_qual_cpu((cpu_wrapped_phase(:,:,:)));
%toc;
%display_vid_fun_simple(gather(cpu_phase2))
%%
tic;
cpu_phase2=unwrap2(double(cpu_wrapped_phase));
toc;
%{
tic;
for ii=1: size(cpu_wrapped_phase,3)
    cpu_wrapped_phase(:,:,ii)=single(unwrap2(double(cpu_wrapped_phase(:,:,ii))));
end
toc;
%}
%}
