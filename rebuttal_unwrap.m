%% function loading
clc;clear;
use_GPU=true;
MULTI_GPU=false;
current_dir=fileparts(matlab.desktop.editor.getActiveFilename);
code_dir=fullfile(current_dir,'Codes');
addpath(genpath(code_dir));

%% set folder
[fdata, RIdata] = load_meta_data();
sample_dir_set={...
    fdata.Yeast.x0x_1,...
    fdata.Yeast.x0x_2,...
    fdata.Yeast.x12x_1,...
    fdata.Yeast.x12x_2...
};
RI_bg_set=[RIdata.x0x RIdata.x0x RIdata.x12x RIdata.x12x];
%%
for i=1:length(RI_bg_set)
    sample_dir = sample_dir_set{i};
    if ~isfolder(fullfile(sample_dir,'data3d\000000PNG'))
        mkdir(fullfile(sample_dir,'data3d\000000PNG'));
        mkdir(fullfile(sample_dir,'bgImagesPNG')); 
    end
    unlock_tomocube_data(fullfile(sample_dir,'data3d\000000\images.dat'),fullfile(sample_dir,'data3d\000000PNG'))
    unlock_tomocube_data(fullfile(sample_dir,'bgImages\calibration.dat'),fullfile(sample_dir,'bgImagesPNG'))
end

%% Set parameters for TV algorithm
%변수관리 개판
% field retrieval paramters
params=struct;
params.NA=1.2;
%params.RI_bg=1.3355;
params.wavelength=0.532;
params.resolution=[1 1 1]*params.wavelength/4/params.NA;
params.resolution=[1 1 2]*params.wavelength/4/params.NA;
params.vector_simulation=false;
params.size=[0 0 150];
params.use_GPU = use_GPU;


for i=1:length(RI_bg_set)
    sample_dir = sample_dir_set{i};
    bg_file=fullfile(sample_dir,'bgImagesPNG');
    sp_file=fullfile(sample_dir,'data3d/000000PNG');
    params.RI_bg=RI_bg_set(i);
    %field retrieval
    mat_file=fullfile(sample_dir,'field.mat');
    

    field_retrieval_params=params;
    field_retrieval_params.RI_bg=RI_bg_set(i);
    field_retrieval_params.resolution_image=[1 1]*(4.8/58.33);
    field_retrieval_params.conjugate_field=true;
    field_retrieval_params.use_abbe_correction=true;
    field_retrieval_params.verbose = false;

    field_retrieval=FIELD_EXPERIMENTAL_RETRIEVAL(field_retrieval_params);
    [input_field,output_field]=load_data(bg_file,sp_file);
    [input_field,output_field,rytov_params]=field_retrieval.get_fields(input_field,output_field);
    [bg,sp]=vector2scalarfield(input_field,output_field);
    save(mat_file,'bg','sp');
    
    Phase_recon = true;
    if Phase_recon
        Phase=angle(sp(:,:,1)./bg(:,:,1));
        Phase=gather(unwrapp2_gpu(gpuArray(single(Phase))));
        L0Phase=gather(L0_unwrap(single(Phase)));
        RelibilityPhase=unwrap_phase(single(Phase));
        deterPhase = unwrap2_deterministic(single(Phase));
        save(fullfile(sample_dir,'normal_phase.mat'),'Phase','L0Phase','RelibilityPhase','deterPhase');
        %continue
    end
end

