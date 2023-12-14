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
    unlockTomocubeDat(fullfile(sample_dir,'data3d\000000\images.dat'),fullfile(sample_dir,'data3d\000000PNG'))
    unlockTomocubeDat(fullfile(sample_dir,'bgImages\calibration.dat'),fullfile(sample_dir,'bgImagesPNG'))
end


%% Set parameters for TV algorithm
%변수관리 개판
% field retrieval paramters
params=struct;
params.NA=1.2;
params.wavelength=0.532;
params.resolution=[1 1 1]*params.wavelength/4/params.NA;
params.vector_simulation=false;
params.size=[0 0 100];
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
    field_retrieval_params.conjugate_field=true;
    field_retrieval_params.use_abbe_correction=true;
    
    field_retrieval=FIELD_EXPERIMENTAL_RETRIEVAL(field_retrieval_params);
    [input_field,output_field]=load_data(bg_file,sp_file);
    [input_field,output_field,rytov_params]=field_retrieval.get_fields(input_field,output_field);
    [bg,sp]=vector2scalarfield(input_field,output_field);
    save(mat_file,'bg','sp');
    
    Phase_recon = false;
    if Phase_recon
        Phase=angle(sp(:,:,1)./bg(:,:,1));
        Phase=gather(unwrapp2_gpu(gpuArray(single(Phase))));
        save(fullfile(sample_dir,'normal_phase.mat'),'Phase');
        %continue
    end
    
    %3D recontruciton taking advantage of Rytov approximation
    rytov_params.use_non_negativity=false;
    rytov_params.non_negativity_iteration=100;
    rytov_solver=BACKWARD_SOLVER_RYTOV(rytov_params);
    [RI_rytov,ORytov]=rytov_solver.solve(input_field,output_field);
    clear input_field
    clear output_field
    mask=ORytov ~= 0;
    clear ORytov
    %Total variance
    

    tv_params=TV.get_default_parameters();
    tv_params.use_non_negativity=true;
    tv_params.outer_itterations=100;
    tv_params.inner_itterations=50;
    tv_params.min_real = params.RI_bg;
    tv_params.TV_strength=0.1;
    if tv_params.use_non_negativity
        fstr = 'TV_RI_nneg strength_%0.3f.mat';
    else
        fstr = 'TV_RI strength_%0.3f.mat';
    end
    mat_file=fullfile(sample_dir,sprintf(fstr,tv_params.TV_strength));
    if isfile(mat_file)
        %continue
    end

    pot2RI=@(pot) single(RI_bg_set(i)*sqrt(1+pot./((2*pi*params.RI_bg/params.wavelength).^2)));
    RI2pot=@(RI)  single((2*pi*RI_bg_set(i)/params.wavelength)^2*(RI.^2/RI_bg_set(i)^2-1));

    regulariser=TV(tv_params);
    data=single(RI2pot(RI_rytov));
    data=regulariser.solve(data,mask);
    
    RI_TV=pot2RI(data);
    %store TV result
    save(mat_file,'RI_TV','RI_rytov');
end


%%
function unlockTomocubeDat(srcFile,dstDir)
    ImgCnt = length(h5info(srcFile,'/images').Datasets);
    for i = 0:ImgCnt-1
        h5loc = sprintf('/images/%06d',i);
        ImgBinary = h5read(srcFile,h5loc);
        dstFile = fullfile(dstDir,sprintf('%06d.png',i));
        fileID = fopen(dstFile,'w');
        fwrite(fileID,ImgBinary);
        fclose(fileID);
    end
end


