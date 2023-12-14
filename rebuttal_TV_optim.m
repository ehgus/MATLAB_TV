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
    fdata.Bead.x60x,...
    fdata.Yeast.x12x_1,...
    fdata.Celegans.x20x...
};
RI_bg_set=[RIdata.x60x RIdata.x12x RIdata.x20x];
sample_name=['bead','yeast','elegans'];

%% Set parameters for TV algorithm
params=struct;
params.NA=1.2;
params.wavelength=0.532;
params.resolution=[1 1 1]*params.wavelength/4/params.NA;
params.vector_simulation=false;
params.size=[0 0 100];
params.use_GPU = use_GPU;

for i=1:1%length(RI_bg_set)
    %data loading
    sample_dir = sample_dir_set{i};
    bg_file=fullfile(sample_dir,'bgImagesPNG');
    sp_file=fullfile(sample_dir,'data3d/000000PNG');
    params.RI_bg=RI_bg_set(i);
    %field retrieval
    field_retrieval_params=params;
    field_retrieval_params.RI_bg=RI_bg_set(i);
    field_retrieval_params.resolution_image=[1 1]*(4.8/58.33);
    field_retrieval_params.conjugate_field=true;
    field_retrieval_params.use_abbe_correction=true;

    field_retrieval=FIELD_EXPERIMENTAL_RETRIEVAL(field_retrieval_params);
    [input_field,output_field]=load_data(bg_file,sp_file);
    [input_field,output_field,rytov_params]=field_retrieval.get_fields(input_field,output_field);
    [bg,sp]=vector2scalarfield(input_field,output_field);
    
    rytov_params.use_non_negativity=true;
    rytov_params.non_negativity_iteration=40;
    rytov_solver=BACKWARD_SOLVER_RYTOV(rytov_params);
    [RI_rytov,ORytov]=rytov_solver.solve(input_field,output_field);
    RI_rytov = real(RI_rytov);
    if rytov_params.use_non_negativity
        mat_file = fullfile(sample_dir,'RI_nneg.mat');
    else
        mat_file = fullfile(sample_dir,'RI_rytov.mat');
    end
    save(mat_file,'RI_rytov');

    clear input_field
    clear output_field
    %RI_rytov=real(RI_rytov); % ignore absorbance part
    mask=ORytov ~= 0;
    clear ORytov
    %Total variance
    tv_params=TV.get_default_parameters();
    tv_params.use_non_negativity=true;
    tv_params.outer_itterations=40;
    tv_params.inner_itterations=20;
    tv_params.min_real = params.RI_bg;

    for TV_strength = [0.005,0.05,0.07,0.1,0.3,0.5,5]
        if tv_params.use_non_negativity 
            mat_file=fullfile(sample_dir,sprintf('TV_RI_nneg strength_%0.3f.mat',TV_strength));
        else
            mat_file=fullfile(sample_dir,sprintf('TV_RI strength_%0.3f.mat',TV_strength));
        end
        disp(mat_file)
        if isfile(mat_file)
            %continue
        end
        
        tv_params.TV_strength=TV_strength;
        pot2RI=@(pot) single(RI_bg_set(i)*sqrt(1+pot./((2*pi*params.RI_bg/params.wavelength).^2)));
        RI2pot=@(RI)  single((2*pi*RI_bg_set(i)/params.wavelength)^2*(RI.^2/RI_bg_set(i)^2-1));
    
        regulariser=TV(tv_params);
        data=single(RI2pot(RI_rytov));
        data=regulariser.solve(data,mask);
        RI_TV=pot2RI(data);
        %store TV result
        save(mat_file,'RI_TV');
    end
end
