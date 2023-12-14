%% function loading
clc;clear;
use_GPU=true;
MULTI_GPU=false;
current_dir=fileparts(matlab.desktop.editor.getActiveFilename);
code_dir=fullfile(current_dir,'Codes');
addpath(genpath(code_dir));

%% set folder
head_dir = 'C:\Users\labdo\Desktop\02_project-OptiPrep\21_data\07 BG';

params=struct;
params.NA=1.2;
params.wavelength=0.532;
params.resolution=[1 1 3]*params.wavelength/4/params.NA;
params.vector_simulation=false;
params.size=[0 0 150];
params.use_GPU = use_GPU;

bg_file=fullfile(head_dir,'IX0bgImagesPNG');
sp_file=fullfile(head_dir,'IX10bgImagesPNG');
field_retrieval_params=params;
field_retrieval_params.RI_bg=1.333;
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