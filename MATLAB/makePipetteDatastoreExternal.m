%% Create a directory of preprocessed images for neural net training
%{ 
This script will read from folderPath and its subfolders, preprocess
based on settings input by user, and save the processed images in a 
separate folder so that training can use a table and reduce required RAM
for training. It will also save the resulting table of shuffled images
for training and validation as a mat file to be easily loaded later on.

NOTE: Make sure paths have \ at the end

Mercedes Gonzalez. 

%}
clear all; clc;
tic
%% Settings for preprocessing
MIRROR_LR = true;           % mirror images across the vertical axis
MIRROR_UD = true;           % mirror images across the horizontal axis
MIRROR_LRUD = true;         % do both LR and UD mirroring
IMG_SIZE = [224 224 ];      % standard image size for nasnetlarge() is [331 331 3]
PERC_VAL = 1;    % number between 0 and 1
letter = { 'a' , 'b', 'c', 'd', 'e', 'f', 'g','h','i','j','k','l','m','n','o','p','q'}; % to be used for folder identification since some files can have the same name across days
LAB_RIG = false; % set true if using lab rig for path definition
SLICEVAL = true;
if LAB_RIG
    RAW_ROOT_PATH = 'C:\Users\myip7\Dropbox (GaTech)\Shared folders\Pipette and cell finding\2019-2020 NET\CNN LabVIEW\multibot\';
    NEW_ROOT = 'C:\Users\myip7\Dropbox (GaTech)\Shared folders\Pipette and cell finding\2019-2020 NET\Training and Validation Data\';
else % on mercedes laptop
    RAW_ROOT_PATH =  'C:\Users\mgonzalez91\Dropbox (GaTech)\Research\PBL\Pipette Detection\Pipette and cell finding\2019-2020 NET\CNN LabVIEW\multibot\TEST_IMAGES\';
    NEW_ROOT = "C:\Users\mgonzalez91\Dropbox (GaTech)\Research\PBL\Pipette Detection\Pipette and cell finding\2019-2020 NET\Training and Validation Data\test\";
end

% Create folder for the day and subfolders for validation and training
today_folder = strcat(string(date),'-data');
today_path = strcat(NEW_ROOT,today_folder);

mkdir(NEW_ROOT,today_folder)

% Name mat file name to save table later on
MATfilename = strcat('pipetteXYZ-table-',string(date),'.mat')

fprintf('Settings set successfully. Folders created. \n')
%% Count subfolders in folder_path
% get all folders inside folderPath then count the folders. In addition,
% determine the name of all subfolders to be used later.
S = dir(RAW_ROOT_PATH);
n_folders = sum([S(~ismember({S.name},...
    {'.','..','.dropbox','desktop.ini','.DS_Store','Icon','trainingImages'})).isdir]);
subfolder_info = S(~ismember({S.name},...
    {'.','..','.dropbox','desktop.ini','.DS_Store','Icon','trainingImages'}));
info_idx = 1;
%% Set up environment to read images
% add the folder and its sub folders to the search path
addpath(genpath(RAW_ROOT_PATH)) 

% define the accepted file extensions to search for.
file_ext = {'.png','.jpg','.tif'};

% create an image datastore for raw images
raw_imds = imageDatastore(RAW_ROOT_PATH,...
    'FileExtensions',file_ext,...
    'Includesubfolders',true);

%% create image labels
% read the name of the folder. use this to find the name of the log file.

for subfolder_idx = 1:n_folders
    logname = subfolder_info(subfolder_idx).name;
    pathname = subfolder_info(subfolder_idx).folder;
    temp_logname = [logname '.txt'];
    fprintf('reading from %s \n',temp_logname)
    log = tdfread(temp_logname);
    fprintf('Subfolder %i of %i\n',subfolder_idx,n_folders)
    %% Augmentation
    for img_idx = 1:length(log.img)
        % get the image filename from the log
        filename = num2str(log.img(img_idx));
        abs_filepath = [pathname '\' logname '\' filename '.png'];

        % now we append the image and data if the file should be appended
        if exist(abs_filepath, 'file')==2
            I = customPreprocess(imread(abs_filepath),IMG_SIZE);
            
                img_name = strcat(letter{subfolder_idx},'-',filename,'.png');
                temp_path = fullfile(today_path,img_name);
                imwrite(I, temp_path);
                info(info_idx).filepath = temp_path; 
                info(info_idx).xyz = [log.x(img_idx) log.y(img_idx) log.z(img_idx)];
                info_idx = info_idx + 1; 
            
            if MIRROR_LR
                I_lr = fliplr(I);  % flip the image
                lr_filename = strcat(letter{subfolder_idx},'-',filename,'-LR.png');
                temp_path = fullfile(today_path,lr_filename);
                imwrite(I_lr, temp_path);
                info(info_idx).filepath = temp_path; 
                info(info_idx).xyz = [-log.x(img_idx) log.y(img_idx) log.z(img_idx)];
                info_idx = info_idx + 1; 
            end
            
            if MIRROR_UD
                I_ud = flipud(I);  % flip the image
                ud_filename = strcat(letter{subfolder_idx},'-',filename,'-UD.png');
                temp_path = fullfile(today_path,ud_filename);
                imwrite(I_ud, temp_path);
                info(info_idx).filepath = temp_path; 
                info(info_idx).xyz = [log.x(img_idx) -log.y(img_idx) log.z(img_idx)];
                info_idx = info_idx + 1; 
            end
            
            if MIRROR_LRUD
                I_lrud = flipud(fliplr(I));  % flip the image
                lrud_filename = strcat(letter{subfolder_idx},'-',filename,'-LRUD.png');
                temp_path = fullfile(today_path,lrud_filename);
                imwrite(I_lrud, temp_path);
                info(info_idx).filepath = temp_path; 
                info(info_idx).xyz = [-log.x(img_idx) -log.y(img_idx) log.z(img_idx)];
                info_idx = info_idx + 1; 
            end
        end
    end
    
end

n_images = length(info);
n_valid = round(n_images*PERC_VAL);
n_train = n_images-n_valid;
%% Make training and validation tables
if SLICEVAL % validation images are slice only. 
    % Assign slice only images to be validation. Randomize. 
    ordered_train = linspace(1,n_train,n_train);
    random_train = ordered_train(randperm(n_train));
    train_only = random_train(1:n_train)';
    
    ordered_val = linspace(n_train+1,n_images,n_valid);
    random_val = ordered_val(randperm(n_valid));
    val_only = random_val(1:n_valid)';
else % all random everything     
    ordered = linspace(1,n_images,n_images);
    random = ordered(randperm(n_images));
    val_only = random(1:n_valid)';
    train_only = random(n_valid+1:end)';
end

% create structs to easily transform to table
for v = 1:n_valid
    val(v).file = info(val_only(v)).filepath;
    val(v).x = info(val_only(v)).xyz(1);
    val(v).y = info(val_only(v)).xyz(2);
    val(v).z = info(val_only(v)).xyz(3);

end
for t = 1:n_train
    train(t).file = info(train_only(t)).filepath;
    train(t).x = info(train_only(t)).xyz(1);
    train(t).y = info(train_only(t)).xyz(2);
    train(t).z = info(train_only(t)).xyz(3);
end

% create tables
val_data = struct2table(val);
train_data = struct2table(train);

% create augmented datastore from table
val_imds = augmentedImageDatastore(IMG_SIZE,...
    val_data,...
    'OutputSizeMode','centercrop',...
    'ColorPreprocessing','gray2rgb');
train_imds = augmentedImageDatastore(IMG_SIZE,...
    train_data,...
    'OutputSizeMode','centercrop',...
    'ColorPreprocessing','gray2rgb');
    
    %% report and save as table
    fprintf('\nPROCESSING COMPLETE\n')
    fullmatfilename = fullfile(today_path,MATfilename);
    save(fullmatfilename,'val_data','train_data','val_imds','train_imds','info') 
    fprintf('\nDatastores saved. Fin.\n')

    time_to_prepare = toc
