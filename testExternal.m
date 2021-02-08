filepath = "C:\Users\mgonzalez91\Dropbox (GaTech)\Research\PBL\Pipette Detection\Pipette and cell finding\2019-2020 NET\CNN LabVIEW\multibot\TEST_IMAGES\20201117_cnn_test_dataset\";

% load('C:\Users\mgonzalez91\Dropbox (GaTech)\Research\PBL\Pipette Detection\Pipette and cell finding\2019-2020 NET\18-Jun-2020-datastores\regressionNET-11-Mar-2020.mat')

guess = zeros(length(file_list),3);
for i = 1:length(file_list)
    f = strcat(string(file_list(i)),'.png');
    I = imread(fullfile(filepath,f));
    guess(i,:) = findCoordsExternal(I,net);
    waitforbuttonpress
    clf
end

csvwrite('output.txt',guess) 