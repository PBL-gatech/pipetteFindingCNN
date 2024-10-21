tic
%{
    This script is used to test the pipette regression analysis data. 
    Make sure val_imds, val_data, and net are variables in the wksp
    
    Colby Lewallen. August 2018
    Updated Mercedes Gonzalez June 2020

%} 

marker_size = 9; % size of marker
c1 = 'cyan'; % Real position
c2 = 'red'; % Guess
SHOW_ERRORS = true; % supress plotting errors
new_dimension = 224; 

% use the CNN to guess the position <x,y,z> of the pipette
guess = predict(net,test_imds);

% calculate the <dx,dy,dz> for each image
dx = test_data.xyz(:,1) - guess(:,1);
dy = test_data.xyz(:,2) - guess(:,2);
dz = test_data.xyz(:,3) - guess(:,3);

% Convert from pixels to steps to um
dx_um = dx*0.1/1.093;
dy_um = dy*0.1/1.093;
dz_um = dz*0.1/1.093;

% average the distance from the expected value
xerror = mean(abs(dx_um));
yerror = mean(abs(dy_um));
zerror = mean(abs(dz_um));
xstd = std(abs(dx_um));
ystd = std(abs(dy_um));
zstd = std(abs(dz_um));
fprintf('Average\ndx: %1.2f microns\ndy: %1.2f microns\ndz: %1.2f microns\n',xerror,yerror,zerror)
fprintf('Standard Deviation\ndx: %1.2f microns\ndy: %1.2f microns\ndz: %1.2f microns\n',xstd,ystd,zstd)
lowlim = -2;
uplim = 2;
if SHOW_ERRORS 
    %% show worst X errors
    figure()
    [dxMax,dxIdx] = sort(abs(dx));
    originalSize = [1024 1280];
    ysize = originalSize(1);
    xsize = originalSize(2);
    minDimension = min(originalSize);

    subplot(2,2,1)
        I = imread(test_data.file(dxIdx(end)));
        posReal = test_data.xyz(dxIdx(end),1:2);
        posGuess = guess(dxIdx(end),1:2);
        pos = [posReal; posGuess];
        transformedpoint = pos; 
        transformedpoint(:,1) = new_dimension*((pos(:,1)+(xsize/2))-(xsize-minDimension)/2)/minDimension;
        transformedpoint(:,2) = new_dimension*((pos(:,2)+(ysize/2))-(ysize-minDimension)/2)/minDimension;
        Imarker = insertMarker(I,transformedpoint(1,:),'Color',c1,'Size',marker_size);
        hold on
        Imarker2 = insertMarker(Imarker,transformedpoint(2,:),'Color',c2,'Size',marker_size);
        imshow(Imarker2)
        worstX = ['X error: ' num2str(round(dxMax(end)*0.1/1.093),2) 'microns '];
        title(worstX)

    subplot(2,2,2)
        I = imread(test_data.file(dxIdx(end-1)));
        posReal = test_data.xyz(dxIdx(end-1),1:2);
        posGuess = guess(dxIdx(end-1),1:2);
        pos = [posReal; posGuess];
        transformedpoint = pos; 
        transformedpoint(:,1) = new_dimension*((pos(:,1)+(xsize/2))-(xsize-minDimension)/2)/minDimension; 
        transformedpoint(:,2) = new_dimension*((pos(:,2)+(ysize/2))-(ysize-minDimension)/2)/minDimension;
        Imarker = insertMarker(I,transformedpoint(1,:),'Color',c1,'Size',marker_size);
        hold on
        Imarker2 = insertMarker(Imarker,transformedpoint(2,:),'Color',c2,'Size',marker_size);
        imshow(Imarker2)
        worstX = ['X error: ' num2str(round(dxMax(end-1)*0.1/1.093),2) 'microns '];
        title(worstX)

    subplot(2,2,3)
        I = imread(test_data.file(dxIdx(end-2)));
        posReal = test_data.xyz(dxIdx(end-2),1:2);
        posGuess = guess(dxIdx(end-2),1:2);
        pos = [posReal; posGuess];
        transformedpoint = pos; 
        transformedpoint(:,1) = new_dimension*((pos(:,1)+(xsize/2))-(xsize-minDimension)/2)/minDimension; 
        transformedpoint(:,2) = new_dimension*((pos(:,2)+(ysize/2))-(ysize-minDimension)/2)/minDimension;
        Imarker = insertMarker(I,transformedpoint(1,:),'Color',c1,'Size',marker_size);
        hold on
        Imarker2 = insertMarker(Imarker,transformedpoint(2,:),'Color',c2,'Size',marker_size);
        imshow(Imarker2)
        worstX = ['X error: ' num2str(round(dxMax(end-2)*0.1/1.093),2) 'microns '];
        title(worstX)

    subplot(2,2,4)
        I = imread(test_data.file(dxIdx(end-3)));
        posReal = test_data.xyz(dxIdx(end-3),1:2);
        posGuess = guess(dxIdx(end-3),1:2);
        pos = [posReal; posGuess];
        transformedpoint = pos; 
        transformedpoint(:,1) = new_dimension*((pos(:,1)+(xsize/2))-(xsize-minDimension)/2)/minDimension; 
        transformedpoint(:,2) = new_dimension*((pos(:,2)+(ysize/2))-(ysize-minDimension)/2)/minDimension;
        Imarker = insertMarker(I,transformedpoint(1,:),'Color',c1,'Size',marker_size);
        hold on
        Imarker2 = insertMarker(Imarker,transformedpoint(2,:),'Color',c2,'Size',marker_size);
        imshow(Imarker2)
        worstX = ['X error: ' num2str(round(dxMax(end-3)*0.1/1.093),2) 'microns '];
        title(worstX)

    %% show worst Y errors
    figure()
    [dyMax,dyIdx] = sort(abs(dy));

    subplot(2,2,1)
        I = imread(test_data.file(dyIdx(end)));
        posReal = test_data.xyz(dyIdx(end),1:2);
        posGuess = guess(dyIdx(end),1:2);
        pos = [posReal; posGuess];
        transformedpoint = pos; 
        transformedpoint(:,1) = new_dimension*((pos(:,1)+(xsize/2))-(xsize-minDimension)/2)/minDimension; 
        transformedpoint(:,2) = new_dimension*((pos(:,2)+(ysize/2))-(ysize-minDimension)/2)/minDimension;
        Imarker = insertMarker(I,transformedpoint(1,:),'Color',c1,'Size',marker_size);
        hold on
        Imarker2 = insertMarker(Imarker,transformedpoint(2,:),'Color',c2,'Size',marker_size);
        imshow(Imarker2)
        worstX = ['Y error: ' num2str(round(dyMax(end)*0.1/1.093),2) 'microns '];
        title(worstX)

    subplot(2,2,2)
        I = imread(test_data.file(dyIdx(end-1)));
        posReal = test_data.xyz(dyIdx(end-1),1:2);
        posGuess = guess(dyIdx(end-1),1:2);
        pos = [posReal; posGuess];
        transformedpoint = pos; 
        transformedpoint(:,1) = new_dimension*((pos(:,1)+(xsize/2))-(xsize-minDimension)/2)/minDimension; 
        transformedpoint(:,2) = new_dimension*((pos(:,2)+(ysize/2))-(ysize-minDimension)/2)/minDimension;
        Imarker = insertMarker(I,transformedpoint(1,:),'Color',c1,'Size',marker_size);
        hold on
        Imarker2 = insertMarker(Imarker,transformedpoint(2,:),'Color',c2,'Size',marker_size);
        imshow(Imarker2)
        worstX = ['Y error: ' num2str(round(dyMax(end-1)*0.1/1.093),2) 'microns '];
        title(worstX)

    subplot(2,2,3)
        I = imread(test_data.file(dyIdx(end-2)));
        posReal = test_data.xyz(dyIdx(end-2),1:2);
        posGuess = guess(dyIdx(end-2),1:2);
        pos = [posReal; posGuess];
        transformedpoint = pos; 
        transformedpoint(:,1) = new_dimension*((pos(:,1)+(xsize/2))-(xsize-minDimension)/2)/minDimension; 
        transformedpoint(:,2) = new_dimension*((pos(:,2)+(ysize/2))-(ysize-minDimension)/2)/minDimension;
        Imarker = insertMarker(I,transformedpoint(1,:),'Color',c1,'Size',marker_size);
        hold on
        Imarker2 = insertMarker(Imarker,transformedpoint(2,:),'Color',c2,'Size',marker_size);
        imshow(Imarker2)
        worstX = ['Y error: ' num2str(round(dyMax(end-2)*0.1/1.093),2) 'microns '];
        title(worstX)

    subplot(2,2,4)
        I = imread(test_data.file(dyIdx(end-3)));
        posReal = test_data.xyz(dyIdx(end-3),1:2);
        posGuess = guess(dyIdx(end-3),1:2);
        pos = [posReal; posGuess];
        transformedpoint = pos; 
        transformedpoint(:,1) = new_dimension*((pos(:,1)+(xsize/2))-(xsize-minDimension)/2)/minDimension; 
        transformedpoint(:,2) = new_dimension*((pos(:,2)+(ysize/2))-(ysize-minDimension)/2)/minDimension;
        Imarker = insertMarker(I,transformedpoint(1,:),'Color',c1,'Size',marker_size);
        hold on
        Imarker2 = insertMarker(Imarker,transformedpoint(2,:),'Color',c2,'Size',marker_size);
        imshow(Imarker2)
        worstX = ['Y error: ' num2str(round(dyMax(end-3)*0.1/1.093),2) 'microns '];
        title(worstX)

end
%% plot error histograms to show repeatability 
figure()

subplot(3,1,1)
    histogram(dx_um)
    xlim('auto')
    ylabel('dx bin')
    xlabel('microns')
    title('XYZ error histogram')
            xlim([lowlim uplim])

subplot(3,1,2)
    histogram(dy_um)
    xlim('auto')
    ylabel('dy bin')
    xlabel('microns')
            xlim([lowlim uplim])

subplot(3,1,3)
    histogram(dz_um)
    xlim('auto')
    ylabel('dz bin')
    xlabel('microns')
        xlim([lowlim uplim])

timeToTest = toc
