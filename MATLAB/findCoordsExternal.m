function guess = findCoordsExternal(I,net)
%{    
    As far as I can tell, you need to input an augimds to the "predict()"
    function, so this function will make a temporary mini auimds with
    only one image so we can use the predict function. 
%}
    tic 
    new_dimension = 224; % new dimension of image output (should be a square)
    img_size = [new_dimension new_dimension]; 
    
    % set folder for creating temporary images 
%     temp_folder = 'C:\Users\myip7\Dropbox (GaTech)\Shared folders\Pipette and cell finding\2019-2020 NET\Training and Validation Data\trash\';
    temp_folder = 'C:\Users\mgonzalez91\Dropbox (GaTech)\Research\PBL\Pipette Detection\Pipette and cell finding\2019-2020 NET\Training and Validation Data\trash';
    %% resizing paramters 
    [ysize,xsize] = size(I);
    min_dimension = min([xsize ysize]);
    
    %% Preprocess image using custom function
    pipetteImg = customPreprocess(I,img_size);
    
    %% Save temporary image to temporary folder 
    temp_path = strcat(temp_folder,num2str(cputime),'.png');
    imwrite(pipetteImg, temp_path);
    
    %% Save information to struct for imds
    imgstruct(1).file = string(temp_path);
    imgstruct(1).x = NaN;
    imgstruct(1).y = NaN;
    imgstruct(1).z = NaN;
    imgdata = struct2table(imgstruct);

    %% Create augmented imds for predict function
    auimds = augmentedImageDatastore(img_size,imgdata,...
        'OutputSizeMode','centercrop',...
        'ColorPreprocessing','gray2rgb');
    
    %% Predict x,y,z coords (in pixels) of pipette tip using neural net
    guess = predict(net,auimds);
    
    %% Transform point so that it can be plotted on the resized version
    transformedpoint = guess(1:2);
    transformedpoint(1) = new_dimension*((guess(1)+(xsize/2))-(xsize-min_dimension)/2)/min_dimension; 
    transformedpoint(2) = new_dimension*((guess(2)+(ysize/2))-(ysize-min_dimension)/2)/min_dimension;
    
    guess_um = guess*.1/1.093 % convert to microns

    fullpoint = guess(1:2);
    fullpoint(1) = fullpoint(1)+xsize/2
    fullpoint(2) = fullpoint(2)+ysize/2

    %% Plot
    figure(1)
    tip = insertMarker(I,fullpoint,'x','Color','red','Size',20);
    imshow(tip,'InitialMagnification',200) % zoom so you can see it clearly
%     imagesc(tip)

%     pxbar = 10*1.093/.1 % 10 um scalebar
%     plot([50,50+pxbar],[950,950],'Color','red','LineWidth',2)
%     imwrite(tip,'C:\Users\mgonzalez91\Dropbox (GaTech)\Research\Pipette CNN Paper\ex3_scale.png')
    toc
end