function dataOut = customPreprocess(dataIn,image_size)
    I = dataIn;
    [ysize,xsize,zsize] = size(I);
    minDimension = min([xsize ysize]);

    xmin = floor(xsize/2-minDimension/2);
    ymin = floor(ysize/2-minDimension/2);
    width = minDimension;
    height = minDimension;
    
    n=2; % 2 std above or below mean are max and min
    Isingle = im2single(I);
    avg = mean2(Isingle);
    sigma = std2(Isingle);
    min_val = avg-n*sigma; if min_val < 0; min_val = 0; end
    max_val = avg+n*sigma; if max_val > 1; max_val = 1; end
    Iadjusted = imadjust(Isingle,[min_val max_val],[]);
%     imcrop(Iadjusted,[xmin ymin width height])
    dataOut = imresize(imcrop(Iadjusted,[xmin ymin width height]),image_size,'bilinear');
end
