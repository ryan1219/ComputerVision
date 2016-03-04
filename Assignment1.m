function [] = Assignment1(image_name)
%Assignment1 template
%   image_name       Full path/name of the input image (e.g. 'Test Image (1).JPG')
image_name = 'C:\Users\Administrator\Desktop\ecse415\Test images\Test Image (3).png';
%% Load the input RGB image
input_RGB = imread(image_name);
%input_RGB = im2double(input_RGB);
% input_RGB = imnoise(input_RGB,'gaussian',0,0.01);
%% Create a gray-scale duplicate into grayImage variable for processing
grayImage = rgb2gray(input_RGB);
%figure;imshow(grayImage);
%% List all the template files starting with 'Template-' ending with '.png'
% Assuming the images are located in the same directory as this m-file
% Each template file name is accessible by templateFileNames(i).name
templateFileNames = dir('Template-*.png');
%% Get the number of templates (this should return 13)
numTemplates = length(templateFileNames);
%% Set the values of SSD_THRESH and NCC_THRESH
SSD_THRESH = 16000000;
NCC_THRESH = 0.600000;
NSSD_THRESH = 3.0;
%% Initialize two output images to the RGB input image
SSD_OUTPUT = input_RGB;
NCC_OUTPUT = input_RGB;
NSSD_OUTPUT = input_RGB;
sum_ssd = 0;
sum_ncc = 0;
%% For each template, do the following
for i=1:numTemplates
    %% Load the RGB template image, into variable T 
    T = imread(templateFileNames(i).name);
    %figure;imshow(T);
    %% Convert the template to gray-scale   
    T = rgb2gray(T);  
    %figure;imshow(T);   
    %% Extract the card name from its file name (look between '-' and '.' chars)
    % use the cardName variable for generating output images
    cardNameIdx1 = findstr(templateFileNames(i).name,'-') + 1;
    cardNameIdx2 = findstr(templateFileNames(i).name,'.') - 1;
    cardName = templateFileNames(i).name(cardNameIdx1:cardNameIdx2); 
    
    %% Find the best match [row column] using Sum of Square Difference (SSD)
    [SSDrow, SSDcol,SSD_value_for_one_template] = SSD(grayImage, T, SSD_THRESH);
%    sum_ssd = sum_ssd + SSD_value_for_one_template;
    % If the best match exists
    % overlay the card name on the best match location on the SSD output image                      
    % Insert the card name on the output images (use small font size, e.g. 6)
    % set the overlay locations to the best match locations, plus-minus a random integer
    if SSDrow ~= 0 
        SSD_OUTPUT = insertText(SSD_OUTPUT,[SSDcol+randi(20), SSDrow+randi(20)], cardName,'FontSize',15,'BoxColor','yellow');
    end
    % Find the best match [row column] using Normalized Cross Correlation (NCC)
%     [NCCrow, NCCcol, NCC_value_for_one_template] = NCC(grayImage, T, NCC_THRESH);
% %  sum_ncc = sum_ncc + NCC_value_for_one_template;
%     % If the best match exists
%     % overlay the card name on the best match location on the NCC output image                      
%     % Insert the card name on the output images (use small font size, e.g. 6)
%     % set the overlay locations to the best match locations, plus-minus a random integer
%      if NCCrow ~= 0 
%             NSSD_OUTPUT = insertText(NSSD_OUTPUT,[NCCcol+randi(20), NCCrow+randi(20)], cardName,'FontSize',15,'BoxColor','yellow');
%      end
     
    % normalized SSD 
%      [NSSDrow, NSSDcol] = SSDnormed(grayImage, T, NSSD_THRESH);
%       if NSSDrow ~= 0 
%             NSSD_OUTPUT = insertText(NSSD_OUTPUT,[NSSDcol+randi(20), NSSDrow+randi(20)], cardName,'FontSize',15,'BoxColor','yellow');
%       end
   
end

%% Display the output images
figure;
subplot(1,2,1)
imshow(SSD_OUTPUT);
title('SSD');
subplot(1,2,2)
imshow(NSSD_OUTPUT); 
title('normalized SSD');
end

%% Implement the SSD-based template matching here
function [SSDrow, SSDcol, SSD_value_for_one_template] = SSD(grayImage, T, SSD_THRESH)
% inputs
%           grayImag        gray-scale image
%           T               gray-scale template
%           SSD_THRESH      threshold below which a match is accepted
% outputs
%           SSDrow          row of the best match (empty if unavailable)
%           SSDcol          column of the best match (empty if unavailable)
 
  [nrT, ncT] = size(T);
  [nrG, ncG] = size (grayImage);
%  test=double(zeros(size(grayImage)));
  smallest_point = 10000 * SSD_THRESH;
%%
 for i=1:(nrG-nrT)
    for j=1:(ncG-ncT)
        temp = double(grayImage((i+1):(i+nrT),(j+1):(j+ncT)));
        diff=double(double(T)-double(temp));
        sum_at_one_pixel = double(double(sum(double(sum(diff .^2)))));
        %display(sum_at_one_pixel);
        if sum_at_one_pixel < smallest_point
                smallest_point = sum_at_one_pixel;
                SSDrow = i;
                SSDcol = j;
      
        end        
    end        
 end
    SSD_value_for_one_template = smallest_point;
    
    if smallest_point < SSD_THRESH
        display(sum_at_one_pixel);
        return
    else
        SSDrow=0;
        SSDcol=0;
    end
%%
end

%% Implement the NCC-based template matching here
function [NCCrow, NCCcol,NCC_value_for_one_template] = NCC(grayImage, T, NCC_THRESH)
% inputs
%           grayImag        gray-scale image
%           T               gray-scale template
%           NCC_THRESH      threshold above which a match is accepted
% outputs
%           NCCrow          row of the best match (empty if unavailable)
%           NCCcol          column of the best match (empty if unavailable)
[nrT, ncT] = size(T);
[nrG, ncG] = size (grayImage);
biggest_point = -10;
    %// Cast to double for precision
    T = im2double(T);
    grayImage = im2double(grayImage);

    for i=1:(nrG-nrT)
        for j=1:(ncG-ncT)
            grayImagePiece = double(grayImage((i+1):(i+nrT),(j+1):(j+ncT)));
             grayImagePieceMeanSubtract = grayImagePiece - mean2(grayImagePiece);
             templateMeanSubtract = T - mean2(T);
             
             numerator = grayImagePieceMeanSubtract .* templateMeanSubtract;
             sumNumerator = sum(numerator(:));
             denominator1 = sqrt(sum(sum(grayImagePieceMeanSubtract.*grayImagePieceMeanSubtract)));
             denominator2 = sqrt(sum(sum(templateMeanSubtract.*templateMeanSubtract)));
             Denominator = denominator1 .* denominator2;
             sumDenominator = sum(Denominator(:));
             ncc_at_one_pixel = sumNumerator ./ sumDenominator;
             if ncc_at_one_pixel > biggest_point
                biggest_point = ncc_at_one_pixel;
                NCCrow = i;
                NCCcol = j;
                display(ncc_at_one_pixel);
            end        
        end
    end
 NCC_value_for_one_template = biggest_point;   
    if biggest_point > NCC_THRESH
        return      
    else
        NCCrow = 0;
        NCCcol = 0;
    end           
end

function [NSSDrow, NSSDcol] = SSDnormed(grayImage, T, NSSD_THRESH)
    % inputs
%           grayImag        gray-scale image
%           T               gray-scale template
%           SSD_THRESH      threshold below which a match is accepted
% outputs
%           SSDrow          row of the best match (empty if unavailable)
%           SSDcol          column of the best match (empty if unavailable)
 T = im2double(T);
  grayImage = im2double(grayImage);
  [nrT, ncT] = size(T);
  [nrG, ncG] = size (grayImage);
%  test=double(zeros(size(grayImage)));
  smallest_point = 10000 * NSSD_THRESH;
%%
 for i=1:(nrG-nrT)
    for j=1:(ncG-ncT)
        grayImagePiece = double(grayImage((i+1):(i+nrT),(j+1):(j+ncT)));
        diff=double(double(T)-double(grayImagePiece));
        nominator = double(double(sum(double(sum(diff .^2)))));
           
        denominator1 = T-grayImagePiece;       
        Denominator = sqrt(sum(denominator1 .* denominator1));
        sumDenominator = sum(Denominator(:));
        sum_at_one_pixel = nominator ./ sumDenominator;
        if sum_at_one_pixel < smallest_point
                smallest_point = sum_at_one_pixel;
                NSSDrow = i;
                NSSDcol = j;
                
        end        
    end        
 end
    if smallest_point < NSSD_THRESH
        display(sum_at_one_pixel);
        return
    else
        NSSDrow=0;
        NSSDcol=0;
    end
%%
end