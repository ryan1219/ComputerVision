%% Load the input video
vidReader = VideoReader('C:\Users\Administrator\Desktop\ecse415\a5\flow.avi');
%% Create optical flow objects
opticFlowLK = opticalFlowLK('NoiseThreshold',0.009);
opticFlowHS = opticalFlowHS;
%% do for each video frame
while hasFrame(vidReader)
    % read a video frame
    frameRGB = readFrame(vidReader);
    %frameGray = rgb2gray(frameRGB);
    % estimate the LK-based motion field
    flowLK = estimateFlow(opticFlowLK,frameRGB);
    % estimate the HS-based motion field  
    flowHS = estimateFlow(opticFlowHS,frameRGB);
    % display the LK optical flow 
    subplot(1,2,1);
    imshow(frameRGB)
    hold on
    plot(flowLK,'DecimationFactor',[5 5],'ScaleFactor',10)
    hold off
    % display the HS optical flow
    subplot(1,2,2);
    imshow(frameRGB)
    hold on
    plot(flowHS,'DecimationFactor',[5 5],'ScaleFactor',25)
    hold off
    % pause execution (helps in updating the subplots)
    pause(0)
end