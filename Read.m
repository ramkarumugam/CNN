clear all;
clc;

myFolder = 'path\to\dataset';

if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, '*.png');
jpegFiles = dir(filePattern);

%Image size is for the grayscale images mentioned dataset-(height,weight)

imageArraytemp=zeros(40,100,length(jpegFiles));

%do include another dimension after image size, in case if you use RGB images. 

n=[];
for k = 1:length(jpegFiles)
  baseFileName = jpegFiles(k).name;
  n(k)=strncmpi(baseFileName,'p', 1);
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  imtemp=imread(fullFileName);
  imageArraytemp(:,:,k) =im2double(imtemp);
  imshow(imageArraytemp(:,:,k));  % Display image.
  drawnow; % Force display to update immediately.
end
