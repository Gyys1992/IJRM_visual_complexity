function num_objects = object(img,detector)
images = dir('P:\Champion\Champion of Images\Data2\Images\*.jpg');
img = imread(append(images(1).folder,'\',images(4).name));
img = preprocess(detector,img);
img = im2single(img);
[bboxes,scores,labels] = detect(detector,img,'DetectionPreprocessing','none');
num_objects = sum(scores>0.5);