cd /Users/gijso/Downloads/
addpath /Users/gijso/Downloads/VCMatlab/
addpath /Users/gijso/Downloads/VCMatlab/Photography/
addpath /Users/gijso/Downloads/VCMatlab/Complexity
addpath /Users/gijso/Downloads/VCMatlab/SFFCMCode/SFFCMCode/
addpath /Users/gijso/Downloads/VCMatlab/Alternative_Complexity/
addpath /Users/gijso/Downloads/VCMatlab/CESC

images = dir('Instagram/NewData/*.jpg');



% neural net to extract objects 
net = vgg16('Weights','imagenet');
layer = 'prob';
imds = imageDatastore('/Users/gijso/Downloads/Instagram/NewData/');
inputSize = net.Layers(1).InputSize;
augimds = augmentedImageDatastore(inputSize(1:2),imds);
objects = activations(net,augimds,layer,'OutputAs','rows');



N = size(images,1);

diag_domv=zeros(N,1);
rotv=zeros(N,1);
vphdv=zeros(N,1);
hphdv=zeros(N,1);
horizontalv=zeros(N,1);
verticalv=zeros(N,1);
size_difv=zeros(N,1);
col_difv=zeros(N,1);
text_difv=zeros(N,1);
brightnessv=zeros(N,1);
saturationv=zeros(N,1);
contrastv=zeros(N,1);
clarityv=zeros(N,1);
warmthv=zeros(N,1);
fcv=zeros(N,1);
sev=zeros(N,1);
lcv=zeros(N,1);
ccv=zeros(N,1);
edv=zeros(N,1);
%ocv=zeros(N,1);

m1v = zeros(1,N);
m2v = zeros(1,N);
m3v = zeros(1,N);
m4v = zeros(1,N);
m5v = zeros(1,N);
m8v = zeros(1,N);
m9v = zeros(1,N);
m10v = zeros(1,N);
m11v = zeros(1,N);

sdv = zeros(N,1);

for k = 1:N
    
    im = strcat('Instagram/NewData/',images(k).name);
    img = imread(im);

%% composition
% segmentation

sdv(k) = std(double(img(:)));


if sdv(k)<10
   continue 
end


%[regs, ~, ~] = segmentation(img,10); % input RGB image



regs = superpixels(img,10);

%visual saliency calculation

[vis_sal] = vsalience(img); %input same RGB image


% determine salient region
saliency_region = zeros(max(max(regs)),1);
for i=1:max(max(regs))
    saliency_region(i) =mean(mean(vis_sal(regs==i)));
end

a=regionprops(regs,'Centroid'); %find centers for each region
centroids = [a.Centroid];
xCentroids = centroids(1:2:end);
yCentroids = centroids(2:2:end);

m = find(saliency_region==max(saliency_region));
center_salient = [xCentroids(m) yCentroids(m)];



%diagonal dominance 

diag_dom = diagonaldominance(img, center_salient);

%rule of thirds
rot = ruleofthirds(img, center_salient);

% physical balance
[vphd, hphd] = physicalbalance(img, xCentroids,yCentroids,saliency_region);

% color balance
[horizontal, vertical] = colorbalance(img);


%% figure-ground

A = figureground(img);

% size-difference
size_dif = sizedifference(A);
% color-difference
col_dif = colordifference(img,A);

% texture difference
text_dif = texturedifference(img,A);

%depth of field 
%hsv = colorspace('HSV<-RGB',img);

%wavelet = dbaux(hsv(:,:,1));
%% color 

[brightness, saturation, contrast, clarity, warmth] = color_photography(img);

%% Visual Complexity
option1 = imread('Option 1.png');
option2 = imread('Option 2.png');
option3 = imread('Option 3.png');
option4 = imread('Option 4.png');
img = option4;



[fc, se] = clutter(img); % clutter, feature congestion and subband entropy

lc = luminance_complexity(img); % luminance complexity
cc = color_complexity(img); % color complexity
ed = edge_density(img); % edge density 



oc = objectcomplexity(objects(k,:));

%% Alternative Complexity 
% Corchs et al. complexity measures, m6 is equal to edge density, m7 is
% jpeg file size
[m1, m2, m3, m4] = complexity1_4(img);
m5 = freqfactor(img);
m9 = colorfulness(img);
m10 = numofcolors(img);
[m11, m8]=color_harmony(img,10);


diag_domv(k)=diag_dom;
rotv(k)=rot;
vphdv(k)=vphd;
hphdv(k)=hphd;
horizontalv(k)=horizontal;
verticalv(k)=vertical;
size_difv(k)=size_dif;
col_difv(k)=col_dif;
text_difv(k)=text_dif;
brightnessv(k)=brightness;
saturationv(k)=saturation;
contrastv(k)=contrast;
clarityv(k)=clarity;
warmthv(k)=warmth;
fcv(k)=fc;
sev(k)=se;
lcv(k)=lc;
ccv(k)=cc;
edv(k)=ed;
ocv(k)=oc;
    



ahv =zeros(N,1);
avv = zeros(N,1);
irv = zeros(N,1);

for k = 1:N % to continue later
    im = strcat('Instagram/NewData/',images(k).name);
    img = imread(im); 
    if numel(size(img))<3
        continue
    end
    
    if lcv2(k)==0 | sdv(k)<10
        continue 
    end
    [ahv(k) avv(k) irv(k)] =arrangement(img);
    if mod(k,1000)==0
        fprintf('%d\n',k)
    end
end
m8= m8v;
m11 = m11v

all_image_features = table(diag_domv,rotv,vphdv,hphdv,horizontalv,verticalv,size_difv,col_difv,text_difv,brightnessv,saturationv,contrastv,clarityv,warmthv,lcv,lcv2,ccv,edv,ocv,ahv,avv,irv,m1,m2,m3,m4,m5,m8,m9,m10,m11,CEv',SCv',indicator);
image_names = struct2table(images);
image_features = [all_image_features image_names.name];

writetable(image_features,'image_features.xlsx')



