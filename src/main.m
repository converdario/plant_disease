clear, clc
warning('off', 'images:graycomatrix:scaledImageContainsNan');
warning('off', 'stats:kmeans:FailedToConvergeRep');
%tic

currentScriptFolder = fileparts(mfilename('fullpath'));
folder = fullfile(currentScriptFolder, '..', 'public', 'dataset', 'healthy');
imageFiles = dir(fullfile(folder, '*.jpg'));
num_images = length(imageFiles);
allExtractedFeatures = table(...
    'Size', [0, 11], ... % 0 righe inizialmente, 13 colonne (3 info + 5 KMeans + 5 Hist)
    'VariableTypes', {'string', ... % ImageName
                      'double', 'double', 'double', 'double', 'double', ... % KMeans_Energy, ..., Entropy
                      'double', 'double', 'double', 'double', 'double'}, ... % Hist_Energy, ..., Entropy
    'VariableNames', {'ImageName', ...
                     'KMeans_Energy', 'KMeans_Contrast', 'KMeans_Correlation', 'KMeans_Homogeneity', 'KMeans_Entropy', ...
                     'Hist_Energy', 'Hist_Contrast', 'Hist_Correlation', 'Hist_Homogeneity', 'Hist_Entropy'}); %tabella che conterrà i risultati in output 

outputKMeans = fullfile(folder, 'ROI_KMeans');
outputHist   = fullfile(folder, 'ROI_Histogram');

if ~exist(outputKMeans, 'dir'), mkdir(outputKMeans); end
if ~exist(outputHist, 'dir'), mkdir(outputHist); end

for k = 1:length(imageFiles)
%for k = 1:5
    filename = fullfile(folder, imageFiles(k).name);
    rgbImage = imread(filename);

    labImage = rgb2lab(rgbImage);
    L = labImage(:,:,1);
    a = labImage(:,:,2);
    b = labImage(:,:,3);
    %% Rimozione sfondo
    %maskLeaf = removeBackground(rgbImage);
    %maskLeaf = imbinarize(rgb2gray(rgbImage), graythresh(rgb2gray(rgbImage)));
    maskLeaf = removeBackgroundSuperpixel(rgbImage);

    %% KMeans sulla sola foglia

    ab = im2single(cat(3,a,b));

    pixelDataAll = reshape(ab,[],2);

    leafPixels = pixelDataAll(maskLeaf(:),:);
    nColors = 4;

if size(leafPixels,1) < nColors
    fprintf("Immagine %s: foglia non trovata, salto.\n", imageFiles(k).name);
    continue;
end

    opts = statset('MaxIter',500);

    [cluster_idx, ~] = kmeans( ...
        leafPixels, ...
        nColors, ...
        'Distance','sqEuclidean', ...
        'Replicates',5, ...
        'Start','plus', ...
        'Options',opts);

    pixelLabels = zeros(size(maskLeaf));

    pixelLabels(maskLeaf) = cluster_idx;

    meanL = zeros(nColors,1);
    clusterArea = zeros(nColors,1);

    for i = 1:nColors

        currentMask = (pixelLabels == i);

        clusterArea(i) = sum(currentMask(:));

        if clusterArea(i) == 0
            meanL(i) = Inf;
        else
            meanL(i) = mean(L(currentMask));
        end

        fprintf('Cluster %d -> Area=%d pixels, L*=%.2f\n', ...
            i, clusterArea(i), meanL(i));
    end

    minArea = 0.01 * sum(maskLeaf(:));

    validClusters = clusterArea > minArea;

    meanL(~validClusters) = Inf;

    [~, diseaseCluster] = min(meanL);

    maskKMeans = (pixelLabels == diseaseCluster);
    fprintf('Leaf area      = %d\n', sum(maskLeaf(:)));
    fprintf('Disease area   = %d\n', sum(maskKMeans(:)));
    fprintf('Disease ratio  = %.2f %%\n', ...
        100*sum(maskKMeans(:))/sum(maskLeaf(:)));

    ROI_kmeans = rgbImage;
    for ch = 1:3
        temp_glcm = ROI_kmeans(:,:,ch);
        temp_glcm(~maskKMeans) = 0;
        ROI_kmeans(:,:,ch) = temp_glcm;
    end

    imwrite(ROI_kmeans, fullfile(outputKMeans, ['K_', imageFiles(k).name]));

    %% Thresholding su canale a*
    aChannel = a;
    aChannel = mat2gray(aChannel);
    level = graythresh(aChannel);
    maskHist = imbinarize(aChannel, level);
    maskHist = maskHist & maskLeaf;
    fprintf('Histogram area = %d\n', sum(maskHist(:)));

    ROI_hist = rgbImage;
    for ch = 1:3
        temp_glcm = ROI_hist(:,:,ch);
        temp_glcm(~maskHist) = 0;
        ROI_hist(:,:,ch) = temp_glcm;
    end

    imwrite(ROI_hist, fullfile(outputHist, ['H_', imageFiles(k).name]));

    %% Matrice GLCM

    I_gray_original = uint8(rgb2gray(rgbImage));

    offsets = [0 1; -1 1; -1 0; -1 -1];
    num_levels = 128;
    is_symmetric = true;

    %% ROI KMEANS

    statsK = regionprops(maskKMeans,'Area','BoundingBox');

    if isempty(statsK)
        fprintf('ROI KMeans vuota\n');
        continue;
    end

    [~,idxK] = max([statsK.Area]);

    bboxK = round(statsK(idxK).BoundingBox);

    roiGrayK = imcrop(I_gray_original,bboxK);
    roiMaskK = imcrop(maskKMeans,bboxK);

    roiGrayK(~roiMaskK) = 0;

    glcm_kmeans = graycomatrix( ...
        roiGrayK,...
        'Offset',offsets,...
        'NumLevels',num_levels,...
        'Symmetric',is_symmetric);

    %% ROI HISTOGRAM

    statsH = regionprops(maskHist,'Area','BoundingBox');

    if isempty(statsH)
        fprintf('ROI Histogram vuota\n');
        continue;
    end

    [~,idxH] = max([statsH.Area]);

    bboxH = round(statsH(idxH).BoundingBox);

    roiGrayH = imcrop(I_gray_original,bboxH);
    roiMaskH = imcrop(maskHist,bboxH);

    roiGrayH(~roiMaskH) = 0;

    glcm_hist = graycomatrix( ...
        roiGrayH,...
        'Offset',offsets,...
        'NumLevels',num_levels,...
        'Symmetric',is_symmetric);

    %% Proprietà GLCM

    props_kmeans = graycoprops(glcm_kmeans);
    props_hist   = graycoprops(glcm_hist);
    % Calculo entropia (non inclusa in graycoprops)
    entropy_kmeans = zeros(1, size(glcm_kmeans, 3));
    for i = 1:size(glcm_kmeans, 3)
        temp_glcm = glcm_kmeans(:,:,i); % Estrazione matrice 2D i-esima
        temp_glcm_norm = temp_glcm / sum(temp_glcm(:)); % Normalizzazione matrice
        non_zero_elements = temp_glcm_norm(temp_glcm_norm>0);
        entropy_kmeans(i) = -sum(non_zero_elements .* log(non_zero_elements));
    end

    entropy_hist = zeros(1, size(glcm_hist, 3));
    for i = 1:size(glcm_hist, 3)
        temp_glcm = glcm_hist(:,:,i);
        temp_glcm_norm = temp_glcm / sum(temp_glcm(:));
        non_zero_elements = temp_glcm_norm(temp_glcm_norm>0);
        entropy_hist(i) = -sum(non_zero_elements .* log(non_zero_elements));
    end

    % Calcolo delle medie delle proprietà per ottenere un unico valore dai
    % 4 delle 4 direzioni
    mean_energy_kmeans = mean([props_kmeans.Energy]);
    mean_contrast_kmeans = mean([props_kmeans.Contrast]);
    mean_correlation_kmeans = mean([props_kmeans.Correlation]);
    mean_homogeneity_kmeans = mean([props_kmeans.Homogeneity]);
    mean_entropy_kmeans = mean(entropy_kmeans);

    mean_energy_hist = mean([props_hist.Energy]);
    mean_contrast_hist = mean([props_hist.Contrast]);
    mean_correlation_hist = mean([props_hist.Correlation]);
    mean_homogeneity_hist = mean([props_hist.Homogeneity]);
    mean_entropy_hist = mean(entropy_hist);

    newRow = {imageFiles(k).name, ... 
              mean_energy_kmeans, ...
              mean_contrast_kmeans, ...
              mean_correlation_kmeans, ...
              mean_homogeneity_kmeans, ...
              mean_entropy_kmeans, ...
              mean_energy_hist, ...
              mean_contrast_hist, ...
              mean_correlation_hist, ...
              mean_homogeneity_hist, ...
              mean_entropy_hist};
            
    allExtractedFeatures = [allExtractedFeatures; newRow]; % aggiunta alla tabella della riga con le feature calcolate
    

    fprintf('Elaborata immagine %d/%d: %s\n', k, num_images, imageFiles(k).name);
    fprintf('  K-Means ROI: Energy=%.4f, Contrast=%.4f, Correlation=%.4f, Homogeneity=%.4f, Entropy=%.4f\n', ...
        mean_energy_kmeans, mean_contrast_kmeans, mean_correlation_kmeans, mean_homogeneity_kmeans, mean_entropy_kmeans);
    fprintf('  Histogram ROI: Energy=%.4f, Contrast=%.4f, Correlation=%.4f, Homogeneity=%.4f, Entropy=%.4f\n', ...
        mean_energy_hist, mean_contrast_hist, mean_correlation_hist, mean_homogeneity_hist, mean_entropy_hist);



    %     figure;
    %     subplot(1,3,1), imshow(rgbImage), title('Originale');
    %     subplot(1,3,2), imshow(ROI_kmeans), title('K-means ROI');
    %     subplot(1,3,3), imshow(ROI_hist), title('Histogram ROI');
    %     pause(0.5);
    %     close all;
end

outputFilePath = fullfile(folder, 'all_extracted_texture_features.mat');
save(outputFilePath, 'allExtractedFeatures');
fprintf('\nTutte le feature sono state estratte e salvate in:\n%s\n', outputFilePath);

%tempo_trascorso = toc;
%tempo_trascorso = datestr(tempo_trascorso/(24*3600), 'HH:MM:SS');
%disp(['Il tempo di elaborazione è stato: ', tempo_trascorso]) ;

function maskLeaf = removeBackground(rgbImage)

hsvImage = rgb2hsv(rgbImage);

H = hsvImage(:,:,1);
S = hsvImage(:,:,2);
V = hsvImage(:,:,3);

% Maschera iniziale del verde
maskLeaf = S > graythresh(S);

maskLeaf = imfill(maskLeaf,'holes');

maskLeaf = bwareaopen(maskLeaf,1000);

maskLeaf = imclose(maskLeaf,strel('disk',15));

% Riempimento buchi
maskLeaf = imfill(maskLeaf,'holes');

% Rimozione piccoli oggetti
maskLeaf = bwareaopen(maskLeaf,500);

% Mantieni la componente con area maggiore
CC = bwconncomp(maskLeaf);

if CC.NumObjects > 0
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [~, idx] = max(numPixels);

    maskLeaf = false(size(maskLeaf));
    maskLeaf(CC.PixelIdxList{idx}) = true;
end

% Chiusura morfologica
se = strel('disk',10);
maskLeaf = imclose(maskLeaf,se);

end

function maskLeaf = removeBackgroundSuperpixel(rgbImage)

% Numero di superpixel (più alto = più preciso ma più lento)
numSuperpixels = 300;

% Segmentazione superpixel
[L,N] = superpixels(rgbImage, numSuperpixels);

% Feature per ogni superpixel
lab = rgb2lab(rgbImage);
Lch = lab(:,:,1);
Ach = lab(:,:,2);
Bch = lab(:,:,3);

meanL = zeros(N,1);
meanS = zeros(N,1);

hsv = rgb2hsv(rgbImage);
S = hsv(:,:,2);

% Calcolo feature per ogni superpixel
for i = 1:N
    mask = (L == i);

    meanL(i) = mean(Lch(mask));
    meanS(i) = mean(S(mask));
end

% 🔍 Regola di selezione:
% foglia = più verde + più texture (semplice euristica)
score = meanS .* (1 - mat2gray(meanL));

% soglia automatica
t = graythresh(score);

selectedLabels = find(score > t);

% costruzione maschera finale
maskLeaf = ismember(L, selectedLabels);

% pulizia morfologica
maskLeaf = imfill(maskLeaf,'holes');
maskLeaf = bwareaopen(maskLeaf, 500);
maskLeaf = imclose(maskLeaf, strel('disk',10));

end
