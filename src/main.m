clear, clc

currentScriptFolder = fileparts(mfilename('fullpath'));
folder = fullfile(currentScriptFolder, '..', 'public', 'dataset', 'Olive_leaf', 'train', 'aculus_olearius');
imageFiles = dir(fullfile(folder, '*.jpg'));
imageFiles = [imageFiles; dir(fullfile(folder, '*.JPG'))];

num_images = length(imageFiles);

outputKMeans = fullfile(folder, 'ROI_KMeans');
outputHist   = fullfile(folder, 'ROI_Histogram');

if ~exist(outputKMeans, 'dir'), mkdir(outputKMeans); end
if ~exist(outputHist, 'dir'), mkdir(outputHist); end

for k = 1:length(imageFiles)
    filename = fullfile(folder, imageFiles(k).name);
    rgbImage = imread(filename);
    
    labImage = rgb2lab(rgbImage);
    L = labImage(:,:,1);
    a = labImage(:,:,2);
    b = labImage(:,:,3);

    %% KMeans
    ab = im2single(cat(3, a, b));
    pixelData = reshape(ab, [], 2);
    nColors = 3;
    [cluster_idx, ~] = kmeans(pixelData, nColors, 'Distance', 'sqEuclidean', 'Replicates', 3);
    pixelLabels = reshape(cluster_idx, size(rgbImage,1), size(rgbImage,2));
    
    % Trova cluster con L* più basso
    meanL = zeros(nColors, 1);
    for i = 1:nColors
        meanL(i) = mean(L(pixelLabels == i), 'all');
    end
    
    [~, diseaseCluster] = min(meanL);
    maskKMeans = pixelLabels == diseaseCluster;

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

    ROI_hist = rgbImage;
    for ch = 1:3
        temp_glcm = ROI_hist(:,:,ch);
        temp_glcm(~maskHist) = 0;
        ROI_hist(:,:,ch) = temp_glcm;
    end

    imwrite(ROI_hist, fullfile(outputHist, ['H_', imageFiles(k).name]));

    %% Matrice GLCM
    I_gray_original = double(rgb2gray(rgbImage)); %trasformazione dell'immagine originale in scala di grigi e conversione in double (per supportare i valori NaN)
    
    I_gray_masked_kmeans = I_gray_original; 
    I_gray_masked_kmeans(~maskKMeans) = NaN; % Imposta a NaN i pixel di sfondo (neri, ma che non fanno parte della ROI)
    
    I_gray_masked_hist = I_gray_original;
    I_gray_masked_hist(~maskHist) = NaN;

    %parametri GLCM
    offsets = [0 1; -1 1; -1 0; -1 -1]; %offset di 1 per tutte e 4 le direzioni
    num_levels = 128;
    is_symmetric = true;

    % Calcolo GLCM
    glcm_kmeans = graycomatrix(I_gray_masked_kmeans, 'Offset', offsets, 'NumLevels', num_levels, 'Symmetric', is_symmetric);
    glcm_hist = graycomatrix(I_gray_masked_hist, 'Offset', offsets, 'NumLevels', num_levels, 'Symmetric', is_symmetric);

    %Calcolo proprietà GLCM
    props_kmeans = graycoprops(glcm_kmeans);
    props_hist = graycoprops(glcm_hist);

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
