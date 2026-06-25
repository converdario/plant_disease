clear, clc
warning('off', 'images:graycomatrix:scaledImageContainsNan');
warning('off', 'stats:kmeans:FailedToConvergeRep');
%tic

currentScriptFolder = fileparts(mfilename('fullpath'));
folder = fullfile(currentScriptFolder, '..', 'public', 'dataset', 'Olive_leaf', 'train', 'olive_peacock_spot');
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

for k = 1:100
%for k = 1:length(imageFiles)
    filename = fullfile(folder, imageFiles(k).name);
    rgbImage = imread(filename);
    
    labImage = rgb2lab(rgbImage);
    L = labImage(:,:,1);
    a = labImage(:,:,2);
    b = labImage(:,:,3);

   
    lab_single = im2single(labImage);
    pixelData = reshape(lab_single, [], 3); % Usiamo tutti i canali L, a, b
    
    nColors = 3; 
    
    % Applichiamo K-Means direttamente a tutta l'immagine
    [cluster_idx, ~] = kmeans(pixelData, nColors, 'Distance', 'sqEuclidean', 'Replicates', 3);
    pixelLabels = reshape(cluster_idx, size(rgbImage,1), size(rgbImage,2));
    
    % --- Identificazione Automatica dei Cluster ---
    % Logica deduttiva basata sui colori per assegnare i 3 cluster:
    % 1. Sfondo: Ipotizziamo sia il cluster che tocca maggiormente i bordi
    % 2. Foglia Sana: Valore a* più basso (più verde)
    % 3. Malattia: Valore a* più alto (più marrone/giallo/necrotico)
    
    borderMask = true(size(pixelLabels));
    borderMask(2:end-1, 2:end-1) = false;
    
    count1 = sum(pixelLabels(borderMask) == 1);
    count2 = sum(pixelLabels(borderMask) == 2);
    count3 = sum(pixelLabels(borderMask) == 3);
    
    % Trova il cluster che domina i bordi e scartalo (Sfondo)
    [~, bgCluster] = max([count1, count2, count3]);
    
    % Calcola la media del canale a* per i due cluster rimanenti
    meanA = zeros(3, 1);
    for i = 1:3
        if i ~= bgCluster
            meanA(i) = mean(a(pixelLabels == i)); 
        else
            meanA(i) = -Inf; % Ignoriamo lo sfondo
        end
    end
    
    % La malattia è il cluster con il valore a* più alto tra quelli rimasti
    [~, diseaseCluster] = max(meanA);
    
    maskKMeans = (pixelLabels == diseaseCluster);

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
    I_gray_original = double(rgb2gray(rgbImage)) / 255; %trasformazione dell'immagine originale in scala di grigi e conversione in double (per supportare i valori NaN)
    
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

    % Calcolo entropia (non inclusa in graycoprops)
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

save('all_extracted_texture_features.mat', 'allExtractedFeatures');
fprintf('\nTutte le feature sono state estratte e salvate in all_extracted_texture_features.mat\n');

%tempo_trascorso = toc;
%tempo_trascorso = datestr(tempo_trascorso/(24*3600), 'HH:MM:SS');
%disp(['Il tempo di elaborazione è stato: ', tempo_trascorso]) ;

