clear, clc
warning('off', 'images:graycomatrix:scaledImageContainsNan');
warning('off', 'stats:kmeans:FailedToConvergeRep');

%% =========================================================================
%% CONFIGURAZIONE PARAMETRI 
%% =========================================================================

% --- Parametri Segmentazione K-Means ---
NUM_CLUSTERS = 3; 

% --- Parametri Estrazione Feature (GLCM) ---
GLCM_OFFSETS = [0 1; -1 1; -1 0; -1 -1]; 
GLCM_SYMMETRIC = true;
GLCM_NUM_LEVELS = 128; 

%% =========================================================================
%% INIZIALIZZAZIONE
%% =========================================================================

currentScriptFolder = fileparts(mfilename('fullpath'));
folder = fullfile(currentScriptFolder, '..', 'public', 'dataset', 'aculus_olearius'); % Directory dove prelevare le immagini

imageFiles = dir(fullfile(folder, '*.jpg'));
if isempty(imageFiles)
    imageFiles = dir(fullfile(folder, '*.JPG'));
end
num_images = length(imageFiles);

allExtractedFeatures = table(...
    'Size', [0, 12], ...
    'VariableTypes', {'string', 'double', ...
                      'double', 'double', 'double', 'double', 'double', ...
                      'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'ImageName', 'Infection_Percentage', ...
                     'KMeans_Energy', 'KMeans_Contrast', 'KMeans_Correlation', 'KMeans_Homogeneity', 'KMeans_Entropy', ...
                     'Hist_Energy', 'Hist_Contrast', 'Hist_Correlation', 'Hist_Homogeneity', 'Hist_Entropy'});

outputKMeans = fullfile(folder, 'ROI_KMeans');
outputHist   = fullfile(folder, 'ROI_Histogram');
outputNoBackground = fullfile(folder, 'NoBackground');

if ~exist(outputNoBackground, 'dir')
    mkdir(outputNoBackground);
end

if ~exist(outputKMeans, 'dir'), mkdir(outputKMeans); end
if ~exist(outputHist, 'dir'), mkdir(outputHist); end

%% =========================================================================
%% CICLO PRINCIPALE ELABORAZIONE IMMAGINI
%% =========================================================================

for k = 1:10
    filename = fullfile(folder, imageFiles(k).name);
    rgbImage = imread(filename);

    labImage = rgb2lab(rgbImage);
    L = labImage(:,:,1);
    a = labImage(:,:,2);
    b = labImage(:,:,3);

    %% --- RIMOZIONE SFONDO ---
    maskLeaf = removeBackgroundSuperpixel(rgbImage);

    %% --- SALVATAGGIO IMMAGINE SENZA SFONDO ---
    rgbNoBackground = rgbImage;
    
    for ch = 1:3
        temp = rgbNoBackground(:,:,ch);
        temp(~maskLeaf) = 0;          % imposta lo sfondo a nero
        rgbNoBackground(:,:,ch) = temp;
    end
    
    imwrite(rgbNoBackground, ...
        fullfile(outputNoBackground, ['NB_', imageFiles(k).name]));

    %% --- SEGMENTAZIONE K-MEANS SULLA FOGLIA (Ottimizzata) ---
    ab = im2single(cat(3, a, b));
    
    % Applichiamo la maschera per azzerare lo sfondo sui canali a* e b*
    ab(:, :, 1) = ab(:, :, 1) .* maskLeaf;
    ab(:, :, 2) = ab(:, :, 2) .* maskLeaf;
    
    % Segmentazione K-Means diretta sull'immagine
    % 'centers' è una matrice 3x2 contenente i baricentri [a*, b*] dei cluster
    [pixelLabels, centers] = imsegkmeans(ab, NUM_CLUSTERS);
    
    % 1. Individuiamo quale dei 3 cluster è stato assegnato allo sfondo
    % (È il valore più frequente al di fuori della maschera della foglia)
    bgCluster = mode(pixelLabels(~maskLeaf));
    
    % 2. Escludiamo lo sfondo dalla ricerca impostando il suo centro a -Inf
    centers(bgCluster, 1) = -Inf; 
    
    % 3. La malattia è il cluster con il baricentro a* (prima colonna) più alto
    [~, diseaseCluster] = max(centers(:, 1));
    
    % Generazione maschera finale
    maskKMeans = (pixelLabels == diseaseCluster) & maskLeaf;

    % Calcolo della percentuale di infezione
    leaf_pixels = sum(maskLeaf(:));
    disease_pixels = sum(maskKMeans(:));
    
    if leaf_pixels > 0
        infection_percentage = (disease_pixels / leaf_pixels) * 100;
    else
        infection_percentage = 0;
    end

    % Salvataggio ROI K-Means
    ROI_kmeans = rgbImage;
    for ch = 1:3
        temp_img = ROI_kmeans(:,:,ch);
        temp_img(~maskKMeans) = 0;
        ROI_kmeans(:,:,ch) = temp_img;
    end
    imwrite(ROI_kmeans, fullfile(outputKMeans, ['K_', imageFiles(k).name]));

    %% --- THRESHOLDING SU CANALE a* (Metodo di confronto) ---
    aChannel = mat2gray(a);
    level = graythresh(aChannel);
    maskHist = imbinarize(aChannel, level);
    maskHist = maskHist & maskLeaf; 
    
    ROI_hist = rgbImage;
    for ch = 1:3
        temp_img = ROI_hist(:,:,ch);
        temp_img(~maskHist) = 0;
        ROI_hist(:,:,ch) = temp_img;
    end
    imwrite(ROI_hist, fullfile(outputHist, ['H_', imageFiles(k).name]));

    %% --- MATRICE GLCM ED ESTRAZIONE FEATURE ---
    I_gray_original = uint8(rgb2gray(rgbImage));

    % === ROI KMEANS ===
    statsK = regionprops(maskKMeans,'Area','BoundingBox');
    if isempty(statsK)
        mean_energy_kmeans = NaN; mean_contrast_kmeans = NaN;
        mean_correlation_kmeans = NaN; mean_homogeneity_kmeans = NaN; mean_entropy_kmeans = NaN;
    else
        [~,idxK] = max([statsK.Area]);
        bboxK = round(statsK(idxK).BoundingBox);
        roiGrayK = imcrop(I_gray_original,bboxK);
        roiMaskK = imcrop(maskKMeans,bboxK);
        roiGrayK(~roiMaskK) = 0;

        glcm_kmeans = graycomatrix(roiGrayK, 'Offset', GLCM_OFFSETS, 'NumLevels', GLCM_NUM_LEVELS, 'Symmetric', GLCM_SYMMETRIC);
        
        props_kmeans = graycoprops(glcm_kmeans);
        entropy_kmeans = zeros(1, size(glcm_kmeans, 3));
        for i = 1:size(glcm_kmeans, 3)
            temp_glcm = glcm_kmeans(:,:,i);
            temp_glcm_norm = temp_glcm / sum(temp_glcm(:));
            non_zero_elements = temp_glcm_norm(temp_glcm_norm>0);
            entropy_kmeans(i) = -sum(non_zero_elements .* log(non_zero_elements));
        end
        
        mean_energy_kmeans = mean([props_kmeans.Energy]);
        mean_contrast_kmeans = mean([props_kmeans.Contrast]);
        mean_correlation_kmeans = mean([props_kmeans.Correlation]);
        mean_homogeneity_kmeans = mean([props_kmeans.Homogeneity]);
        mean_entropy_kmeans = mean(entropy_kmeans);
    end

    % === ROI HISTOGRAM ===
    statsH = regionprops(maskHist,'Area','BoundingBox');
    if isempty(statsH)
        mean_energy_hist = NaN; mean_contrast_hist = NaN;
        mean_correlation_hist = NaN; mean_homogeneity_hist = NaN; mean_entropy_hist = NaN;
    else
        [~,idxH] = max([statsH.Area]);
        bboxH = round(statsH(idxH).BoundingBox);
        roiGrayH = imcrop(I_gray_original,bboxH);
        roiMaskH = imcrop(maskHist,bboxH);
        roiGrayH(~roiMaskH) = 0;

        glcm_hist = graycomatrix(roiGrayH, 'Offset', GLCM_OFFSETS, 'NumLevels', GLCM_NUM_LEVELS, 'Symmetric', GLCM_SYMMETRIC);
        
        props_hist = graycoprops(glcm_hist);
        entropy_hist = zeros(1, size(glcm_hist, 3));
        for i = 1:size(glcm_hist, 3)
            temp_glcm = glcm_hist(:,:,i);
            temp_glcm_norm = temp_glcm / sum(temp_glcm(:));
            non_zero_elements = temp_glcm_norm(temp_glcm_norm>0);
            entropy_hist(i) = -sum(non_zero_elements .* log(non_zero_elements));
        end

        mean_energy_hist = mean([props_hist.Energy]);
        mean_contrast_hist = mean([props_hist.Contrast]);
        mean_correlation_hist = mean([props_hist.Correlation]);
        mean_homogeneity_hist = mean([props_hist.Homogeneity]);
        mean_entropy_hist = mean(entropy_hist);
    end

    % Salvataggio riga corrente
    newRow = {imageFiles(k).name, ... 
              infection_percentage, ...
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
            
    allExtractedFeatures = [allExtractedFeatures; newRow];
    
    fprintf('Elaborata immagine %d/%d: %s (Area infetta: %.2f%%)\n', k, num_images, imageFiles(k).name, infection_percentage);
end

%% =========================================================================
%% SALVATAGGIO FINALE E CALCOLO CORRELAZIONE
%% =========================================================================

outputFilePath = fullfile(folder, 'all_extracted_texture_features.mat');
save(outputFilePath, 'allExtractedFeatures');
fprintf('\nTutte le feature sono state estratte e salvate in:\n%s\n', outputFilePath);

% Rimuove le righe con NaN per garantire che corrcoef funzioni correttamente
validRows = ~isnan(allExtractedFeatures.KMeans_Energy) & ~isnan(allExtractedFeatures.Infection_Percentage);
validFeatures = allExtractedFeatures(validRows, :);

if height(validFeatures) > 1
    inf_perc = validFeatures.Infection_Percentage;
    energy_k = validFeatures.KMeans_Energy;
    entropy_k = validFeatures.KMeans_Entropy;

    R_energy = corrcoef(energy_k, inf_perc);
    r_energy = R_energy(1,2);

    R_entropy = corrcoef(entropy_k, inf_perc);
    r_entropy = R_entropy(1,2);

    fprintf('\n==================================================\n');
    fprintf('RISULTATI DI CORRELAZIONE\n');
    fprintf('==================================================\n');
    fprintf('Energia  : r = %7.4f\n', r_energy);
    fprintf('Entropia : r = %7.4f\n', r_entropy);
    fprintf('==================================================\n');
else
    disp('Non ci sono abbastanza immagini valide per calcolare la correlazione.');
end

%% =========================================================================
%% FUNZIONI AUSILIARIE (Rimozione Sfondo)
%% =========================================================================

function maskLeaf = removeBackgroundSuperpixel(rgbImage)

%% =====================================================
% 1) Segmentazione iniziale HSV
% ======================================================

hsvImage = rgb2hsv(rgbImage);

S = hsvImage(:,:,2);
V = hsvImage(:,:,3);


% Maschera basata sulla saturazione
thresholdS = graythresh(S);

maskLeaf = S > thresholdS;


% Elimina zone quasi bianche (spesso riflessi o sfondo chiaro)
maskLeaf = maskLeaf & (V < 0.97);



%% =====================================================
% 2) Pulizia iniziale
% ======================================================

maskLeaf = imfill(maskLeaf,'holes');

maskLeaf = bwareaopen(maskLeaf,1000);

maskLeaf = imclose(maskLeaf,strel('disk',10));

maskLeaf = imfill(maskLeaf,'holes');



%% =====================================================
% 3) Mantiene solo la foglia principale
% ======================================================

CC = bwconncomp(maskLeaf);


if CC.NumObjects > 0

    areas = cellfun(@numel,CC.PixelIdxList);

    [~,idx] = max(areas);

    tempMask = false(size(maskLeaf));

    tempMask(CC.PixelIdxList{idx}) = true;

    maskLeaf = tempMask;

end



%% =====================================================
% 4) Raffinamento colore LAB
% ======================================================

labImage = rgb2lab(rgbImage);


L = labImage(:,:,1);
A = labImage(:,:,2);
B = labImage(:,:,3);



% Colore medio della foglia individuata

leafPixels = find(maskLeaf);


meanLeafColor = [
    mean(L(leafPixels))
    mean(A(leafPixels))
    mean(B(leafPixels))
    ];



% distanza LAB di ogni pixel dal colore della foglia

colorDistance = sqrt( ...
    (L-meanLeafColor(1)).^2 + ...
    (A-meanLeafColor(2)).^2 + ...
    (B-meanLeafColor(3)).^2 );



% soglia adattiva:
% prende il 95% dei pixel della foglia trovata

leafDistance = colorDistance(maskLeaf);

thresholdLAB = prctile(leafDistance,85);



% mantiene solo pixel compatibili con la foglia

refinedMask = colorDistance < thresholdLAB;



%% =====================================================
% 5) Vincolo spaziale
%    evita che ritorni lo sfondo lontano
% ======================================================

% permette una piccola espansione del bordo

allowedRegion = imdilate(maskLeaf,strel('disk',15));


refinedMask = refinedMask & allowedRegion;



%% =====================================================
% 6) Pulizia finale
% ======================================================

maskLeaf = refinedMask;


maskLeaf = imfill(maskLeaf,'holes');


maskLeaf = bwareaopen(maskLeaf,500);


maskLeaf = imclose(maskLeaf,strel('disk',8));


%% =====================================================
% 7) Ultimo controllo componente maggiore
% ======================================================

CC = bwconncomp(maskLeaf);


if CC.NumObjects > 1

    areas = cellfun(@numel,CC.PixelIdxList);

    [~,idx] = max(areas);

    finalMask=false(size(maskLeaf));

    finalMask(CC.PixelIdxList{idx})=true;

    maskLeaf=finalMask;

end


end
function maskLeaf = removeBackgroundSuperpixelOld(rgbImage)
    numSuperpixels = 5000;
    [L,N] = superpixels(rgbImage, numSuperpixels);
    
    lab = rgb2lab(rgbImage);
    Lch = lab(:,:,1);
    
    hsv = rgb2hsv(rgbImage);
    S = hsv(:,:,2);
    
    meanL = zeros(N,1);
    meanS = zeros(N,1);
    
    for i = 1:N
        mask = (L == i);
        meanL(i) = mean(Lch(mask));
        meanS(i) = mean(S(mask));
    end
    
    score = meanS .* (1 - mat2gray(meanL));
    t = graythresh(score);
    selectedLabels = find(score > t);
    
    maskLeaf = ismember(L, selectedLabels);
    maskLeaf = imfill(maskLeaf,'holes');
    maskLeaf = bwareaopen(maskLeaf, 500);
    maskLeaf = imclose(maskLeaf, strel('disk',10));
end