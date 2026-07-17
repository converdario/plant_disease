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
folder = fullfile(currentScriptFolder, '..', 'public', 'dataset', 'peacock_spot'); % Directory dove prelevare le immagini

imageFiles = dir(fullfile(folder, '*.jpg'));
if isempty(imageFiles)
    imageFiles = dir(fullfile(folder, '*.JPG'));
end
num_images = length(imageFiles);

varNames = {...
'ImageName', ...
'Infection_Percentage', ...
'KMeans_Energy', ...
'KMeans_Contrast', ...
'KMeans_Correlation', ...
'KMeans_Homogeneity', ...
'KMeans_Entropy', ...
'Hist_Energy', ...
'Hist_Contrast', ...
'Hist_Correlation', ...
'Hist_Homogeneity', ...
'Hist_Entropy'};

varTypes = [{'string'}, repmat({'double'},1,length(varNames)-1)];

allExtractedFeatures = table( ...
    'Size',[0,length(varNames)], ...
    'VariableTypes',varTypes,...
    'VariableNames',varNames);

imageFeatures = table( ...
    'Size',[0,length(varNames)], ...
    'VariableTypes',varTypes,...
    'VariableNames',varNames);

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

for k = 1:num_images
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
   abImage = im2single(cat(3,a,b));

    pixelsAB = reshape(abImage,[],2);
    
    leafPixels = find(maskLeaf);
    
    dataLeaf = pixelsAB(leafPixels,:);
    
    
    [idx,centers] = kmeans(...
        dataLeaf,...
        NUM_CLUSTERS,...
        'Replicates',5);
    
    
    pixelLabels=zeros(size(maskLeaf));
    
    pixelLabels(leafPixels)=idx;
    
    %% Colore medio della foglia

    abPixels = reshape(abImage,[],2);
    
    leafPixels = find(maskLeaf);
    
    meanLeafAB = mean(abPixels(leafPixels,:),1);
    
    
    %% Distanza dei cluster dal colore medio foglia
    
    distanceFromLeaf = sqrt( ...
        (centers(:,1)-meanLeafAB(1)).^2 + ...
        (centers(:,2)-meanLeafAB(2)).^2 );
    
    
    %% Il cluster più distante rappresenta la macchia
    
    [~,diseaseCluster] = max(distanceFromLeaf);


    %% Maschera malattia
    
    maskKMeans = false(size(maskLeaf));
    
    maskKMeans(leafPixels)=idx==diseaseCluster;
    
    
    %% Il cluster più diverso è la macchia
    
    [~, diseaseCluster] = max(distanceFromLeaf);
    
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

    %% Salvataggio ROI KMeans

    ROI_kmeans = rgbImage;
    
    for ch=1:3
        
        temp = ROI_kmeans(:,:,ch);
        
        temp(~maskKMeans)=0;
        
        ROI_kmeans(:,:,ch)=temp;
    
    end
    
    
    imwrite(ROI_kmeans,...
    fullfile(outputKMeans,...
    ['K_',imageFiles(k).name]));

    



    %% --- THRESHOLDING SU CANALE a* (Metodo di confronto) ---
    aChannel = mat2gray(a);
    level = graythresh(aChannel);
    maskHist = imbinarize(aChannel,level);

    % scelgo la polarità con meno pixel
    if sum(maskHist(:)) > 0.5*sum(maskLeaf(:))
        maskHist = ~maskHist;
    end
    
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

    I_gray_noBackground = I_gray_original;

    I_gray_noBackground(~maskLeaf)=0;

    %% === ROI KMEANS ===

    featuresK = computeROITextureFeatures( ...
        I_gray_original,...
        maskKMeans,...
        GLCM_OFFSETS,...
        GLCM_NUM_LEVELS,...
        GLCM_SYMMETRIC);


    % === ROI HISTOGRAM ===
    featuresH = computeROITextureFeatures(...
        I_gray_original,...
        maskHist,...
        GLCM_OFFSETS,...
        GLCM_NUM_LEVELS,...
        GLCM_SYMMETRIC);


values = double([ ...
infection_percentage,...
featuresK.Energy,...
featuresK.Contrast,...
featuresK.Correlation,...
featuresK.Homogeneity,...
featuresK.Entropy,...
featuresH.Energy,...
featuresH.Contrast,...
featuresH.Correlation,...
featuresH.Homogeneity,...
featuresH.Entropy
]);


newRow = array2table(values,...
    'VariableNames',varNames(2:end));


newRow = addvars(newRow,...
    string(imageFiles(k).name),...
    'Before',1,...
    'NewVariableNames','ImageName');


allExtractedFeatures = [allExtractedFeatures; newRow];  
imageFeatures = [imageFeatures; newRow];
    
    fprintf('Elaborata immagine %d/%d: %s (Area infetta: %.2f%%)\n', k, num_images, imageFiles(k).name, infection_percentage);
end

fprintf('\n==============================================================\n');
fprintf('FEATURE PER SINGOLA IMMAGINE\n');
fprintf('==============================================================\n');


for i = 1:height(imageFeatures)

    fprintf('\nImmagine: %s\n', imageFeatures.ImageName(i));

    fprintf('Infezione: %.2f %%\n',...
        imageFeatures.Infection_Percentage(i));


    %% ============================
    % KMEANS ROI
    % ============================

    fprintf('\n--- KMeans ROI ---\n');


    fprintf('Energy       %.4f\n',...
        imageFeatures.KMeans_Energy(i));

    fprintf('Contrast     %.4f\n',...
        imageFeatures.KMeans_Contrast(i));

    fprintf('Correlation  %.4f\n',...
        imageFeatures.KMeans_Correlation(i));

    fprintf('Homogeneity  %.4f\n',...
        imageFeatures.KMeans_Homogeneity(i));

    fprintf('Entropy      %.4f\n',...
        imageFeatures.KMeans_Entropy(i));



    %% ============================
    % HISTOGRAM ROI
    % ============================

    fprintf('\n--- Histogram ROI ---\n');


    fprintf('Energy       %.4f\n',...
        imageFeatures.Hist_Energy(i));

    fprintf('Contrast     %.4f\n',...
        imageFeatures.Hist_Contrast(i));

    fprintf('Correlation  %.4f\n',...
        imageFeatures.Hist_Correlation(i));

    fprintf('Homogeneity  %.4f\n',...
        imageFeatures.Hist_Homogeneity(i));

    fprintf('Entropy      %.4f\n',...
        imageFeatures.Hist_Entropy(i));

end

fprintf('\n==============================================================\n');

%% =========================================================================
%% SALVATAGGIO FINALE E CALCOLO CORRELAZIONE
%% =========================================================================

outputFilePath = fullfile(folder, 'all_extracted_texture_features.mat');
save(outputFilePath, ...
    'allExtractedFeatures',...
    'imageFeatures');

fprintf('\nTutte le feature sono state estratte e salvate in:\n%s\n', outputFilePath);

% Rimuove le righe dove la percentuale di infezione è NaN
validRows = ~isnan(allExtractedFeatures.Infection_Percentage);
validFeatures = allExtractedFeatures(validRows, :);

if height(validFeatures) > 1
    % Isoliamo il target
    inf_perc = validFeatures.Infection_Percentage;
    
    fprintf('\n====================================================================================\n');
    fprintf('ANALISI DESCRITTIVA DELLE FEATURE E CORRELAZIONE CON IL TARGET\n');
    fprintf('====================================================================================\n');
    
    % Estraiamo tutti i nomi delle colonne
    allVarNames = validFeatures.Properties.VariableNames;
    
    % Escludiamo 'ImageName' e il target 'Infection_Percentage' dal ciclo
    featuresToProcess = setdiff(allVarNames, {'ImageName', 'Infection_Percentage'}, 'stable');

    for i = 1:length(featuresToProcess)
        featureName = featuresToProcess{i};
        featureData = validFeatures.(featureName);
        
        % Calcolo della Media della feature
        mean_val = mean(featureData, 'omitnan');
        
        % Filtro validità per allineare gli array nel calcolo di Pearson
        validIdx = ~isnan(featureData) & ~isnan(inf_perc);
        
        if sum(validIdx) > 1
            % Calcolo matrice di correlazione tra feature corrente e target
            R_matrix = corrcoef(featureData(validIdx), inf_perc(validIdx));
            r_val = R_matrix(1,2);
        else
            r_val = NaN;
        end
        
        % Stampa dei risultati formattata
        fprintf('%-22s : Media = %10.4f | r (vs %% Inf) = %7.4f\n', ...
                featureName, mean_val, r_val);
    end
    fprintf('====================================================================================\n');
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

function features = extractGLCMFeatures(grayROI, offsets, numLevels, symmetric)

%==========================================================================
% Estrazione feature GLCM
%
% INPUT
%   grayROI     : immagine grayscale
%   offsets     : offset GLCM
%   numLevels   : numero livelli GLCM
%   symmetric   : true/false
%
% OUTPUT
%   features.Energy
%   features.Contrast
%   features.Correlation
%   features.Homogeneity
%   features.Entropy
%==========================================================================

grayROI = im2uint8(mat2gray(grayROI));

glcm = graycomatrix(grayROI,...
    'Offset',offsets,...
    'NumLevels',numLevels,...
    'Symmetric',symmetric);

props = graycoprops(glcm);

entropyValues = zeros(1,size(glcm,3));

for i=1:size(glcm,3)

    P = glcm(:,:,i);

    sumP = sum(P(:));

if sumP > 0
    P = P./sumP;
else
    entropyValues(i)=NaN;
    continue
end

    P = P(P>0);

    entropyValues(i) = -sum(P.*log(P));

end

features.Energy      = mean(props.Energy);
features.Contrast    = mean(props.Contrast);
features.Correlation = mean(props.Correlation);
features.Homogeneity = mean(props.Homogeneity);
features.Entropy     = mean(entropyValues);

end

function features = computeROITextureFeatures( ...
                    grayImage,...
                    roiMask,...
                    offsets,...
                    numLevels,...
                    symmetric)

%=====================================================================
% Estrazione texture GLCM secondo approccio paper:
%
% Olive Spot Disease Detection and Classification using Analysis of
% Leaf Image Textures
%
% Feature:
%   - Energy
%   - Contrast
%   - Correlation
%   - Homogeneity
%   - Entropy
%
% La GLCM viene calcolata solamente sulla ROI segmentata.
%=====================================================================


features = struct();

names = {'Energy',...
         'Contrast',...
         'Correlation',...
         'Homogeneity',...
         'Entropy'};


if sum(roiMask(:))==0

    for i=1:length(names)
        features.(names{i}) = NaN;
    end

    return

end


%% ROI malattia

roi = grayImage;

roi(~roiMask)=0;


%% Bounding box automatica solo per ridurre dimensione

stats = regionprops(roiMask,'BoundingBox');

bbox = stats(1).BoundingBox;


roi = imcrop(roi,bbox);
mask = imcrop(roiMask,bbox);


roi(~mask)=0;


%% GLCM

features = extractGLCMFeatures(...
    roi,...
    offsets,...
    numLevels,...
    symmetric);

end