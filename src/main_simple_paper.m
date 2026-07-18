clear;
clc;
close all;

warning('off', 'images:graycomatrix:scaledImageContainsNan');
warning('off', 'stats:kmeans:FailedToConvergeRep');

%% =========================================================================
% OLIVE SPOT DISEASE DETECTION
%
% Replica della pipeline:
%
% 1. Lettura immagine
% 2. Rimozione dello sfondo
% 3. Segmentazione della ROI con K-Means
% 4. Segmentazione della ROI tramite thresholding del canale a*
% 5. Estrazione delle feature GLCM
% 6. Calcolo della percentuale di area infetta
% 7. Correlazione tra texture e percentuale di infezione
%
% =========================================================================

%% =========================================================================
% CONFIGURAZIONE
% =========================================================================

NUM_CLUSTERS = 3;

GLCM_OFFSETS = [
0  1;     % 0°
-1  1;     % 45°
-1  0;     % 90°
-1 -1      % 135°
];

GLCM_SYMMETRIC = true;

% Numero di livelli di grigio per la GLCM
GLCM_NUM_LEVELS = 128;

%% =========================================================================
% CARTELLE
% =========================================================================

currentScriptFolder = fileparts(mfilename('fullpath'));

folder = fullfile( ...
currentScriptFolder, ...
'..', ...
'public', ...
'dataset', ...
'peacock_spot');

outputKMeans = fullfile(folder, 'ROI_KMeans');

outputHist = fullfile(folder, 'ROI_Histogram');

outputNoBackground = fullfile(folder, 'NoBackground');

if ~exist(outputKMeans, 'dir')
mkdir(outputKMeans);
end

if ~exist(outputHist, 'dir')
mkdir(outputHist);
end

if ~exist(outputNoBackground, 'dir')
mkdir(outputNoBackground);
end

%% =========================================================================
% LETTURA IMMAGINI
% =========================================================================

imageFiles = dir(fullfile(folder, '*.jpg'));

if isempty(imageFiles)
imageFiles = dir(fullfile(folder, '*.JPG'));
end

num_images = length(imageFiles);

if num_images == 0

error( ...
    'Nessuna immagine JPG trovata nella cartella:\n%s', ...
    folder);

end

fprintf('\nNumero immagini trovate: %d\n', num_images);

%% =========================================================================
% TABELLA RISULTATI
% =========================================================================

varNames = { ...
    'ImageName', ...
    'KMeans_Infection_Percentage', ...
    'KMeans_Energy', ...
    'KMeans_Contrast', ...
    'KMeans_Correlation', ...
    'KMeans_Homogeneity', ...
    'KMeans_Entropy', ...
    'Histogram_Infection_Percentage', ...
    'Hist_Energy', ...
    'Hist_Contrast', ...
    'Hist_Correlation', ...
    'Hist_Homogeneity', ...
    'Hist_Entropy' ...
};

varTypes = [ ...
    {'string'}, ...
    repmat({'double'}, 1, length(varNames)-1) ...
];

allFeatures = table( ...
'Size', ...
[0, length(varNames)], ...
'VariableTypes', ...
varTypes, ...
'VariableNames', ...
varNames);


%% =========================================================================
% CICLO PRINCIPALE
% =========================================================================

for k = 1:20

fprintf('\n============================================================\n');

fprintf( ...
    'Elaborazione immagine %d/%d: %s\n', ...
    k, ...
    num_images, ...
    imageFiles(k).name);


%% -------------------------------------------------------------
% LETTURA IMMAGINE
%% -------------------------------------------------------------

filename = fullfile( ...
    folder, ...
    imageFiles(k).name);


rgbImage = imread(filename);


% Assicura un'immagine RGB
if size(rgbImage, 3) == 1

    rgbImage = repmat(rgbImage, [1 1 3]);

end


%% -------------------------------------------------------------
% CONVERSIONE LAB
%% -------------------------------------------------------------

labImage = rgb2lab(rgbImage);


L = labImage(:,:,1);

a = labImage(:,:,2);

b = labImage(:,:,3);


%% =============================================================
% PARTE 1
% RIMOZIONE DELLO SFONDO
% =============================================================

maskLeaf = removeBackground(rgbImage);


% Se la segmentazione fallisce
if nnz(maskLeaf) == 0

    warning( ...
        'Maschera foglia vuota per %s', ...
        imageFiles(k).name);


    continue;

end


%% -------------------------------------------------------------
% SALVATAGGIO IMMAGINE SENZA SFONDO
%% -------------------------------------------------------------

rgbNoBackground = rgbImage;


for ch = 1:3

    temp = rgbNoBackground(:,:,ch);

    temp(~maskLeaf) = 0;

    rgbNoBackground(:,:,ch) = temp;

end


imwrite( ...
    rgbNoBackground, ...
    fullfile( ...
        outputNoBackground, ...
        ['NB_' imageFiles(k).name]));


%% =============================================================
% PARTE 2
% K-MEANS SEGMENTATION
% =============================================================


fprintf('Esecuzione K-Means...\n');

% Pixel appartenenti alla foglia
leafPixels = find(maskLeaf);
leafPixels = leafPixels(:);

L_values = L(leafPixels);
a_values = a(leafPixels);
b_values = b(leafPixels);

dataLAB = [L_values, a_values, b_values];

% Controllo dimensioni
fprintf('Numero pixel foglia: %d\n', numel(leafPixels));
fprintf('Numero righe dataLAB: %d\n', size(dataLAB,1));

[clusterLabels, clusterCenters] = kmeans( ...
    dataLAB, ...
    NUM_CLUSTERS, ...
    'Distance', 'sqeuclidean', ...
    'Replicates', 10, ...
    'MaxIter', 500, ...
    'Start', 'plus');

clusterLabels = clusterLabels(:);

% Controllo finale
if numel(leafPixels) ~= numel(clusterLabels)

    error( ...
        'Errore K-Means: pixel foglia = %d, labels = %d', ...
        numel(leafPixels), ...
        numel(clusterLabels));

end

pixelLabels = zeros(size(maskLeaf));

pixelLabels(leafPixels) = clusterLabels;


%% =============================================================
% SELEZIONE DEL CLUSTER INFETTO
% =============================================================

% Il metodo utilizza il clustering per separare le regioni
% cromaticamente differenti della foglia.
%
% Per selezionare la regione candidata alla malattia,
% viene utilizzato il cluster cromaticamente più distante
% dal colore medio della foglia.
%
% Questa selezione è più coerente con il concetto di
% segmentazione per anomalia cromatica rispetto a imporre
% che la malattia sia sempre più scura.


meanLeafA = mean(a(maskLeaf));

meanLeafB = mean(b(maskLeaf));


clusterDistances = zeros(NUM_CLUSTERS, 1);


for c = 1:NUM_CLUSTERS


    clusterMask = ...
        pixelLabels == c & maskLeaf;


    if nnz(clusterMask) == 0

        clusterDistances(c) = -Inf;

        continue;

    end


    clusterMeanA = mean(a(clusterMask));

    clusterMeanB = mean(b(clusterMask));


    clusterDistances(c) = sqrt( ...
        (clusterMeanA - meanLeafA)^2 + ...
        (clusterMeanB - meanLeafB)^2);


end


% Cluster cromaticamente più distante
[~, diseaseCluster] = max(clusterDistances);


maskKMeans = ...
    pixelLabels == diseaseCluster;


maskKMeans = ...
    maskKMeans & maskLeaf;


%% -------------------------------------------------------------
% PULIZIA MINIMA DELLA ROI K-MEANS
%% -------------------------------------------------------------

maskKMeans = bwareaopen( ...
    maskKMeans, ...
    20);


maskKMeans = imopen( ...
    maskKMeans, ...
    strel('disk', 1));


maskKMeans = maskKMeans & maskLeaf;


%% -------------------------------------------------------------
% PERCENTUALE AREA INFETTA K-MEANS
%% -------------------------------------------------------------

leafPixelsCount = nnz(maskLeaf);

diseasePixelsCount = nnz(maskKMeans);


if leafPixelsCount > 0
    kmeansInfectionPercentage = ...
        100 * ...
        diseasePixelsCount / ...
        leafPixelsCount;

else

    kmeansInfectionPercentage = NaN;

end


%% -------------------------------------------------------------
% SALVATAGGIO ROI K-MEANS
%% -------------------------------------------------------------

ROI_KMeans = rgbImage;


for ch = 1:3

    temp = ROI_KMeans(:,:,ch);

    temp(~maskKMeans) = 0;

    ROI_KMeans(:,:,ch) = temp;

end


imwrite( ...
    ROI_KMeans, ...
    fullfile( ...
        outputKMeans, ...
        ['K_' imageFiles(k).name]));


%% =============================================================
% PARTE 3
% HISTOGRAM THRESHOLDING SUL CANALE a*
% =============================================================


fprintf('Esecuzione thresholding canale a*...\n');


% Si considerano solamente i pixel della foglia.
%
% Questo evita che lo sfondo influenzi l'istogramma.


aLeaf = a(maskLeaf);


% Normalizzazione del canale a*
aLeafNormalized = mat2gray(aLeaf);


% Soglia globale di Otsu
level = graythresh(aLeafNormalized);


% Canale a* normalizzato dell'intera immagine
aNormalized = mat2gray(a);


% Due possibili polarità
maskHist1 = ...
    aNormalized >= level;


maskHist2 = ...
    aNormalized < level;


% Limite alla foglia
maskHist1 = maskHist1 & maskLeaf;

maskHist2 = maskHist2 & maskLeaf;


%% -------------------------------------------------------------
% SELEZIONE DELLA POLARITÀ
%% -------------------------------------------------------------

% Il canale a* può separare la regione malata in una delle
% due polarità.
%
% Viene scelta la regione più piccola, assumendo che la
% regione malata sia una porzione minoritaria della foglia.


area1 = nnz(maskHist1);

area2 = nnz(maskHist2);


if area1 < area2

    maskHist = maskHist1;

else

    maskHist = maskHist2;

end


%% -------------------------------------------------------------
% PULIZIA MINIMA DELLA MASCHERA HISTOGRAM
%% -------------------------------------------------------------

maskHist = bwareaopen( ...
    maskHist, ...
    20);


maskHist = imopen( ...
    maskHist, ...
    strel('disk', 1));


maskHist = maskHist & maskLeaf;


%% -------------------------------------------------------------
% PERCENTUALE AREA INFETTA HISTOGRAM
%% -------------------------------------------------------------

histDiseasePixels = nnz(maskHist);


if leafPixelsCount > 0

    histInfectionPercentage = ...
        100 * ...
        histDiseasePixels / ...
        leafPixelsCount;

else

    histInfectionPercentage = NaN;

end


%% -------------------------------------------------------------
% SALVATAGGIO ROI HISTOGRAM
%% -------------------------------------------------------------

ROI_Hist = rgbImage;


for ch = 1:3

    temp = ROI_Hist(:,:,ch);

    temp(~maskHist) = 0;
    ROI_Hist(:,:,ch) = temp;

end


imwrite( ...
    ROI_Hist, ...
    fullfile( ...
        outputHist, ...
        ['H_' imageFiles(k).name]));


%% =============================================================
% PARTE 4
% CONVERSIONE IN GRAYSCALE
% =============================================================


grayImage = rgb2gray(rgbImage);


%% =============================================================
% PARTE 5
% GLCM - K-MEANS ROI
% =============================================================


featuresKMeans = ...
    computeROITextureFeatures( ...
        grayImage, ...
        maskKMeans, ...
        GLCM_OFFSETS, ...
        GLCM_NUM_LEVELS, ...
        GLCM_SYMMETRIC);


%% =============================================================
% PARTE 6
% GLCM - HISTOGRAM ROI
% =============================================================


featuresHist = ...
    computeROITextureFeatures( ...
        grayImage, ...
        maskHist, ...
        GLCM_OFFSETS, ...
        GLCM_NUM_LEVELS, ...
        GLCM_SYMMETRIC);


%% =============================================================
% CREAZIONE RIGA RISULTATI
% =============================================================


values = [

    kmeansInfectionPercentage

    featuresKMeans.Energy

    featuresKMeans.Contrast

    featuresKMeans.Correlation

    featuresKMeans.Homogeneity

    featuresKMeans.Entropy

    histInfectionPercentage

    featuresHist.Energy

    featuresHist.Contrast

    featuresHist.Correlation

    featuresHist.Homogeneity

    featuresHist.Entropy

]';


newRow = array2table( ...
    values, ...
    'VariableNames', ...
    varNames(2:end));


newRow = addvars( ...
    newRow, ...
    string(imageFiles(k).name), ...
    'Before', ...
    1, ...
    'NewVariableNames', ...
    'ImageName');


allFeatures = [

    allFeatures

    newRow

];


%% -------------------------------------------------------------
% OUTPUT
%% -------------------------------------------------------------

fprintf( ...
    'K-Means area infetta: %.2f%%\n', ...
    kmeansInfectionPercentage);


fprintf( ...
    'Histogram area infetta: %.2f%%\n', ...
    histInfectionPercentage);

end

%% =========================================================================
% SALVATAGGIO RISULTATI
% =========================================================================

outputFilePath = fullfile( ...
folder, ...
'paper_replication_texture_features.mat');


save( ...
outputFilePath, ...
'allFeatures');


fprintf('\n');

fprintf('============================================================\n');

fprintf('RISULTATI SALVATI IN:\n');

fprintf('%s\n', outputFilePath);

fprintf('============================================================\n');

%% =========================================================================
% CORRELAZIONE CON PERCENTUALE AREA INFETTA
% =========================================================================

fprintf('\n');

fprintf('============================================================\n');

fprintf('CORRELAZIONI K-MEANS\n');

fprintf('============================================================\n');

validRows = ...
~isnan(allFeatures.KMeans_Infection_Percentage);


validFeatures = ...
allFeatures(validRows, :);


if height(validFeatures) > 1


target = ...
    validFeatures.KMeans_Infection_Percentage;


featureNames = {

    'KMeans_Energy'

    'KMeans_Contrast'

    'KMeans_Correlation'

    'KMeans_Homogeneity'

    'KMeans_Entropy'

};


for i = 1:length(featureNames)


    featureName = featureNames{i};


    featureData = ...
        validFeatures.(featureName);


    validIdx = ...
        ~isnan(featureData) & ...
        ~isnan(target);


    if nnz(validIdx) > 1


        R = corrcoef( ...
            featureData(validIdx), ...
            target(validIdx));


        r = R(1,2);


    else


        r = NaN;


    end


    fprintf( ...
        '%-25s : r = %7.4f\n', ...
        featureName, ...
        r);


end


else


fprintf( ...
    'Numero insufficiente di immagini valide.\n');


end

fprintf('\n');

fprintf('============================================================\n');

fprintf('CORRELAZIONI HISTOGRAM\n');

fprintf('============================================================\n');

validRows = ...
~isnan(allFeatures.Histogram_Infection_Percentage);


validFeatures = ...
allFeatures(validRows, :);

if height(validFeatures) > 1

target = ...
    validFeatures.Histogram_Infection_Percentage;


featureNames = {

    'Hist_Energy'

    'Hist_Contrast'

    'Hist_Correlation'

    'Hist_Homogeneity'

    'Hist_Entropy'

};


for i = 1:length(featureNames)


    featureName = featureNames{i};


    featureData = ...
        validFeatures.(featureName);


    validIdx = ...
        ~isnan(featureData) & ...
        ~isnan(target);


    if nnz(validIdx) > 1


        R = corrcoef( ...
            featureData(validIdx), ...
            target(validIdx));


        r = R(1,2);


    else


        r = NaN;


    end


    fprintf( ...
        '%-25s : r = %7.4f\n', ...
        featureName, ...
        r);


end


else

fprintf( ...
    'Numero insufficiente di immagini valide.\n');

end

%% =========================================================================
% FUNZIONE RIMOZIONE SFONDO
% =========================================================================

function maskLeaf = removeBackground(rgbImage)

%% -------------------------------------------------------------
% CONVERSIONE HSV
%% -------------------------------------------------------------

hsvImage = rgb2hsv(rgbImage);


S = hsvImage(:,:,2);

V = hsvImage(:,:,3);


%% -------------------------------------------------------------
% SEGMENTAZIONE INIZIALE
%% -------------------------------------------------------------

% La saturazione è utilizzata per separare la foglia
% dallo sfondo.


thresholdS = graythresh(S);


maskLeaf = S > thresholdS;


%% -------------------------------------------------------------
% PULIZIA
%% -------------------------------------------------------------

maskLeaf = imopen( ...
    maskLeaf, ...
    strel('disk', 3));


maskLeaf = imclose( ...
    maskLeaf, ...
    strel('disk', 10));


maskLeaf = imfill( ...
    maskLeaf, ...
    'holes');


maskLeaf = bwareaopen( ...
    maskLeaf, ...
    500);


%% -------------------------------------------------------------
% MANTIENI LA COMPONENTE PRINCIPALE
%% -------------------------------------------------------------

CC = bwconncomp(maskLeaf);


if CC.NumObjects > 0


    areas = ...
        cellfun(@numel, CC.PixelIdxList);


    [~, largestComponent] = ...
        max(areas);


    newMask = ...
        false(size(maskLeaf));


    newMask( ...
        CC.PixelIdxList{largestComponent}) = true;


    maskLeaf = newMask;


end


%% -------------------------------------------------------------
% RAFFINAMENTO MORFOLOGICO
%% -------------------------------------------------------------

maskLeaf = imclose( ...
    maskLeaf, ...
    strel('disk', 5));


maskLeaf = imfill( ...
    maskLeaf, ...
    'holes');


%% -------------------------------------------------------------
% ULTIMO CONTROLLO
%% -------------------------------------------------------------

CC = bwconncomp(maskLeaf);


if CC.NumObjects > 1


    areas = ...
        cellfun(@numel, CC.PixelIdxList);


    [~, largestComponent] = ...
        max(areas);


    finalMask = ...
        false(size(maskLeaf));


    finalMask( ...
        CC.PixelIdxList{largestComponent}) = true;


    maskLeaf = finalMask;


end


end

%% =========================================================================
% ESTRAZIONE FEATURE GLCM
% =========================================================================

function features = computeROITextureFeatures( ...
    grayImage, ...
    roiMask, ...
    offsets, ...
    numLevels, ...
    symmetric)


featureNames = {

    'Energy'

    'Contrast'

    'Correlation'

    'Homogeneity'

    'Entropy'

};


%% -------------------------------------------------------------
% ROI VUOTA
%% -------------------------------------------------------------

if nnz(roiMask) == 0


    for i = 1:length(featureNames)

        features.(featureNames{i}) = NaN;

    end


    return;


end


%% -------------------------------------------------------------
% BOUNDING BOX
%% -------------------------------------------------------------

stats = ...
    regionprops( ...
        roiMask, ...
        'BoundingBox');


if isempty(stats)


    for i = 1:length(featureNames)

        features.(featureNames{i}) = NaN;

    end


    return;


end


% Per sicurezza, si considera la componente principale
% della ROI.


CC = bwconncomp(roiMask);


areas = ...
    cellfun(@numel, CC.PixelIdxList);


[~, largestComponent] = ...
    max(areas);


mainMask = ...
    false(size(roiMask));


mainMask( ...
    CC.PixelIdxList{largestComponent}) = true;


bbox = ...
    regionprops( ...
        mainMask, ...
        'BoundingBox');


bbox = bbox.BoundingBox;


%% -------------------------------------------------------------
% CROP
%% -------------------------------------------------------------

x1 = max(1, floor(bbox(1)));

y1 = max(1, floor(bbox(2)));

x2 = min( ...
    size(grayImage,2), ...
    ceil(bbox(1) + bbox(3)));


y2 = min( ...
    size(grayImage,1), ...
    ceil(bbox(2) + bbox(4)));


roiGray = ...
    grayImage(y1:y2, x1:x2);


roiMaskCrop = ...
    mainMask(y1:y2, x1:x2);


%% -------------------------------------------------------------
% CONVERSIONE A DOUBLE
%% -------------------------------------------------------------

roiGray = double(roiGray);


%% -------------------------------------------------------------
% QUANTIZZAZIONE
%% -------------------------------------------------------------

% La GLCM viene calcolata utilizzando l'intera bounding box,
% ma i pixel esterni alla ROI vengono esclusi tramite NaN.


roiGray(~roiMaskCrop) = NaN;


%% -------------------------------------------------------------
% GLCM
%% -------------------------------------------------------------

glcm = graycomatrix( ...
    roiGray, ...
    'Offset', ...
    offsets, ...
    'NumLevels', ...
    numLevels, ...
    'GrayLimits', ...
    [0 255], ...
    'Symmetric', ...
    symmetric);


%% -------------------------------------------------------------
% PROPRIETÀ STANDARD
%% -------------------------------------------------------------

props = ...
    graycoprops(glcm);


%% -------------------------------------------------------------
% ENTROPIA
%% -------------------------------------------------------------

entropyValues = ...
    NaN(1, size(glcm,3));


for i = 1:size(glcm,3)


    P = glcm(:,:,i);


    total = sum(P(:));


    if total == 0

        continue;

    end


    P = P ./ total;


    P = P(P > 0);


    entropyValues(i) = ...
        -sum(P .* log2(P));


end


%% -------------------------------------------------------------
% MEDIA DELLE QUATTRO DIREZIONI
%% -------------------------------------------------------------

features.Energy = ...
    mean(props.Energy, 'omitnan');
features.Contrast = ...
    mean(props.Contrast, 'omitnan');
features.Correlation = ...
    mean(props.Correlation, 'omitnan');
features.Homogeneity = ...
    mean(props.Homogeneity, 'omitnan');
features.Entropy = ...
    mean(entropyValues, 'omitnan');

end
