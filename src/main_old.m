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

%% ============================================================
%% SEGMENTAZIONE MALATTIA
%% K-MEANS + LAB ANOMALY SCORE
%% + THRESHOLDING LOCALE + MORFOLOGIA
%% ============================================================


%% ------------------------------------------------------------
% 1) Conversione nello spazio LAB
%% ------------------------------------------------------------

labImage = rgb2lab(rgbImage);

L = labImage(:,:,1);
a = labImage(:,:,2);
b = labImage(:,:,3);


%% ------------------------------------------------------------
% 2) Estrazione dei soli pixel della foglia
%% ------------------------------------------------------------

leafPixels = find(maskLeaf);

dataLab = [ ...
    L(leafPixels), ...
    a(leafPixels), ...
    b(leafPixels)];


%% ------------------------------------------------------------
% 3) K-MEANS
%% ------------------------------------------------------------

[idx, centers] = kmeans( ...
    dataLab, ...
    NUM_CLUSTERS, ...
    'Replicates', 10, ...
    'MaxIter', 500, ...
    'Start', 'plus');


%% ------------------------------------------------------------
% 4) Ricostruzione della mappa dei cluster
%% ------------------------------------------------------------

pixelLabels = zeros(size(maskLeaf));

pixelLabels(leafPixels) = idx;


%% ------------------------------------------------------------
% 5) COLORE MEDIO DELLA FOGLIA
%% ------------------------------------------------------------

meanLeafL = mean(L(maskLeaf));
meanLeafA = mean(a(maskLeaf));
meanLeafB = mean(b(maskLeaf));


stdLeafL = std(L(maskLeaf));
stdLeafA = std(a(maskLeaf));
stdLeafB = std(b(maskLeaf));


%% ============================================================
%% 6) ANALISI DEI CLUSTER
%% ============================================================

clusterScore = zeros(NUM_CLUSTERS,1);

clusterColorDistance = zeros(NUM_CLUSTERS,1);

clusterDarkness = zeros(NUM_CLUSTERS,1);

clusterArea = zeros(NUM_CLUSTERS,1);


for c = 1:NUM_CLUSTERS

    clusterMask = ...
        (pixelLabels == c) & maskLeaf;


    if sum(clusterMask(:)) == 0

        clusterScore(c) = -Inf;

        continue

    end


    % Colore medio del cluster
    clusterMeanL = mean(L(clusterMask));
    clusterMeana = mean(a(clusterMask));
    clusterMeanb = mean(b(clusterMask));


    % Distanza cromatica dal colore medio della foglia
    colorDistance = sqrt( ...
        (clusterMeana - meanLeafA)^2 + ...
        (clusterMeanb - meanLeafB)^2);


    % Oscurità relativa rispetto alla foglia
    darkness = max( ...
        0, ...
        meanLeafL - clusterMeanL);


    % Percentuale di area occupata
    areaRatio = ...
        sum(clusterMask(:)) / sum(maskLeaf(:));


    % Normalizzazione area
    % Penalizza cluster enormi che rappresentano probabilmente
    % la foglia sana
    areaPenalty = 1 - areaRatio;


    clusterColorDistance(c) = colorDistance;

    clusterDarkness(c) = darkness;

    clusterArea(c) = areaRatio;


    % Score del cluster
    %clusterScore(c) = ...
    %    0.50 * colorDistance + ...
    %    0.40 * darkness + ...
    %    0.10 * areaPenalty;

end

%% ============================================================
%% NORMALIZZAZIONE DEI PUNTEGGI DEI CLUSTER
%% ============================================================

colorDistanceNorm = normalizeRobust(clusterColorDistance);
darknessNorm      = normalizeRobust(clusterDarkness);

areaPenaltyNorm = normalizeRobust(areaPenalty);


clusterScore = ...
    0.50 * colorDistanceNorm + ...
    0.40 * darknessNorm + ...
    0.10 * areaPenaltyNorm;
%% ============================================================
%% 7) SELEZIONE DEI CLUSTER CANDIDATI
%% ============================================================

% Ordina i cluster in base allo score
[~, sortedClusters] = sort( ...
    clusterScore, ...
    'descend');


% Selezione dei due cluster più anomali
%numCandidateClusters = min(2, NUM_CLUSTERS);
numCandidateClusters = 1;

candidateClusters = ...
    sortedClusters(1:numCandidateClusters);


% Maschera dei cluster candidati
maskClusterCandidates = ...
    ismember(pixelLabels, candidateClusters);


maskClusterCandidates = ...
    maskClusterCandidates & maskLeaf;


%% ============================================================
%% 8) LAB ANOMALY SCORE PIXEL-WISE
%% ============================================================

% Distanza cromatica pixel-wise
colorAnomaly = sqrt( ...
    (a - meanLeafA).^2 + ...
    (b - meanLeafB).^2);


% Anomalia di luminosità:
% le regioni più scure della foglia ricevono valori maggiori
darknessAnomaly = ...
    max(0, meanLeafL - L);


%% ------------------------------------------------------------
% Normalizzazione robusta tramite percentili
%% ------------------------------------------------------------

colorValues = colorAnomaly(maskLeaf);

darknessValues = darknessAnomaly(maskLeaf);


colorLow = prctile(colorValues, 5);
colorHigh = prctile(colorValues, 95);


darkLow = prctile(darknessValues, 5);
darkHigh = prctile(darknessValues, 95);


colorAnomalyNorm = ...
    (colorAnomaly - colorLow) ./ ...
    (colorHigh - colorLow + eps);


darknessAnomalyNorm = ...
    (darknessAnomaly - darkLow) ./ ...
    (darkHigh - darkLow + eps);


% Limita i valori nell'intervallo [0,1]
colorAnomalyNorm = ...
    min(max(colorAnomalyNorm,0),1);


darknessAnomalyNorm = ...
    min(max(darknessAnomalyNorm,0),1);


%% ============================================================
%% 9) ANOMALY SCORE FINALE
%% ============================================================

% 50% anomalia cromatica
% 50% anomalia di luminosità

anomalyScore = ...
    0.65 * colorAnomalyNorm + ...
    0.35 * darknessAnomalyNorm;


% Fuori dalla foglia = non malato
anomalyScore(~maskLeaf) = 0;


%% ============================================================
%% 10) THRESHOLDING LOCALE
%% ============================================================

localThreshold = adaptthresh( ...
    anomalyScore, ...
    0.55, ...
    'NeighborhoodSize', [31 31], ...
    'Statistic', 'mean');


maskLocal = imbinarize( ...
    anomalyScore, ...
    localThreshold);


maskLocal = ...
    maskLocal & maskLeaf;


%% ============================================================
%% 11) COMBINAZIONE K-MEANS + ANOMALY SCORE
%% ============================================================

% Un pixel viene considerato candidato se:
%
% 1) appartiene a un cluster anomalo
% 2) supera il threshold locale
%
maskKMeans = ...
    maskClusterCandidates & maskLocal;


%% ============================================================
%% 12) FALLBACK
%% ============================================================

% Se la maschera è troppo piccola, utilizziamo
% la mappa di anomalia locale

minDiseasePixels = 50;


if sum(maskKMeans(:)) < minDiseasePixels

    maskKMeans = maskLocal;

end


%% ============================================================
%% 13) RIMOZIONE DEL RUMORE
%% ============================================================

maskKMeans = bwareaopen( ...
    maskKMeans, ...
    50);


%% ============================================================
%% 14) APERTURA MORFOLOGICA
%% ============================================================

maskKMeans = imopen( ...
    maskKMeans, ...
    strel('disk',2));


%% ============================================================
%% 15) CHIUSURA MORFOLOGICA
%% ============================================================

maskKMeans = imclose( ...
    maskKMeans, ...
    strel('disk',4));


%% ============================================================
%% 16) RIEMPIMENTO DEI BUCHI
%% ============================================================

maskKMeans = imfill( ...
    maskKMeans, ...
    'holes');


%% ============================================================
%% 17) LIMITAZIONE FINALE ALLA FOGLIA
%% ============================================================

maskKMeans = ...
    maskKMeans & maskLeaf;


%% ============================================================
%% 18) PERCENTUALE DI INFEZIONE
%% ============================================================

leaf_pixels = sum(maskLeaf(:));

disease_pixels = sum(maskKMeans(:));


if leaf_pixels > 0

    infection_percentage = ...
        (disease_pixels / leaf_pixels) * 100;

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

    %% === KMEANS FEATURES ===
    featuresK = computeROITextureFeatures( ...
        I_gray_original,...
        maskKMeans, ...
        GLCM_OFFSETS,...
        GLCM_NUM_LEVELS,...
        GLCM_SYMMETRIC);


    % === HISTOGRAM FEATURES ===
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
    fprintf('CORRELAZIONE CON PERCENTUALE AREA INFETTA\n');
    fprintf('====================================================================================\n');
    
    % Estraiamo tutti i nomi delle colonne
    allVarNames = validFeatures.Properties.VariableNames;
    
    % Escludiamo 'ImageName' e il target 'Infection_Percentage' dal ciclo
    featuresToProcess = setdiff(allVarNames, {'ImageName', 'Infection_Percentage'}, 'stable');

    for i = 1:length(featuresToProcess)
        featureName = featuresToProcess{i};
        featureData = validFeatures.(featureName);
        
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
        fprintf('%-22s : r = %7.4f\n', ...
                featureName, r_val);
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

% Rimuoviamo im2uint8 e mat2gray: se normalizzassimo l'immagine, 
% comprometteremmo i NaN (im2uint8 li converte in 0).
% Passiamo invece l'immagine double con i NaN direttamente a graycomatrix, 
% forzando i limiti nativi [0 255] affinché i livelli (numLevels=128) siano costanti.

glcm = graycomatrix(grayROI,...
    'Offset',offsets,...
    'NumLevels',numLevels,...
    'Symmetric',symmetric,...
    'GrayLimits', [0 255]); 

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

features = struct();

names = {'Energy','Contrast','Correlation','Homogeneity','Entropy'};

if sum(roiMask(:))==0
    for i=1:length(names)
        features.(names{i}) = NaN;
    end
    return
end

%% ROI malattia
% È fondamentale convertire in double per poter iniettare i NaN
roi = double(grayImage);

% I pixel non infetti vengono ignorati matematicamente
roi(~roiMask) = NaN; 

%% Bounding box automatica solo per ridurre la dimensione della matrice
stats = regionprops(roiMask,'BoundingBox');
bbox = stats(1).BoundingBox;

roi = imcrop(roi,bbox);

%% GLCM
features = extractGLCMFeatures(...
    roi,...
    offsets,...
    numLevels,...
    symmetric);

end

function normalized = normalizeRobust(values)

low = prctile(values,5);
high = prctile(values,95);

normalized = ...
    (values - low) ./ ...
    (high - low + eps);

normalized = min(max(normalized,0),1);

end