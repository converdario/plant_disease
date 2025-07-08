folder = "D:\Vari\Uni\Magistrale\Hands-On\Image Processing\plant_disease\public\dataset\Olive_leaf\train\aculus_olearius"; 

imageFiles = dir(fullfile(folder, '*.jpg'));
imageFiles = [imageFiles; dir(fullfile(folder, '*.JPG'))];  % se vuoi aggiungere altri formati

% Leggi tutte le immagini in un cell array
images = cell(1, length(imageFiles));

for k = 1:length(imageFiles)
    filename = fullfile(folder, imageFiles(k).name);
    images{k} = imread(filename);
end

for k = 1:length(images)
    a = images{k};

    if size(a, 3) == 3
        a = rgb2gray(a);
    end

    disp(['Immagine ', num2str(k)]);

    [L, Centers] = imsegkmeans(a, 3);
    B = labeloverlay(a, L);

    figure;
    imshow(B);
    waitforbuttonpress;
    close all;
    title(['Labeled image ', num2str(k)]);
end