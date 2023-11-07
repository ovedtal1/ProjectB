% SENSE exercise
%%
clear all;
clc;
%%
% Create synthetic coil sensitivity profiles for a two-coil system

% Define image size
image_size = [512, 512];

% Create a coordinate grid
[x, y] = meshgrid(1:image_size(2), 1:image_size(1));

% Define the centers and widths of the sensitivity profiles for both coils
center_x_coil1 = image_size(2) / 2;
center_y_coil1 = image_size(1) / 4;
sigma_x_coil1 = image_size(2) / 6;
sigma_y_coil1 = image_size(1) / 6;

center_x_coil2 = 2 * image_size(2) / 4;
center_y_coil2 = image_size(1) / 1.5;
sigma_x_coil2 = image_size(2) / 6;
sigma_y_coil2 = image_size(1) / 6;

% Generate 2D Gaussian sensitivity profiles for both coils
coil_sensitivity1 = exp(-((x - center_x_coil1).^2 / (2 * sigma_x_coil1^2) + (y - center_y_coil1).^2 / (2 * sigma_y_coil1^2)));
coil_sensitivity2 = exp(-((x - center_x_coil2).^2 / (2 * sigma_x_coil2^2) + (y - center_y_coil2).^2 / (2 * sigma_y_coil2^2)));

% Normalize the sensitivity profiles
coil_sensitivity1 = coil_sensitivity1 / max(coil_sensitivity1(:));
coil_sensitivity2 = coil_sensitivity2 / max(coil_sensitivity2(:));

% Display the synthetic coil sensitivity maps for both coils
figure(1);
subplot(1, 2, 1);
imshow(coil_sensitivity1, []);
title('Coil 1 Sensitivity');

subplot(1, 2, 2);
imshow(coil_sensitivity2, []);
title('Coil 2 Sensitivity');

%%
% Load the k-space data and coil sensitivity maps (you should replace this with your actual data)
load('brain.mat'); % Replace with your own k-space data
%load('coil_sensitivity.mat'); % Replace with your own coil sensitivity maps
coil_sensitivities = zeros(512,512,2);
coil_sensitivities(:,:,1)= coil_sensitivity1;
coil_sensitivities(:,:,2)= coil_sensitivity2;
undersampled_kspace_data = zeros(512,512,2);
kspace_sense1 = fft2c(im.*(coil_sensitivity1));
kspace_sense1(1:2:end,:) = 0;
kspace_sense2 = fft2c(im.*(coil_sensitivity2));
kspace_sense2(1:2:end,:) = 0;
undersampled_kspace_data(:,:,1)= kspace_sense1;
undersampled_kspace_data(:,:,2)= kspace_sense2;

%% SENSE application

in(:,:,1,1) = kspace_sense1;
in(:,:,1,2) = kspace_sense2;
sen(:,:,1,1) = coil_sensitivity1;
sen(:,:,1,2) = coil_sensitivity2;
SENSE_out = SENSE_func(ifft2c(in),sen,2);

figure(2);
subplot(1,2,1);
imagesc(abs(im.*coil_sensitivity1));
colormap('gray');
title('Coil 1');
subplot(1,2,2);
imagesc(abs(im.*coil_sensitivity2));
title('Coil 2');

figure(3);
subplot(1,2,1);
imagesc(abs(ifft2c(kspace_sense1)));
colormap('gray');
title('Coil 1 after down sampling');
subplot(1,2,2);
imagesc(abs(ifft2c(kspace_sense2)));
title('Coil 2 after down sampling');

figure(4);
subplot(1,2,1);
imagesc(abs(SENSE_out));
colormap('gray');
title('SENSE reconstruction');
subplot(1,2,2);
imagesc(abs(im));
title('Original image');

function out = SENSE_func(input,sens,R)
    [Nx,Ny,Nz,Nc] = size(input);
    out = zeros(Nx,Ny);
    % loop over the top-1/R of the image
    for x = 1:Nx/R
        x_idx = x:Nx/R:Nx;
        % loop over the entire left-right extent
        for y = 1:Ny
            % pick out the sub-problem sensitivities
            S = transpose(reshape(sens(x_idx,y,1,:),R,[]));
            % solve the sub-problem in the least-squares sense
            out(x_idx,y) = pinv(S)*reshape(input(x,y,1,:),[],1);
        end
    end
end