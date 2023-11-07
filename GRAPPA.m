% GRAPPA simulation
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
sigma_x_coil1 = image_size(2) / 4;
sigma_y_coil1 = image_size(1) / 4;

center_x_coil2 = 2 * image_size(2) / 4;
center_y_coil2 = image_size(1) / 1.5;
sigma_x_coil2 = image_size(2) / 4;
sigma_y_coil2 = image_size(1) / 4;

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

%% Down sampling coils's sense
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

%% Build GRAPPA kernel - 3xn
n = 3;
kspace1_full_sampled = fft2c(im.*(coil_sensitivity1));
kspace2_full_sampled = fft2c(im.*(coil_sensitivity2));

start_Idx = 240;
end_Idx = 260;

kNo = 1; %kernel number
for ky=start_Idx:end_Idx
    for kx=1:512-n+1
        S_coil1_temp(kNo,:) = reshape(kspace1_full_sampled(ky:ky+2,kx:kx+n-1)',[1,3*n]); 
        S_coil2_temp(kNo,:) = reshape(kspace2_full_sampled(ky:ky+2,kx:kx+n-1)',[1,3*n]); 
        kNo = kNo + 1;
    end
end

% Erasing the unwanted rows
S_coil1 = S_coil1_temp(:,[1:n,2*n+1:3*n]);
S_coil2 = S_coil2_temp(:,[1:n,2*n+1:3*n]);

% Get the right target points matrix
T_coil1 = S_coil1_temp(:,round((3*n)/2));
T_coil2 = S_coil2_temp(:,round((3*n)/2));

% Get the total matrix
S = [S_coil1 S_coil2];
T = [T_coil1 T_coil2];

Weights_mat = pinv(S)*T;

% Get missing values
kNo=1;
for ky=1:2:512-2
    for kx=1:512-n+1
        coil1_S_temp(kNo,:) = reshape(kspace_sense1(ky:ky+2,kx:kx+n-1)',[1,3*n]);
        coil2_S_temp(kNo,:) = reshape(kspace_sense2(ky:ky+2,kx:kx+n-1)',[1,3*n]); 
        kNo = kNo + 1;
    end
end

% get S for all Kx,Ky
coil1_S = coil1_S_temp(:,[1:n,2*n+1:3*n]);
coil2_S = coil2_S_temp(:,[1:n,2*n+1:3*n]);

S_all = [coil1_S coil2_S]; 

T_all = S_all*Weights_mat;
%%

T_coil1 = (reshape(T_all(:,1),[512-n+1,255]))';
T_coil2 = (reshape(T_all(:,2),[512-n+1,255]))';

% Fill the new Kx,Ky
coil1_recon = kspace_sense1;
coil2_recon = kspace_sense2;

%%
coil1_extra_samples = kspace_sense1;
coil1_extra_samples(start_Idx:end_Idx,:) = kspace1_full_sampled(start_Idx:end_Idx,:);
coil2_extra_samples = kspace_sense2;
coil2_extra_samples(start_Idx:end_Idx,:) = kspace2_full_sampled(start_Idx:end_Idx,:);

% coil1_recon(1:2:end-2,2:end-n+2) =  T_coil1;
% coil2_recon(1:2:end-2,2:end-n+2) =  T_coil2;

coil1_recon(start_Idx:end_Idx,:) =  kspace1_full_sampled(start_Idx:end_Idx,:);
coil2_recon(start_Idx:end_Idx,:) =  kspace2_full_sampled(start_Idx:end_Idx,:);

im_coil1_recon = ifft2c(coil1_recon);
im_coil2_recon = ifft2c(coil2_recon);

figure(2);
subplot(1,3,1);
imagesc(abs(im));
colormap('gray');
title('Original image');
% subplot(2,2,2);
% imagesc(abs(ifft2c(coil1_extra_samples))+abs(ifft2c(coil2_extra_samples)));
% colormap('gray');
% title('Non GRAPPGA interpolation image');
subplot(1,3,2);
imagesc(abs(im_coil2_recon)+abs(im_coil1_recon));
colormap('gray');
title('GRAPPA reconstructed image');
subplot(1,3,3);
imagesc(abs(ifft2c(kspace_sense1))+abs(ifft2c(kspace_sense2)));
colormap('gray');
title('Downsampled image');