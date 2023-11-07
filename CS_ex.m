% Compressed sensing exercise
%%
clear all;
clc;

%% Taks 1 - a
x = [[1:5]/5 zeros(1,128-5)];
x = x(randperm(128));

std = 0.05;
y = x + std*randn(1,128);

lambda_vec = [0.01 0.05 0.1 0.2];
x_est = zeros(size(lambda_vec,2),size(x,2));
%% - ploting
for i=1:4
    x_est(i,:) = y*(1/(1+lambda_vec(i)));
    figure(1);
    subplot(4,1,i);
    plot(x_est(i,:));
    xlabel('Sample');
end

%% Taks 1 - b
y_axis = linspace(-10,10,1000);
lambda = 2;
x_est_func = zeros(1,size(y_axis,2));
x_est_func(y_axis<-lambda) = y_axis(y_axis<-lambda) + lambda;
x_est_func(y_axis>lambda) = y_axis(y_axis>lambda) - lambda;

figure(2);
plot(y_axis,x_est_func);
title('SoftThresh function, \lambda = 2','FontSize',14);
xlabel('y','FontSize',12);
ylabel('x','FontSize',12);

% SoftThresh to noisy signal
x_est_b = zeros(size(lambda_vec,2),size(x,2));
for i=1:4
    lambda = lambda_vec(i);
    x_est_b(i,:) = SoftThresh(y,lambda);
end

figure(3);
plot(x_est_b(3,:));
title('SoftThresh to noisy signal, \lambda = 0.1','FontSize',14);
ylabel('Estimated x','FontSize',12);

%% Taks 1 - c
%in order 1:4
X = fftc(x,2);
Xu = zeros(128);
Xu(1:4:128) = X(1:4:128);
xu = ifftc(Xu,2)*4;

figure(4);
plot(abs(xu));
title('UnderSampled in FD x, in order sampling ','FontSize',14);
ylabel('x','FontSize',12);

%random sampling 32 samples
Xr = zeros(1,128);
prm = randperm(128);
Xr(prm(1:32)) = X(prm(1:32));
xr = ifftc(Xr,2)*4;
Xr_saved = Xr;

figure(5);
plot(abs(xr));
title('UnderSampled in FD x, random sampling ','FontSize',14);
ylabel('x','FontSize',12);

%% Taks 1 - d
lambda = [0.01 0.05 0.1];
Y = Xr_saved;
error = zeros(3,300);
for i=1:3
    Xr(i,:) = Y;
    figure(5+i);
    for j=1:300
        xr(i,:) = ifftc(Xr(i,:),2);

        plot(abs(xr(i,:)));
        hold on;
        drawnow;
        title('Random sampling - reconstruction ','FontSize',14);
        ylabel('x','FontSize',12);
    
        x_est = SoftThreshComplex(xr(i,:)*4,lambda(i));
        Xr(i,:) = fftc(x_est/4,2);
        Xr(i,:) = Xr(i,:).*(Y==0) + Y;
        error(i,j) = abs(sum(abs(x))-sum(abs(x_est)));
    end
    hold off;
end

figure(9);
plot(x);
hold on;
plot(abs(xr(1,:)));
plot(abs(xr(2,:)));
plot(abs(xr(3,:)));
legend('original','reconstructed -\lambda = 0.01', 'reconstructed -\lambda = 0.05','reconstructed -\lambda = 0.1');
hold off;
%% error graph
figure(10);
plot(error(1,:));
hold on;
plot(error(2,:));
plot(error(3,:));
xlabel('Iteration');
ylabel('Error');
legend('\lambda = 0.01','\lambda = 0.05','\lambda = 0.1' );
hold off;
%% Part 2

load('brain.mat');
W = Wavelet;
im_W = W*im;
figure(10); subplot(1,2,1); 
imshow(abs(im),[]), title('Original Image');
subplot(1,2,2); 
imshow(abs(im_W),[0,1]), title('Coefficients image');

% Zeroing 80% of the data, to be sparser
f=0.2;
m = sort(abs(im_W(:)),'descend');
ndx = floor(length(m)*f);
thresh = m(ndx);
im_W_th = im_W .* (abs(im_W) > thresh);
im_th_20 = W'*im_W_th;
saved_thresh = im_W_th;

figure(11);
imshow(abs(im_W_th),[0,1]), title(' Threshholded Coefficients image');

% Reconstruction 
im_rec = W'*im_W_th;
figure(12);
imshow(abs(im_rec),[]), title('Reconstructed image with thresh 20%');

% Difference plot
figure(13);
imshow(abs(abs(im)-abs(im_rec)),[]), title('Original subtructed by Reconstructed image, with thresh 20%');

% Different f 
f_vec = [0.125 0.1 0.05 0.025];
figure(14)
for i=1:4
    m = sort(abs(im_W(:)),'descend');
    ndx = floor(length(m)*f_vec(i));
    thresh = m(ndx);
    im_W_th = im_W .* (abs(im_W) > thresh);
    im_rec = W'*im_W_th;
    subplot(1,4,i); 
    imshow(abs(im_rec),[]), title(['Reconstructed image with thresh:' num2str(f_vec(i))]);

end

%% add on after conversatin with Efrat -a
sort_thresh = sort(abs(saved_thresh(:)),'descend');
figure(15);
plot(m);
hold on;
plot(sort_thresh);
hold off;
title('In order coeffsm only few are significant');

%% Task 3 -a
load('brain.mat');
M = fft2c(im);
Mu = (M.*mask_unif)./pdf_unif;
imu = ifft2c(Mu);

Mv = (M.*mask_vardens)./pdf_vardens;
imv = ifft2c(Mv);

figure(15);
subplot(2,2,1);
imshow(abs(abs(imu)),[]);
title('Reconstruction using unif-mask');
subplot(2,2,2);
imshow(abs(abs(im)-abs(imu)),[]);
title('Original substructed by reconstruction using unif-mask');
subplot(2,2,3);
imshow(abs(abs(imv)),[]);
title('Reconstruction using vardens-mask');
subplot(2,2,4);
imshow(abs(abs(im)-abs(imv)),[]);
title('Original substructed by reconstruction using vardens-mask');

%% add on after conversatin with Efrat -b
% Mask showing
figure(16);
subplot(2,1,1);
imshow(mask_unif,[]);
title('Uniform mask');
subplot(2,1,2);
imshow(mask_vardens,[]);
title('Vardens mask');

% in FD (where we sample)
figure(17);
title('FFT on image wavelet transform');
imshow(abs(M),[]);

%% Task 3 - b - unif
lambda = 0.2;

figure(16);
subplot(2,2,1);
imshow(abs(im),[]);
title('Original image');

figure(16);
subplot(2,2,2);
imshow(abs(im_th_20),[]);
title('Original image - softhreshold - 20%');

figure(16);
subplot(2,2,3);
imshow(abs(W'*ifft2c((fft2c(W*im).*mask_unif)./pdf_unif)),[]);
title('Zero field reconstructed image');

imu_W_fft = (fft2c(W*im).*mask_unif)./pdf_unif;
Y = imu_W_fft;
figure(16);
for j=1:40
    if mod(j,10)==0
        lambda = lambda /3;
    end
    imu_W = ifft2c(imu_W_fft);
    subplot(2,2,4);
    imshow(abs(W'*imu_W),[]);
    drawnow;
    title('Uniform PDF sampling - reconstruction ');

    imu_est = SoftThreshComplex(imu_W,lambda);
    imu_W_fft = fft2c(imu_est);
    imu_W_fft = imu_W_fft.*(Y==0) + Y;
end

%% add on after conversatin with Efrat -c
% FD of wavelet before and after iterations
figure(16);
subplot(1,3,1);
imshow(abs(fft2c(W*im)),[]);
title('No sampling');
subplot(1,3,2);
imshow(abs(Y),[]);
title('Before Iterations');
subplot(1,3,3)
imshow(abs(imu_W_fft),[]);

title('After Iterations');

% FD of picture before and after iterations
figure(17);
subplot(1,3,1);
imshow(20*log10(abs(fft2c(im))),[]);
title('Original FFT');
subplot(1,3,2);
imshow(20*log10(abs(fft2c(W'*(fft2c(Y))))),[]);
title('Before Iterations');
subplot(1,3,3)
imshow(20*log10(abs(fft2c(W'*imu_W))),[]);
title('After Iterations');

%% Task 3 - b - vardens
lambda = 0.005;

figure(17);
subplot(1,3,1);
imshow(abs(im),[]);
title('Original image');

figure(17);
subplot(1,3,2);
imshow(abs(W'*ifft2c((fft2c(W*im).*mask_vardens)./pdf_vardens)),[]);
title('Zero field reconstructed image');

imv_W_fft = (fft2c(W*im).*mask_vardens)./pdf_vardens;
Y = imv_W_fft;
figure(17);
for j=1:20
%     if mod(j,10)==0
%         lambda = lambda /3;
%     end
    imv_W = ifft2c(imv_W_fft);
    subplot(1,3,3);
    imshow(abs(W'*imv_W),[]);
    drawnow;
    title('Vardens PDF sampling - reconstruction ','FontSize',14);

    imv_est = SoftThreshComplex(imv_W,lambda);
    imv_W_fft = fft2c(imv_est);
    imv_W_fft = imv_W_fft.*(Y==0) + Y;
end

%%
lambda = 0.20;
imuW = W*imv;
figure(20);
imshow(abs(imuW)>lambda,[]);
%imshow(abs(W*im),[0,1]);
