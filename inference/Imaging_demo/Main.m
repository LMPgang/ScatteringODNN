clear
close all
clc
% Demo1: imaging under static/dynamic scattering medium
% Demo 2: imaging under rotate object
%% Parameters
lmd=780e-9; % wavelength
N=256;  % number of pixel

pixel_SLM=12.5e-6;
d1=pixel_SLM*2;    % object plane pixel size
d2=pixel_SLM*2;    % scattering medium plane pixel size
d3=pixel_SLM*2;    % layer1 pixel size
d4=pixel_SLM*2;    % layer2 pixel size
d5=pixel_SLM*2;    % detection plane pixel size

L1=d1*N;         
L2=d2*N;         
L3=d3*N;         
L4=d4*N;         
L5=d5*N;

% Distance between adjacent layers
z0=0.1;  % object==>scattering medium  
z1=0.2;  % scattering medium==>layer1
z2=0.2;  % layer1==> layer2
z3=0.2;  % layer2==> detection plane

% load object
obj=double(imread('object\0.png'));

%% Demo1: imaging under static/dynamic scattering medium

% load ODNN
M=8; % Can change the bit depth of the ODNN.
interval=2*pi/2^M;

load('ODNN\upright\99_1.mat');  angle1=double(data); angle1=floor(angle1/interval)/2^M*2*pi; SLM_plane1=exp(1i*angle1); 
load('ODNN\upright\99_2.mat');  angle2=double(data); angle2=floor(angle2/interval)/2^M*2*pi; SLM_plane2=exp(1i*angle2);

% Forward propagation
m=10; %Number of scattering media;  m = 1 indicates a static scattering medium.

% with ODNN
I_sum=0;
for k=1:m 
    filename = ['diffuser\diffuser' num2str(k) '.png'];
    ang=double(imread(filename));
    phase_diff=ang/255*2*pi;
    diffuser=exp(1i*phase_diff);

    [~, ~, U1] = ang_spec_prop(obj, lmd, d1, d2, z0);
    U2=U1.*diffuser;
    [~, ~, U3] = ang_spec_prop(U2, lmd, d2, d3, z1);
    U4=U3.*SLM_plane1;                            %1
    [~, ~, U5] = ang_spec_prop(U4, lmd, d3, d4, z2);
    U6=U5.*SLM_plane2;                            %2
    [~, ~, U7] = ang_spec_prop(U6, lmd, d4, d5, z3);
    I7=U7.*conj(U7);
    I_sum=I_sum+I7;
end

% without ODNN
I_without_sum=0;
for k=1:m 
    filename = ['diffuser\diffuser' num2str(k) '.png'];
    ang=double(imread(filename));
    phase_diff=ang/255*2*pi;
    diffuser=exp(1i*phase_diff);

    [~, ~, UU1] = ang_spec_prop(obj, lmd, d1, d2, z0);
    UU2=UU1.*diffuser;
    [~, ~, UU3] = ang_spec_prop(UU2, lmd, d2, d3, z1);
    UU4=UU3.*ones(N);                            %1
    [~, ~, UU5] = ang_spec_prop(UU4, lmd, d3, d4, z2);
    UU6=UU5.*ones(N);                            %2
    [~, ~, UU7] = ang_spec_prop(UU6, lmd, d4, d5, z3);
    II7=UU7.*conj(UU7);
    I_without_sum=I_without_sum+II7;
end

figure  
subplot(1,2,1); imagesc(angle1);axis square;axis off;colormap('gray');colorbar();title('layer1')
subplot(1,2,2); imagesc(angle2);axis square;axis off;colormap('gray');colorbar();title('layer2')

figure 
subplot(2,2,1);imagesc(obj);axis square;axis off;colormap('gray');title('Ground truth')
subplot(2,2,2);imagesc(I_sum);axis square;axis off;colormap('gray');title('Result with ODNN')
subplot(2,2,3);imagesc(I_without_sum);axis square;axis off;colormap('gray');title('Result without ODNN')


% Compute the Pearson correlation coefficient between obj and ref.
ref=I_sum;
cov_obj_ref=sum(sum( (obj-mean(mean(obj))).*(ref-mean(mean(ref))) )); 
var_obj=sqrt(sum(sum((obj-mean(mean(obj))).^2))); 
var_ref=sqrt(sum(sum((ref-mean(mean(ref))).^2))); 
PCC=cov_obj_ref / (var_obj*var_ref);

disp('----------------------------------------------');
disp('Demo1:');
fprintf('PCC between obj and ref: PCC = %.2f\n',PCC); 
disp('----------------------------------------------');
%% Demo 2: imaging under rotate object

% load ODNN
M=8; % Can change the bit depth of the ODNN.
interval=2*pi/2^M;

load('ODNN\rotate90\99_1.mat');  angle1=double(data); angle1=floor(angle1/interval)/2^M*2*pi; SLM_plane1=exp(1i*angle1); 
load('ODNN\rotate90\99_2.mat');  angle2=double(data); angle2=floor(angle2/interval)/2^M*2*pi; SLM_plane2=exp(1i*angle2);

% Forward propagation
m=10; %Number of scattering media; m = 1 indicates a static scattering medium.

% with ODNN
I_sum=0;
for k=1:m 
    filename = ['diffuser\diffuser' num2str(k) '.png'];
    ang=double(imread(filename));
    phase_diff=ang/255*2*pi;
    diffuser=exp(1i*phase_diff);

    [~, ~, U1] = ang_spec_prop(obj, lmd, d1, d2, z0);
    U2=U1.*diffuser;
    [~, ~, U3] = ang_spec_prop(U2, lmd, d2, d3, z1);
    U4=U3.*SLM_plane1;                            %1
    [~, ~, U5] = ang_spec_prop(U4, lmd, d3, d4, z2);
    U6=U5.*SLM_plane2;                            %2
    [~, ~, U7] = ang_spec_prop(U6, lmd, d4, d5, z3);
    I7=U7.*conj(U7);
    I_sum=I_sum+I7;
end


% without ODNN
I_without_sum=0;
for k=1:m 
    filename = ['diffuser\diffuser' num2str(k) '.png'];
    ang=double(imread(filename));
    phase_diff=ang/255*2*pi;
    diffuser=exp(1i*phase_diff);

    [~, ~, UU1] = ang_spec_prop(obj, lmd, d1, d2, z0);
    UU2=UU1.*diffuser;
    [~, ~, UU3] = ang_spec_prop(UU2, lmd, d2, d3, z1);
    UU4=UU3.*ones(N);                            %1
    [~, ~, UU5] = ang_spec_prop(UU4, lmd, d3, d4, z2);
    UU6=UU5.*ones(N);                            %2
    [~, ~, UU7] = ang_spec_prop(UU6, lmd, d4, d5, z3);
    II7=UU7.*conj(UU7);
    I_without_sum=I_without_sum+II7;
end

figure  
subplot(1,2,1); imagesc(angle1);axis square;axis off;colormap('gray');colorbar();title('layer1')
subplot(1,2,2); imagesc(angle2);axis square;axis off;colormap('gray');colorbar();title('layer2')

figure 
subplot(2,2,1);imagesc(obj);axis square;axis off;colormap('gray');title('Ground truth')
subplot(2,2,2);imagesc(I_sum);axis square;axis off;colormap('gray');title('Result with ODNN')
subplot(2,2,3);imagesc(I_without_sum);axis square;axis off;colormap('gray');title('Result without ODNN')


% Compute the Pearson correlation coefficient between obj and ref.
obj_90=imrotate(obj,90);
obj=obj_90;
ref=I_sum;
cov_obj_ref=sum(sum( (obj-mean(mean(obj))).*(ref-mean(mean(ref))) )); 
var_obj=sqrt(sum(sum((obj-mean(mean(obj))).^2))); 
var_ref=sqrt(sum(sum((ref-mean(mean(ref))).^2))); 
PCC=cov_obj_ref / (var_obj*var_ref);


disp('Demo2:');
fprintf('PCC between obj and ref: PCC = %.2f\n',PCC); 
disp('----------------------------------------------');



