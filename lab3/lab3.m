%%  L A B    3
clear;close all;clc;
addpath('./MusicFileSamples'),addpath('EmotionLabellingData'),addpath('MIRtoolbox1.6.2');
% %% STEP 1
% % for i = 1:412
% %     filename = sprintf('file%d.wav', i); % find audio file
% %     signal = audioread(filename);
% %     signal_mono = (signal(:,1)+signal(:,2))/2;        % make mono
% %     filename = sprintf('mono_files/file%d.wav', i);
% %     audiowrite(filename,signal_mono, 22050,'BitsPerSample', 8);
% % end
% %% STEP 2
% %%% Labeler1
% labeler1 = load('EmotionLabellingData/Labeler1.mat');
% 
% valence_vec1 = [];
% activation_vec1 = [];
% for i = 1:412       % for all music files
%     a = labeler1.labelList(i).valence;      % find valence and activation
%     valence_vec1 = [valence_vec1 a];        % put them in a vector
%     b = labeler1.labelList(i).activation;   
%     activation_vec1 = [activation_vec1 b];
% end
% mean_val_labeler_1 = mean(valence_vec1);    % calculate mean value
% var_val_labeler_1 = var(valence_vec1);      % calculate variance
% mean_act_labeler_1 = mean(activation_vec1);
% var_act_labeler_1 = var(activation_vec1);
% 
% occ_mat_1 = zeros(5);       % form co-occurence matrix 5x5 with valence and activation values
% for i = 1:412
%     x = valence_vec1(i);
%     y = activation_vec1(i);
%     occ_mat_1(x,y) = occ_mat_1(x,y) + 1;
% end
% 
% %%% Labeler2
% labeler2 = load('EmotionLabellingData/Labeler2.mat');
% 
% valence_vec2 = [];
% activation_vec2 = [];
% for i = 1:412
%     a = labeler2.labelList(i).valence;
%     valence_vec2 = [valence_vec2 a];
%     b = labeler2.labelList(i).activation;
%     activation_vec2 = [activation_vec2 b];
% end
% 
% mean_val_labeler_2 = mean(valence_vec2);
% var_val_labeler_2 = var(valence_vec2);
% mean_act_labeler_2 = mean(activation_vec2);
% var_act_labeler_2 = var(activation_vec2);
% 
% occ_mat_2 = zeros(5);
% for i = 1:412
%     x = valence_vec2(i);
%     y = activation_vec2(i);
%     occ_mat_2(x,y) = occ_mat_2(x,y) + 1;
% end
% 
% %%% Labeler3
% labeler3 = load('EmotionLabellingData/Labeler3.mat');
% 
% valence_vec3 = [];
% activation_vec3 = [];
% for i = 1:412
%     a = labeler3.labelList(i).valence;
%     valence_vec3 = [valence_vec3 a];
%     b = labeler3.labelList(i).activation;
%     activation_vec3 = [activation_vec3 b];
% end
% 
% mean_val_labeler_3 = mean(valence_vec3);
% var_val_labeler_3 = var(valence_vec3);
% mean_act_labeler_3 = mean(activation_vec3);
% var_act_labeler_3 = var(activation_vec3);
% 
% occ_mat_3 = zeros(5);
% for i = 1:412
%     x = valence_vec3(i);
%     y = activation_vec3(i);
%     occ_mat_3(x,y) = occ_mat_3(x,y) + 1;
% end
% % plot 2D co-occ matrices
% % figure,fig1 = imagesc(occ_mat_1); colormap(copper); title('Co-occurence matrix for Labeler1'); xlabel('Valence'); ylabel('Activation'); saveas(fig1, 'co_occ_mat_labeler1.png');
% % figure,fig2 = imagesc(occ_mat_2); colormap(copper); title('Co-occurence matrix for Labeler2'); xlabel('Valence'); ylabel('Activation'); saveas(fig2, 'co_occ_mat_labeler2.png');
% % figure,fig3 = imagesc(occ_mat_3); colormap(copper); title('Co-occurence matrix for Labeler3'); xlabel('Valence'); ylabel('Activation'); saveas(fig3, 'co_occ_mat_labeler3.png');
% %% STEP 3
% % calculate observed agreement and mean agreement for valence and
% % activation, for all pairs of labelers
% 
% %%% Labelers 1 and 2
% lab12_diffvec_val = [];
% lab12_diffvec_act = [];
% for i = 1:412
%     tmp1 = abs(labeler1.labelList(i).valence - labeler2.labelList(i).valence);
%     tmp2 = abs(labeler1.labelList(i).activation - labeler2.labelList(i).activation);
%     
%     lab12_diffvec_val = [lab12_diffvec_val tmp1];  % vectors with observed differences
%     lab12_diffvec_act = [lab12_diffvec_act tmp2];
% end
% agree_val_lab12 = 1 - mean(lab12_diffvec_val/4);
% agree_act_lab12 = 1 - mean(lab12_diffvec_act/4);
% 
% %%% Labelers 1 and 3
% lab13_diffvec_val = [];
% lab13_diffvec_act = [];
% for i = 1:412
%     tmp1 = abs(labeler1.labelList(i).valence - labeler3.labelList(i).valence);
%     tmp2 = abs(labeler1.labelList(i).activation - labeler3.labelList(i).activation);
%     
%     lab13_diffvec_val = [lab13_diffvec_val tmp1]; % vectors with observed differences
%     lab13_diffvec_act = [lab13_diffvec_act tmp2];
% end
% agree_val_lab13 = 1 - mean(lab13_diffvec_val/4);
% agree_act_lab13 = 1 - mean(lab13_diffvec_act/4);
% 
% %%% Labelers 2 and 3
% lab23_diffvec_val = [];
% lab23_diffvec_act = [];
% for i = 1:412
%     tmp1 = abs(labeler2.labelList(i).valence - labeler3.labelList(i).valence);
%     tmp2 = abs(labeler2.labelList(i).activation - labeler3.labelList(i).activation);
%     
%     lab23_diffvec_val = [lab23_diffvec_val tmp1];     % vectors with observed differences
%     lab23_diffvec_act = [lab23_diffvec_act tmp2];
% end
% agree_val_lab23 = 1 - mean(lab23_diffvec_val/4);   % mean value agreement
% agree_act_lab23 = 1 - mean(lab23_diffvec_act/4);
% 
% % show labelers differences histogram
% lab12_hist_diff = zeros(5);
% lab13_hist_diff = zeros(5);
% lab23_hist_diff = zeros(5);
% for i = 1:412
%     x1 = lab12_diffvec_val(i);
%     y1 = lab12_diffvec_act(i);
%     
%     x2 = lab13_diffvec_val(i);
%     y2 = lab13_diffvec_act(i);
%     
%     x3 = lab23_diffvec_val(i);
%     y3 = lab23_diffvec_act(i);
%     
%     lab12_hist_diff(x1+1,y1+1) = lab12_hist_diff(x1+1,y1+1) + 1;   
%     lab13_hist_diff(x2+1,y2+1) = lab13_hist_diff(x2+1,y2+1) + 1;
%     lab23_hist_diff(x3+1,y3+1) = lab23_hist_diff(x3+1,y3+1) + 1;
% end
%     
% % figure,fig4 = imagesc(lab12_hist_diff); colormap(copper); xlabel('Valence'); ylabel('Activation'); title('Labelers 1 and 2 Diffrences Histogram'); saveas(fig4, 'diff_hist_lab12.png');
% % figure,fig5 = imagesc(lab13_hist_diff); colormap(copper); xlabel('Valence'); ylabel('Activation'); title('Labelers 1 and 3 Diffrences Histogram'); saveas(fig5, 'diff_hist_lab13.png');
% % figure,fig6 = imagesc(lab23_hist_diff); colormap(copper); xlabel('Valence'); ylabel('Activation'); title('Labelers 2 and 3 Diffrences Histogram'); saveas(fig6, 'diff_hist_lab23.png');
% %% STEP 4
% % calculate Krippendorff alpha for each dimention
% 
% mat_input_val = [valence_vec1; valence_vec2; valence_vec3];
% mat_input_act = [activation_vec1; activation_vec2; activation_vec3];
% 
% alpha_valence = kriAlpha(mat_input_val, 'ordinal');
% alpha_activation = kriAlpha(mat_input_act, 'ordinal');
% %% STEP 5
% final_label_val = [];
% final_label_act = [];
% for i = 1:412
%     tmp = (labeler1.labelList(i).valence + labeler2.labelList(i).valence + labeler3.labelList(i).valence)/3;
%     final_label_val = [final_label_val tmp];
%     
%     tmp = (labeler1.labelList(i).activation + labeler2.labelList(i).activation + labeler3.labelList(i).activation)/3;
%     final_label_act = [final_label_act tmp];
% end
% 
% % 2D co-occurence matrix  
% values = [1 1.3333 1.6667 2 2.3333 2.6667 3 3.3333 3.6667 4 4.3333 4.6667 5];
% 
% % ensure values
% for i = 1:412
%     for j = values;
%     if(abs(j-final_label_val(i)) < 1e-2)
%         final_label_val(i) = j;
%     end
%     if(abs(j-final_label_act(i)) < 1e-2)
%         final_label_act(i) = j;
%     end
%     end
% end
% 
% occ_mat_final = zeros(13);
% for i = 1:13
%     k = values(i);
%     for j = 1:13
%         l = values(j);
%         qq = find(final_label_val == k & final_label_act == l);
%         occ_mat_final(i,j) = length(qq);
%     end
% end
% % figure,fig7 = imagesc(occ_mat_final); colormap(copper); title('Co-occurence matrix for final Labels'); xlabel('Valence'); ylabel('Activation'); saveas(fig7, 'co_occ_mat_labels_final.png');
% 
% %% STEP 6
% Nc=13;fs=22050;Q=26;K=512;T=.025*fs; Toverlap=.01*fs;
% % MIRtoolbox
% addpath(genpath('MIRtoolbox1.6.2'));
% % read with miraudio and decompose to frames
% for i = 1:412
%     filename = sprintf('mono_files/file%d.wav', i); % find audio file
%     mir_signals{i} = miraudio(filename, 'Frame');
%     % statistics
%     sigs_mean{i} = mirgetdata( mirmean(mir_signals{i})); 
%     sigs_std{i} = mirgetdata(mirstd(mir_signals{i}));
%     sigs_median{i} = mirgetdata(mirmedian(mir_signals{i}));
%     % Auditory roughness
%     aud_rough{i} = mirgetdata(mirroughness(mir_signals{i}));
%     aud_rough_mean{i}=mean(aud_rough{i});
%     aud_rough_var{i}=var(aud_rough{i});
%     med=median(aud_rough{i});
%     aud_r=aud_rough{i};
%     aud_rough_mean_1{i}=mean(aud_r(aud_r>med));
%     aud_rough_mean_2{i}=mean(aud_r(aud_r<med));
%     % Rythmic Periodicity Along Auditory Channels
%     fluct{i} = mirgetdata(mirfluctuation(mir_signals{i}));
%     fluct_max{i} = max(fluct{i});
%     fluct_mean{i} = mean(fluct{i});
%     % Key Clarity
%     k_clar{i}= mirgetdata(mirkey(mir_signals{i}));
%     k_clar_mean{i}=mean(k_clar{i});
%     % Modality
%     modality{i} = mirgetdata(mirmode(mir_signals{i}));
%     modality_mean{i}=mean(modality{i});
%     % Spectral Novelty 
%     spec_novel{i} = mirgetdata(mirnovelty(mir_signals{i}));
%     spec_novel_mean{i}=mean(spec_novel{i});
%     % Harmonic Change Detection Function (HCDF)
%     hcdf{i} = mirgetdata(mirhcdf(mir_signals{i}'));
%     hcdf_mean{i}=mean(hcdf{i});
%     %% STEP 7
%     % MFCC known methodology
%     sP=filter([1 -0.97],1,audioread(filename));
%     frames=buffer(sP,ceil(T),ceil(Toverlap));
%     [~,windows]=size(frames);
%     sI=[];
%     for k=1:windows
%         frame=frames(:,k).*hamming(ceil(T));
%         sI=[sI,frame];
%     end
%     R=[0 fs/2]; 
%     c = M2H(H2M(R(1))+[0:Q+1]*((H2M(R(2))-H2M(R(1)))/(Q+1)));
%     f = linspace(R(1), R(2), K); 
%     H = zeros(Q,K);                    
%     for j = 1:Q 
%         k = f>=c(j)&f<=c(j+1);       
%         H(j,k) = 2*(f(k)-c(j)) / ((c(j+2)-c(j))*(c(j+1)-c(j)));
%         k = f>=c(j+1)&f<=c(j+2);        
%         H(j,k) = 2*(c(j+2)-f(k)) / ((c(j+2)-c(j))*(c(j+2)-c(j+1)));
%     end
%     H = H./repmat(max(H,[],2),1,K); 
%     for w=1:windows
%        SI=abs(fft(sI(:,w),1024));
%        SI=SI(1:512);
%         for j=1:Q
%            X=SI.*(H(j,:))';     
%            x=abs(ifft(X,512));
%            E(w,j)=2*sum(x.^2);
%         end
%     end
%     G=log10(E); 
%     for win=1:windows
%         for n=1:Nc 
%             j=1:Q;
%             tmp(j)=G(win,j).*cos((n-1)*(j-1/2)*pi/Q);% we take n-1 becausin MATLAB indexes begin from 1
%             coefs(win,n)=sum(tmp);
%         end
%         % make the 39 characteristics by adding the 1st & 2nd gradient
%         tmp=coefs(win,:);
%         delta=gradient(tmp);
%         deltadelta=gradient(delta);
%         MFCCS_with_DELTAS{i}(win,:)=[coefs(win,:) ,delta ,deltadelta];
%     end
%     mfccs_dd_mean{i}(:)=mean(MFCCS_with_DELTAS{i}(:,:));
%     mfccs_dd_var{i}(:)=var(MFCCS_with_DELTAS{i}(:,:));
%     % find 10 PER CENT max & min
%     % first sort, then take the 10% first & last
%     range=ceil(.1*windows);
%     tmpp=sort(MFCCS_with_DELTAS{i}(:,:));
%     max_10_PC=tmpp(windows-range+1:windows,:);
%     min_10_PC=tmpp(1:range,:);
%     mfccs_dd_mean_1{i}(:)=mean(max_10_PC);
%     mfccs_dd_mean_2{i}(:)=mean(min_10_PC);
% end


%% step 10

% save('act.mat','final_label_act');
% save('val.mat','final_label_val');
% 
% workspace_prelab3 = matfile('results_all.mat') ;
% 
% final_label_act = workspace_prelab3.final_label_act;
% final_label_val = workspace_prelab3.final_label_val;

final_label_act1 = load('act.mat');
final_label_val1 = load('val.mat');
final_label_act = final_label_act1.final_label_act;
final_label_val = final_label_val1.final_label_val;


% normalise_act = (final_label_act - min(final_label_act)) / ( max(final_label_act) - min(final_label_act) );
% normalise_val = (final_label_val - min(final_label_val)) / ( max(final_label_val) - min(final_label_val) );
% 
% act_thres_01 = im2bw(normalise_act, 0.5);
% val_thres_01 = im2bw(normalise_val, 0.5);
levels = [2.98 3.02];
act_thres_01 = imquantize(final_label_act,levels);
val_thres_01 = imquantize(final_label_val,levels);

act_thres = zeros(1,size(act_thres_01,2));
val_thres = zeros(1,size(val_thres_01,2));

for i=1:412
    if(act_thres_01(i) == 1)
        act_thres(i) = -1;
    elseif(act_thres_01(i) == 3)
        act_thres(i) = +1;
    else        
        act_thres(i) = 0;
    end
    if(val_thres_01(i) == 1)
        val_thres(i) = -1;
    elseif(val_thres_01(i) == 3)
        val_thres(i) = +1;
    else
        val_thres(i) = 0;
    end
end

val_neg = sum(val_thres == -1);
val_pos = sum(val_thres == +1);
act_neg = sum(act_thres == -1);
act_pos = sum(act_thres == +1);

%% step 11

%%mir edit
% mir_all = [aud_rough_mean; aud_rough_var; aud_rough_mean_1; aud_rough_mean_2; k_clar_mean; modality_mean; spec_novel_mean; hcdf_mean];
% save('mir.mat', 'mir_all');
mir_all1 = load('mir.mat');
mir_all = mir_all1.mir_all;
mir_all = cell2mat(mir_all);

%%mfccs edit
% save('mfccs.mat','mfccs_dd_mean');
mfccs_dd_mean1 = load('mfccs.mat');
mfccs_dd_mean = mfccs_dd_mean1.mfccs_dd_mean;
mfccs_all = (cell2mat(mfccs_dd_mean'))';

%mfccs fix NaN and Inf
for i = 1:size(mfccs_all,1)
    for j = 1:size(mfccs_all,2)
        if(isnan(mfccs_all(i,j)) == 1)
            mfccs_all(i,j) = 0.001;
        elseif(isinf(mfccs_all(i,j)) == 1)
            mfccs_all(i,j) = 0.002;
        end
    end
end
% mfccs_all(any(isnan(mfccs_all),1),:) = 0.001;
% mfccs_all(any(isinf(mfccs_all),1),:) = 0.002;


a11 = 1; a12 = round(412/5); a13 = size(mir_all,2);
res_test1_a = act_thres(a11:a12);
res_train1_a = act_thres(a12+1:a13);
res_test1_v = val_thres(a11:a12);
res_train1_v = val_thres(a12+1:a13);
%1st fold - mir
test1_r = mir_all(:,1:round(412/5)) ;
train1_r = mir_all(:, round(412/5)+1 : size(mir_all,2));
%1st fold - mfccs
test1_f = mfccs_all(:,1:round(412/5)) ;
train1_f = mfccs_all(:, round(412/5)+1 : size(mfccs_all,2));

a21 = round(412/5) +1; a22=2*round(412/5); a23=1; a24=size(mir_all,2);
res_test2_a = act_thres(a21:a22);
res_train2_a1 = act_thres(a23:a21-1);
res_train2_a2 = act_thres(a22+1:a24);
res_train2_a = [res_train2_a1 res_train2_a2];
res_test2_v = val_thres(a21:a22);
res_train2_v1 = val_thres(a23:a21-1);
res_train2_v2 = val_thres(a22+1:a24);
res_train2_v = [res_train2_v1 res_train2_v2];
%2nd fold - mir
test2_r = mir_all(:,round(412/5)+1 : 2*round(412/5)) ;
train2_temp1_r = mir_all(:, 1: round(412/5));
train2_temp2_r = mir_all(:, 2*round(412/5) + 1 : size(mir_all,2));
train2_r = [train2_temp1_r train2_temp2_r];
%2nd fold - mfccs
test2_f = mfccs_all(:,round(412/5)+1 : 2*round(412/5)) ;
train2_temp1_f = mfccs_all(:, 1: round(412/5));
train2_temp2_f = mfccs_all(:, 2*round(412/5) + 1 : size(mfccs_all,2));
train2_f = [train2_temp1_f train2_temp2_f];

a31 = 2*round(412/5) +1; a32=3*round(412/5); a33=1; a34=size(mir_all,2);
res_test3_a = act_thres(a31:a32);
res_train3_a1 = act_thres(a33:a31-1);
res_train3_a2 = act_thres(a32+1:a34);
res_train3_a = [res_train3_a1 res_train3_a2];
res_test3_v = val_thres(a31:a32);
res_train3_v1 = val_thres(a33:a31-1);
res_train3_v2 = val_thres(a32+1:a34);
res_train3_v = [res_train3_v1 res_train3_v2];
%3rd fold - mir
test3_r = mir_all(:,2*round(412/5)+1 : 3*round(412/5)) ;
train3_temp1_r = mir_all(:, 1: 2*round(412/5));
train3_temp2_r = mir_all(:, 3*round(412/5) + 1 : size(mir_all,2));
train3_r = [train2_temp1_r train2_temp2_r];
%3rd fold - mfccs
test3_f = mfccs_all(:,2*round(412/5)+1 : 3*round(412/5)) ;
train3_temp1_f = mfccs_all(:, 1: 2*round(412/5));
train3_temp2_f = mfccs_all(:, 3*round(412/5) + 1 : size(mfccs_all,2));
train3_f = [train2_temp1_f train2_temp2_f];

% 3 folds : mir + mfccs
test1_rf = [test1_r; test1_f];
test2_rf = [test2_r; test2_f];
test3_rf = [test3_r; test3_f];
train1_rf = [ train1_r; train1_f];
train2_rf = [ train2_r; train2_f];
train3_rf = [ train3_r; train3_f];



%merge
train111 = [res_train1_a' train1_r'];
train112 = [res_train1_v' train1_r'];
train121 = [res_train1_a' train1_f'];
train122 = [res_train1_v' train1_f'];
train131 = [res_train1_a' train1_rf'];
train132 = [res_train1_v' train1_rf'];

train211 = [res_train2_a' train2_r'];
train212 = [res_train2_v' train2_r'];
train221 = [res_train2_a' train2_f'];
train222 = [res_train2_v' train2_f'];
train231 = [res_train2_a' train2_rf'];
train232 = [res_train2_v' train2_rf'];

train311 = [res_train3_a' train3_r'];
train312 = [res_train3_v' train3_r'];
train321 = [res_train3_a' train3_f'];
train322 = [res_train3_v' train3_f'];
train331 = [res_train3_a' train3_rf'];
train332 = [res_train3_v' train3_rf'];

test111 = [res_test1_a' test1_r'];
test112 = [res_test1_v' test1_r'];
test121 = [res_test1_a' test1_f'];
test122 = [res_test1_v' test1_f'];
test131 = [res_test1_a' test1_rf'];
test132 = [res_test1_v' test1_rf'];

test211 = [res_test2_a' test2_r'];
test212 = [res_test2_v' test2_r'];
test221 = [res_test2_a' test2_f'];
test222 = [res_test2_v' test2_f'];
test231 = [res_test2_a' test2_rf'];
test232 = [res_test2_v' test2_rf'];

test311 = [res_test3_a' test3_r'];
test312 = [res_test3_v' test3_r'];
test321 = [res_test3_a' test3_f'];
test322 = [res_test3_v' test3_f'];
test331 = [res_test3_a' test3_rf'];
test332 = [res_test3_v' test3_rf'];

%% step 12

nums = {111,112,121,122,131,132,211,212,221,222,231,232,311,312,321,322,331,332};
trains = cell(1,size(nums,2));
for i = 1:size(nums,2)
    trains{i} = sprintf('train%d', nums{i});
end
tests = cell(1,size(nums,2));
for i = 1:size(nums,2)
    tests{i} = sprintf('test%d', nums{i});
end

k=3;
for i = 1:size(nums,2)
    train_temp = eval(trains{i});
    test_temp = eval(tests{i});       
    train_temp(any(train_temp == 0,2),:) = [];
    test_temp(any(test_temp == 0,2),:) = [];
    perc_nnrk(i) = nnrk(train_temp, test_temp, k);
end

mean_perc_nnrk = sum(perc_nnrk(perc_nnrk > 50)) / sum(perc_nnrk > 50);
display(mean_perc_nnrk, 'Mean percentage from nnrk = ');

%% step 13

apriori(1,1) = val_neg/(val_neg+val_pos); apriori(1,2) = val_pos/(val_neg+val_pos);
apriori(2,1) = act_neg/(act_neg+act_pos); apriori(2,2) = act_pos/(act_neg+act_pos);

for j = 1:size(nums,2)
    train_temp1 = eval(trains{j});
    test_temp1 = eval(tests{j});
    train_temp1(any(train_temp1 == 0,2),:) = [];
    test_temp1(any(test_temp1 == 0,2),:) = [];
    endd = size(train_temp1,2);
    for i = 2:endd
        mean1(1,i-1) = mean(train_temp1(any(train_temp1(:,1) == -1,2),i));
        mean1(2,i-1) = mean(train_temp1(any(train_temp1(:,1) == +1,2),i));
        var1(1,i-1) = var(train_temp1(any(train_temp1(:,1) == -1,2),i));
        var1(2,i-1) = var(train_temp1(any(train_temp1(:,1) == +1,2),i));
    end
    percBayes(j) = bayes(test_temp1,mean1(:,1:size(train_temp1,2)-1),var1(:,1:size(train_temp1,2)-1),apriori(mod(j,2)+1,:));
end

mean_perc_bayes = sum(percBayes) / size(percBayes,2);
display(mean_perc_bayes, 'Mean percentage from Bayes = ');


%% step 14

% **12** cases total

pca1 = processpca(train1_rf',0.000001);
testpca1 = processpca(test1_rf',0.000001);
pca2 = processpca(train2_rf',0.000001);
testpca2 = processpca(test2_rf',0.000001);
pca3 = processpca(train3_rf',0.000001);
testpca3 = processpca(test3_rf',0.000001);


train11_pca = [res_train1_a' pca1];
test11_pca = [res_test1_a' testpca1];
train12_pca = [res_train1_v' pca1];
test12_pca = [res_test1_v' testpca1];
train21_pca = [res_train2_a' pca2];
test21_pca = [res_test2_a' testpca2];
train22_pca = [res_train2_v' pca2];
test22_pca = [res_test2_v' testpca2];
train31_pca = [res_train3_a' pca3];
test31_pca = [res_test3_a' testpca3];
train32_pca = [res_train3_v' pca3];
test32_pca = [res_test3_v' testpca3];

nums2 = {11,12,21,22,31,32};
for i = 1:size(nums2,2)
    trains2{i} = sprintf('train%d_pca',nums2{i});
    tests2{i} = sprintf('test%d_pca',nums2{i});
end

k=3;
for i = 1:size(nums2,2)
    train_pca = eval(trains2{i});
    test_pca = eval(tests2{i});
    train_pca(any(train_pca == 0,2),:) = [];
    test_pca(any(test_pca == 0,2),:) = [];
    perc_pca(i) = nnrk(train_pca, test_pca, k);
end


mean_perc_pca = sum(perc_pca(perc_pca > 50)) / sum(perc_pca > 50);
display(mean_perc_pca,'Mean percentage of process PCA = ');


%% step 15

arff1 = [mir_all' act_thres'];
arff2 = [mir_all' val_thres'];
arff3 = [mfccs_all' act_thres'];
arff4 = [mfccs_all' val_thres'];
arff5 = [mir_all' mfccs_all' act_thres'];
arff6 = [mir_all' mfccs_all' val_thres'];

arff1(any(arff1 == 0,2),:) = [];
arff2(any(arff2 == 0,2),:) = [];
arff3(any(arff3 == 0,2),:) = [];
arff4(any(arff4 == 0,2),:) = [];
arff5(any(arff5 == 0,2),:) = [];
arff6(any(arff6 == 0,2),:) = [];

fileID = fopen('file1.arff','w');
fprintf(fileID,'@RELATION file1\n\n');
for i = 1:size(arff1,2)-1
    fprintf(fileID,'@ATTRIBUTE a%d NUMERIC\n',i);
end
fprintf(fileID,'@ATTRIBUTE class {-1,1}\n');
fprintf(fileID,'@DATA\n');
for i=1:size(arff1,1)
    for j =1:size(arff1,2)-1
        fprintf(fileID,'%5.3f,',arff1(i,j));
    end
    fprintf(fileID,'%d\n',arff1(i,j+1));
end
fclose(fileID);

fileID = fopen('file2.arff','w');
fprintf(fileID,'@RELATION file2\n\n');
for i = 1:size(arff2,2)-1
    fprintf(fileID,'@ATTRIBUTE a%d NUMERIC\n',i);
end
fprintf(fileID,'@ATTRIBUTE class {-1,1}\n');
fprintf(fileID,'@DATA\n');
for i=1:size(arff2,1)
    for j =1:size(arff2,2)-1
        fprintf(fileID,'%5.3f,',arff2(i,j));
    end
    fprintf(fileID,'%d\n',arff2(i,j+1));
end
fclose(fileID);

fileID = fopen('file3.arff','w');
fprintf(fileID,'@RELATION file3\n\n');
for i = 1:size(arff3,2)-1
    fprintf(fileID,'@ATTRIBUTE a%d NUMERIC\n',i);
end
fprintf(fileID,'@ATTRIBUTE class {-1,1}\n');
fprintf(fileID,'@DATA\n');
for i=1:size(arff3,1)
    for j =1:size(arff3,2)-1
        fprintf(fileID,'%5.3f,',arff3(i,j));
    end
    fprintf(fileID,'%d\n',arff3(i,j+1));
end
fclose(fileID);

fileID = fopen('file4.arff','w');
fprintf(fileID,'@RELATION file4\n\n');
for i = 1:size(arff4,2)-1
    fprintf(fileID,'@ATTRIBUTE a%d NUMERIC\n',i);
end
fprintf(fileID,'@ATTRIBUTE class {-1,1}\n');
fprintf(fileID,'@DATA\n');
for i=1:size(arff4,1)
    for j =1:size(arff4,2)-1
        fprintf(fileID,'%5.3f,',arff4(i,j));
    end
    fprintf(fileID,'%d\n',arff4(i,j+1));
end
fclose(fileID);

fileID = fopen('file5.arff','w');
fprintf(fileID,'@RELATION file5\n\n');
for i = 1:size(arff5,2)-1
    fprintf(fileID,'@ATTRIBUTE a%d NUMERIC\n',i);
end
fprintf(fileID,'@ATTRIBUTE class {-1,1}\n');
fprintf(fileID,'@DATA\n');
for i=1:size(arff5,1)
    for j =1:size(arff5,2)-1
        fprintf(fileID,'%5.3f,',arff5(i,j));
    end
    fprintf(fileID,'%d\n',arff5(i,j+1));
end
fclose(fileID);

fileID = fopen('file6.arff','w');
fprintf(fileID,'@RELATION file6\n\n');
for i = 1:size(arff6,2)-1
    fprintf(fileID,'@ATTRIBUTE a%d NUMERIC\n',i);
end
fprintf(fileID,'@ATTRIBUTE class {-1,1}\n');
fprintf(fileID,'@DATA\n');
for i=1:size(arff6,1)
    for j =1:size(arff6,2)-1
        fprintf(fileID,'%5.3f,',arff6(i,j));
    end
    fprintf(fileID,'%d\n',arff6(i,j+1));
end
fclose(fileID);
