clear all;
clc;
%���ݼ���flame_cluster=2.txt �� Aggregation_cluster=7.txt , Jain_cluster=2.txt ,
%Pathbased_cluster=3.txt , Spiral_cluster=3.txt
udata =load('Mfeat.mat');
% data = udata.data_fou;
%  data = udata.data_fac;
% data = udata.data_kar;
%  data = udata.data_pix;
%  data = udata.data_zer;
data = udata.data_mor;
% data = load('Spiral_cluster=3.txt');

%�������ݣ�2άɢ��ͼ
% s = scatter(data(:,1), data(:,2), 20, 'blue','filled');
% title('ԭʼ���ݣ���Ȧ����ʼ����');

num = size(data,1); %������
%����Ȩ�ؾ��󡪡���˹����
data = data/max(max(abs(data))); %��һ��
sigma = 0.01;
dis_fir = pdist(data); %data������֮��ľ���
dis_zero = squareform(dis_fir); % �Խ��߻�Ϊ0��ij�����xi��xj�ľ��� ||xi-xj||
dis_double = dis_zero.*dis_zero; %||xi-xj||^2
top = -dis_double/(2*sigma*sigma);
res = spfun(@exp,top);
S = full(res);
W = S;
%�����һ������D
D = full(sparse(1:num,1:num,sum(W)));%DΪ���ƶȾ���S��һ��Ԫ�ؼ������ŵ��Խ�����
%������˹���� 
% L=D-W, ��һ��L=D^(-1/2) * L * D^(-1/2) = I-D^(-1/2) * W * D^(-1/2)
L =eye(num)-(D^(-1/2)*W*D^(-1/2));
%������ֵ��������T��������ǰK��
K = 10;
%'sm' ����ֵ��С����ֵ
[T,~] = eigs(L, K, 'SM');
%������������K-means
c = kmeans(T,K);
% figure;
% hold on;
% for i=1:num
%     if c(i,:)== 1 
%         scatter(data(i,1), data(i,2), 20, 'blue','filled');
%     elseif c(i,:) == 2
%         scatter(data(i,1), data(i,2), 20, 'green','filled');
%     elseif c(i,:) == 3
%         scatter(data(i,1), data(i,2), 20, 'yellow','filled');
%     elseif c(i,:) == 4
%         scatter(data(i,1), data(i,2), 20, 'c','filled');%����ɫ
%     elseif c(i,:) == 5
%         scatter(data(i,1), data(i,2), 20, 'black','filled');
%     elseif c(i,:) == 6
%         scatter(data(i,1), data(i,2), 20, 'm','filled');%�Ϻ�ɫ
%     elseif c(i,:)==7
%         scatter(data(i,1), data(i,2), 20, 'red','filled');%�Ϻ�ɫ
%      end
% end
% hold off;

clu = udata.classid;
ACC = ClusteringMeasure(c,clu);
disp(['����׼ȷ��ACCΪ',num2str(ACC(1))]);
disp(['��׼����ϢNMIΪ',num2str(ACC(2))]);
disp(['���ി��PURΪ',num2str(ACC(3))]);
