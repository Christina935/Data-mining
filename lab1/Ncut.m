clear all;
clc;
%数据集：flame_cluster=2.txt ， Aggregation_cluster=7.txt , Jain_cluster=2.txt ,
%Pathbased_cluster=3.txt , Spiral_cluster=3.txt
udata =load('Mfeat.mat');
% data = udata.data_fou;
%  data = udata.data_fac;
% data = udata.data_kar;
%  data = udata.data_pix;
%  data = udata.data_zer;
data = udata.data_mor;
% data = load('Spiral_cluster=3.txt');

%绘制数据，2维散点图
% s = scatter(data(:,1), data(:,2), 20, 'blue','filled');
% title('原始数据：蓝圈；初始簇心');

num = size(data,1); %样本数
%构建权重矩阵——高斯距离
data = data/max(max(abs(data))); %归一化
sigma = 0.01;
dis_fir = pdist(data); %data行与行之间的距离
dis_zero = squareform(dis_fir); % 对角线化为0，ij代表第xi和xj的距离 ||xi-xj||
dis_double = dis_zero.*dis_zero; %||xi-xj||^2
top = -dis_double/(2*sigma*sigma);
res = spfun(@exp,top);
S = full(res);
W = S;
%计算归一化矩阵D
D = full(sparse(1:num,1:num,sum(W)));%D为相似度矩阵S中一列元素加起来放到对角线上
%拉普拉斯矩阵 
% L=D-W, 归一化L=D^(-1/2) * L * D^(-1/2) = I-D^(-1/2) * W * D^(-1/2)
L =eye(num)-(D^(-1/2)*W*D^(-1/2));
%找特征值特征向量T并排序，找前K个
K = 10;
%'sm' 绝对值最小特征值
[T,~] = eigs(L, K, 'SM');
%对特征向量求K-means
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
%         scatter(data(i,1), data(i,2), 20, 'c','filled');%蓝绿色
%     elseif c(i,:) == 5
%         scatter(data(i,1), data(i,2), 20, 'black','filled');
%     elseif c(i,:) == 6
%         scatter(data(i,1), data(i,2), 20, 'm','filled');%紫红色
%     elseif c(i,:)==7
%         scatter(data(i,1), data(i,2), 20, 'red','filled');%紫红色
%      end
% end
% hold off;

clu = udata.classid;
ACC = ClusteringMeasure(c,clu);
disp(['聚类准确率ACC为',num2str(ACC(1))]);
disp(['标准化信息NMI为',num2str(ACC(2))]);
disp(['聚类纯度PUR为',num2str(ACC(3))]);
