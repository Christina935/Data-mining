clear all;
clc;
%udata =load('cornell.mat');
% udata =load('texas.mat');
% udata =load('washington.mat');
udata =load('wisconsin.mat');
% data = udata.A;
data = udata.F;

num = size(data,1); %样本数
%构建权重矩阵——高斯距离
data = data/max(max(abs(data))); %归一化
sigma = 7.15;
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
K = 50;
%'sm' 绝对值最小特征值
[T,~] = eigs(L, K, 'SM');
%对特征向量求K-means
cluster = kmeans(T,K);

k=5;
%初始ncut
ncut = countNcut(0,0,cluster,W,num);
while 1
    difNum = unique(cluster,'rows');
    nc = size(difNum,1);
    if nc==k
        break;
    end
    a = rand(2,1);%生成两行一列 0~1之间的随机小数
    b = a*nc+1; %将随机小数映射到1~nc之间，包括nc
    c = floor(b); %取b的整数部分
    newcut = countNcut(difNum(c(1)),difNum(c(2)),cluster,W,num);
    if newcut < ncut
        r1 = find(cluster==difNum(c(1)));
        cluster(r1)=difNum(c(2));
        ncut = newcut;
    end    
end
%改变簇的序号， 按大小赋予1,2
dif = unique(cluster,'rows');
zn = 1;
for z=1:size(dif)
    rz = (cluster==dif(z));
    cluster(rz) = zn;
    zn = zn+1;
end

clu = udata.label;
ACC = ClusteringMeasure(cluster,clu);
disp(['聚类准确率ACC为',num2str(ACC(1))]);
disp(['标准化信息NMI为',num2str(ACC(2))]);
disp(['聚类纯度PUR为',num2str(ACC(3))]);
