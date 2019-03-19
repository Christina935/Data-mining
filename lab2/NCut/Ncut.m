clear all;
clc;
%udata =load('cornell.mat');
% udata =load('texas.mat');
% udata =load('washington.mat');
udata =load('wisconsin.mat');
% data = udata.A;
data = udata.F;

num = size(data,1); %������
%����Ȩ�ؾ��󡪡���˹����
data = data/max(max(abs(data))); %��һ��
sigma = 7.15;
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
K = 50;
%'sm' ����ֵ��С����ֵ
[T,~] = eigs(L, K, 'SM');
%������������K-means
cluster = kmeans(T,K);

k=5;
%��ʼncut
ncut = countNcut(0,0,cluster,W,num);
while 1
    difNum = unique(cluster,'rows');
    nc = size(difNum,1);
    if nc==k
        break;
    end
    a = rand(2,1);%��������һ�� 0~1֮������С��
    b = a*nc+1; %�����С��ӳ�䵽1~nc֮�䣬����nc
    c = floor(b); %ȡb����������
    newcut = countNcut(difNum(c(1)),difNum(c(2)),cluster,W,num);
    if newcut < ncut
        r1 = find(cluster==difNum(c(1)));
        cluster(r1)=difNum(c(2));
        ncut = newcut;
    end    
end
%�ı�ص���ţ� ����С����1,2
dif = unique(cluster,'rows');
zn = 1;
for z=1:size(dif)
    rz = (cluster==dif(z));
    cluster(rz) = zn;
    zn = zn+1;
end

clu = udata.label;
ACC = ClusteringMeasure(cluster,clu);
disp(['����׼ȷ��ACCΪ',num2str(ACC(1))]);
disp(['��׼����ϢNMIΪ',num2str(ACC(2))]);
disp(['���ി��PURΪ',num2str(ACC(3))]);
