clear all;
clc;
% udata =load('cornell.mat');
% udata =load('texas.mat');
% udata =load('washington.mat');
udata =load('wisconsin.mat');
%dataF = udata.F;
%邻接矩阵
dataA = udata.A; %全局初始化的邻接表，不参与后期运算

n = size(dataA,1);% n:结点个数

%根据邻接矩阵计算边数
m = 0; %边数
for i=1:n
    for j=i+1:n
        if dataA(i,j)==1
            m = m+1;
        end
    end
end

%记录每个结点的簇
cluster = zeros(n,1);
%初始化每个结点为一个簇
for i=1:n
    cluster(i) = i;
end

%迭代次数
count = 0;
%标记是否发生过更新
update = 0;
%图不带权，则将点的度数视为权重~ki~边的权重视为1
weight = sum(dataA~=0,2);
%第一重循环，每一次循环rebuild一次图
while 1
    count = count + 1;
    update = 0;
    %找出有多少个簇difNum,以及每个簇difNum最小从哪行row开始
    [difNum,r] = unique(cluster,'rows');
    num = size(difNum,1);
    ki = zeros(num,1);
    in = zeros(num,1);
    %对于每一个结点 计算出ki,in,tot
    for i=1:num
        %计算ki
        row = find(cluster==difNum(i)); %找出哪些点处于该簇
        nr = size(row,1);
        for j=1:nr
            %把属于同一个簇的点的ki加起来就是该簇的ki
            ki(i) = ki(i)+ weight(row(j)); 
            if nr > 1
                for k=j+1:nr
                    %把起点和终点都属于该簇的边的相加
                    in(i) = in(i) + dataA(row(j),row(k));
                end
            end 
        end    
    end
    tot = ki;
    ki_in = zeros(num,num);
    for i=1:num
        r1 = find(cluster==difNum(i));
        cutQ = zeros(1,num);
        %计算ki_in
         for j=i+1:num
            r2 = find(cluster==difNum(j));
            for k=1:size(r1,1)
                for p=1:size(r2,1)
                    %找出r1中的点与r2中的点相连的边
                    if dataA(r1(k),r2(p))==1
                        ki_in(i,j) = ki_in(i,j)+1;
                    end
                end
            end
            ki_in(j,i) = ki_in(i,j);
         end
      
        for h =1:num
            if h~=i
                %相对增益
                cutQ(h) = ki_in(i) - tot(h)*ki(i)/m;
                %绝对增益
                %cutQ(h) = ((in(h)+ki_in(i))/m - ((tot(h)+ki(i))/(2*m))^2) 
                %-(in(h)/(2*m)-(tot(h)/(2*m))^2-(ki(i)/(2*m))^2);
            end
        end
        [big,pos] = max(cutQ(h));
        if big > 0
            cluster(r1) = difNum(pos);
            update = 1;
        end  
    end
    %如果结点所属的簇不再改变
    if update==0
        break;
    end
end

%改变簇的序号， 按大小赋予1,2
dif = unique(cluster,'rows');
zn = 1;
for z=1:size(dif)
    rz = find(cluster==dif(z));
    cluster(rz) = zn;
    zn = zn+1;
end

clu = udata.label;
ACC = ClusteringMeasure(cluster,clu);
disp(['聚类准确率ACC为',num2str(ACC(1))]);
disp(['标准化信息NMI为',num2str(ACC(2))]);
disp(['聚类纯度PUR为',num2str(ACC(3))]);
