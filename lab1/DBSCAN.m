clear all;
clc;
%读取数据，数据是二维的
%数据集：flame_cluster=2.txt ， Aggregation_cluster=7.txt , Jain_cluster=2.txt ,
%Pathbased_cluster=3.txt , Spiral_cluster=3.txt
% data = load('flame_cluster=2.txt');

udata =load('Mfeat.mat');
% data = udata.data_fou;
data = udata.data_fac;
% data = udata.data_kar;
%  data = udata.data_pix;
%  data = udata.data_zer;
% data = udata.data_mor;

%定义参数Eps和MinPts
MinPts = 10;
Eps = 100;
[num, dimension] = size(data); %样本数量
%Eps=((prod(max(data)-min(data))*MinPts*gamma(.5*dimension+1))/(num*sqrt(pi.^dimension))).^(1/dimension);  

%给每个样本加上序号
data = [(1:num)' data];
[num, dimension] = size(data); %[样本数量,每个样本特征维度]
types = zeros(num,1); %用于区分核心点1，边界点0和噪声点-1
class = zeros(num,1); %记录每个样本的所属的簇
visited = zeros(num, 1); %判断该点是否处理过，0表示未处理
%计算矩阵中点与点之间的欧式距离
all_dis = zeros(num, num);
for i=1:num
    for j=i:num
        all_dis(i,j) = norm(data(i,:)-data(j,:));
        all_dis(j,i) = all_dis(i,j);
    end
end
%计算矩阵中点与点之间的欧式距离

cluster_num = 0;

%处理每一个样本
for i=1:num
    %找到没处理过的点
    if visited(i)==0
        %取得该点到其它所有点的距离
        dis = all_dis(i,:);
        %找到半径EPS内的所有点
        Eps_point = find(dis <= Eps);
        %根据点数区分点的类型
        %噪声点
        if length(Eps_point)==1
            types(i) = -1;
            class(i) = -1;
            visited(i) = 1;
        end
        %边界点
        if length(Eps_point)>1 && length(Eps_point) < MinPts+1
            types(i) = 0;
            class(i) = 0;
        end
        %核心点
        if length(Eps_point) >= MinPts+1
            visited(i) = 1;
            cluster_num = cluster_num+1;
            types(i) = 1;
            %将该核心点EPS范围内的点都划到同一个簇
            class(Eps_point) = cluster_num;
            
            while ~isempty(Eps_point)
                %取Eps_point第一行序号的样本
                point = data(Eps_point(1),:);
                visited(Eps_point(1))=1;
                Eps_point(1)=[];
                dis = all_dis(point(1,1),:);
                eps_point = find(dis<=Eps);
                %处理非噪声点
                if length(eps_point) >1
                    class(eps_point(:,1))=cluster_num;
                    visited(eps_point(:,1))=1;
                    if length(eps_point) >= MinPts+1
                        types(point(1,1))=1;
                    else
                        types(point(1,1))=0;
                    end
                    
                    for j=1:length(eps_point)
                        if visited(eps_point(j))==0
                            Eps_point = [Eps_point eps_point(1)];
                        end    
                    end
                end
            end
        end
    end 
end

%最后处理所有未分类的噪声点
noise = find(class==0);
class(noise(:,1)) = -1;
types(noise(:,1)) = -1;

%画出最终的聚类图
% s = scatter(data(:,2), data(:,3), 20, 'blue','filled');
% title('原始图像');
% figure;
% hold on;
% for i=1:num
%     if class(i) == -1
%         scatter(data(i,2), data(i,3), 20, 'blue','filled');
%     elseif class(i) == 1
%         scatter(data(i,2), data(i,3), 20, 'green','filled');
%     elseif class(i) ==2
%         scatter(data(i,2), data(i,3), 20, 'yellow','filled');
%     elseif class(i) == 3
%         scatter(data(i,2), data(i,3), 20, 'c','filled');%蓝绿色
%     elseif class(i) == 5
%         scatter(data(i,2), data(i,3), 20, 'black','filled');
%     elseif class(i) == 6
%         scatter(data(i,2), data(i,3), 20, 'red','filled');
%     else
%         scatter(data(i,2), data(i,3), 20, 'm','filled');
%     end
% end
% hold off;

clu = udata.classid;
ACC = ClusteringMeasure(class,clu);
disp(['聚类准确率ACC为',num2str(ACC(1))]);
disp(['标准化信息NMI为',num2str(ACC(2))]);
disp(['聚类纯度PUR为',num2str(ACC(3))]);