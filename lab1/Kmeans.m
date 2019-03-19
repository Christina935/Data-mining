%决定质心的个数
k = 10;
%读取数据，数据是二维的
%数据集：flame_cluster=2.txt ， Aggregation_cluster=7.txt , Jain_cluster=2.txt ,
%Pathbased_cluster=3.txt , Spiral_cluster=3.txt
% data = load('Aggregation_cluster=7.txt');
udata =load('Mfeat.mat');
% data = udata.data_fou;
%  data = udata.data_fac;
% data = udata.data_kar;
%  data = udata.data_pix;
%  data = udata.data_zer;
data = udata.data_mor;
% x = data(:,1);
% y = data(:,2);
%绘制数据，2维散点图
% s = scatter(x, y, 20, 'blue','filled');
% title('原始数据：蓝圈；初始簇心');
% 初始化簇心
num = size(data,1); %样本数量
dimension = size(data,2); %每个样本特征维度
% 指定簇心初始位置：随机选择k个数
clusters_center = zeros(k, dimension);
for i=1:k
     clusters_center(i,:)=data(randi(num,1),:);
end
% hold on; %在上次散点图的基础上，准备下次绘图
%绘制初始簇心(实心圆点，表示簇心的初始位置）
% scatter(clusters_center(:,1),clusters_center(:,2),'red','filled');
c = zeros(num, 1); %每个样本所属簇的编号
%设置迭代次数
iterator = 100;
iter_num = 1;
PRECISION = 0.0001; %当更新的质心和原来的质心的距离小于该值，则认定为收敛
while 1
    %遍历所有样本数据，确定所属簇
    for i=1:num
        %记录该样本到每个质心的距离
        distance = zeros(k,1);
        for j=1:k
            %计算该样本到每个质心的欧式距离
            distance(j,1) = norm(data(i,:)-clusters_center(j,:));
        end
        %找出最小的距离
        [min_dis, row] = min(distance);
        c(i,:) = row;
    end
    %遍历所有样本数据，更新质心
    convergence=0; %判断是否收敛
    for i=1:k
        total_dis = 0;
        total_num = 0;
        for j=1:num
            total_dis = total_dis + (c(j,:)==i)*data(j,:);
            total_num = total_num + (c(j,:)==i);
        end
        new_cluster = total_dis/total_num;
        %当更新的质心和原来的质心的距离小于PRECISION，则认定为收敛
        if(norm(clusters_center(i,:)-new_cluster) < PRECISION )
            converagence = 1;
        end
        %更新质心
        clusters_center(i,:) = new_cluster;
    end
%     figure;
%     f = scatter(x, y, 20, 'blue');
%     hold on;
%     scatter(clusters_center(:,1),clusters_center(:,2),'red','filled');
%     title(['第',num2str(iter_num),'次迭代'])
    
    if convergence
        disp(['收敛于第',num2str(iter_num),'次迭代']);
        break;
    end
    if iter_num < iterator
        iter_num = iter_num + 1;
    else
        disp(['已经迭代了',num2str(iterator),'次']);
        break;
    end 
    
end
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
%     elseif c(i,:) == 7
%         scatter(data(i,1), data(i,2), 20, 'red','filled');%紫红色
%      end
% end
% %f = scatter(x, y, 20, 'blue');
% % scatter(clusters_center(:,1),clusters_center(:,2),'red','filled');
% hold off;    

clu = udata.classid;
ACC = ClusteringMeasure(c,clu);
disp(['聚类准确率ACC为',num2str(ACC(1))]);
disp(['标准化信息NMI为',num2str(ACC(2))]);
disp(['聚类纯度PUR为',num2str(ACC(3))]);


