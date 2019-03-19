%�������ĵĸ���
k = 10;
%��ȡ���ݣ������Ƕ�ά��
%���ݼ���flame_cluster=2.txt �� Aggregation_cluster=7.txt , Jain_cluster=2.txt ,
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
%�������ݣ�2άɢ��ͼ
% s = scatter(x, y, 20, 'blue','filled');
% title('ԭʼ���ݣ���Ȧ����ʼ����');
% ��ʼ������
num = size(data,1); %��������
dimension = size(data,2); %ÿ����������ά��
% ָ�����ĳ�ʼλ�ã����ѡ��k����
clusters_center = zeros(k, dimension);
for i=1:k
     clusters_center(i,:)=data(randi(num,1),:);
end
% hold on; %���ϴ�ɢ��ͼ�Ļ����ϣ�׼���´λ�ͼ
%���Ƴ�ʼ����(ʵ��Բ�㣬��ʾ���ĵĳ�ʼλ�ã�
% scatter(clusters_center(:,1),clusters_center(:,2),'red','filled');
c = zeros(num, 1); %ÿ�����������صı��
%���õ�������
iterator = 100;
iter_num = 1;
PRECISION = 0.0001; %�����µ����ĺ�ԭ�������ĵľ���С�ڸ�ֵ�����϶�Ϊ����
while 1
    %���������������ݣ�ȷ��������
    for i=1:num
        %��¼��������ÿ�����ĵľ���
        distance = zeros(k,1);
        for j=1:k
            %�����������ÿ�����ĵ�ŷʽ����
            distance(j,1) = norm(data(i,:)-clusters_center(j,:));
        end
        %�ҳ���С�ľ���
        [min_dis, row] = min(distance);
        c(i,:) = row;
    end
    %���������������ݣ���������
    convergence=0; %�ж��Ƿ�����
    for i=1:k
        total_dis = 0;
        total_num = 0;
        for j=1:num
            total_dis = total_dis + (c(j,:)==i)*data(j,:);
            total_num = total_num + (c(j,:)==i);
        end
        new_cluster = total_dis/total_num;
        %�����µ����ĺ�ԭ�������ĵľ���С��PRECISION�����϶�Ϊ����
        if(norm(clusters_center(i,:)-new_cluster) < PRECISION )
            converagence = 1;
        end
        %��������
        clusters_center(i,:) = new_cluster;
    end
%     figure;
%     f = scatter(x, y, 20, 'blue');
%     hold on;
%     scatter(clusters_center(:,1),clusters_center(:,2),'red','filled');
%     title(['��',num2str(iter_num),'�ε���'])
    
    if convergence
        disp(['�����ڵ�',num2str(iter_num),'�ε���']);
        break;
    end
    if iter_num < iterator
        iter_num = iter_num + 1;
    else
        disp(['�Ѿ�������',num2str(iterator),'��']);
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
%         scatter(data(i,1), data(i,2), 20, 'c','filled');%����ɫ
%     elseif c(i,:) == 5
%         scatter(data(i,1), data(i,2), 20, 'black','filled');
%     elseif c(i,:) == 6
%         scatter(data(i,1), data(i,2), 20, 'm','filled');%�Ϻ�ɫ
%     elseif c(i,:) == 7
%         scatter(data(i,1), data(i,2), 20, 'red','filled');%�Ϻ�ɫ
%      end
% end
% %f = scatter(x, y, 20, 'blue');
% % scatter(clusters_center(:,1),clusters_center(:,2),'red','filled');
% hold off;    

clu = udata.classid;
ACC = ClusteringMeasure(c,clu);
disp(['����׼ȷ��ACCΪ',num2str(ACC(1))]);
disp(['��׼����ϢNMIΪ',num2str(ACC(2))]);
disp(['���ി��PURΪ',num2str(ACC(3))]);


