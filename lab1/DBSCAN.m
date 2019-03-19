clear all;
clc;
%��ȡ���ݣ������Ƕ�ά��
%���ݼ���flame_cluster=2.txt �� Aggregation_cluster=7.txt , Jain_cluster=2.txt ,
%Pathbased_cluster=3.txt , Spiral_cluster=3.txt
% data = load('flame_cluster=2.txt');

udata =load('Mfeat.mat');
% data = udata.data_fou;
data = udata.data_fac;
% data = udata.data_kar;
%  data = udata.data_pix;
%  data = udata.data_zer;
% data = udata.data_mor;

%�������Eps��MinPts
MinPts = 10;
Eps = 100;
[num, dimension] = size(data); %��������
%Eps=((prod(max(data)-min(data))*MinPts*gamma(.5*dimension+1))/(num*sqrt(pi.^dimension))).^(1/dimension);  

%��ÿ�������������
data = [(1:num)' data];
[num, dimension] = size(data); %[��������,ÿ����������ά��]
types = zeros(num,1); %�������ֺ��ĵ�1���߽��0��������-1
class = zeros(num,1); %��¼ÿ�������������Ĵ�
visited = zeros(num, 1); %�жϸõ��Ƿ������0��ʾδ����
%��������е����֮���ŷʽ����
all_dis = zeros(num, num);
for i=1:num
    for j=i:num
        all_dis(i,j) = norm(data(i,:)-data(j,:));
        all_dis(j,i) = all_dis(i,j);
    end
end
%��������е����֮���ŷʽ����

cluster_num = 0;

%����ÿһ������
for i=1:num
    %�ҵ�û������ĵ�
    if visited(i)==0
        %ȡ�øõ㵽�������е�ľ���
        dis = all_dis(i,:);
        %�ҵ��뾶EPS�ڵ����е�
        Eps_point = find(dis <= Eps);
        %���ݵ������ֵ������
        %������
        if length(Eps_point)==1
            types(i) = -1;
            class(i) = -1;
            visited(i) = 1;
        end
        %�߽��
        if length(Eps_point)>1 && length(Eps_point) < MinPts+1
            types(i) = 0;
            class(i) = 0;
        end
        %���ĵ�
        if length(Eps_point) >= MinPts+1
            visited(i) = 1;
            cluster_num = cluster_num+1;
            types(i) = 1;
            %���ú��ĵ�EPS��Χ�ڵĵ㶼����ͬһ����
            class(Eps_point) = cluster_num;
            
            while ~isempty(Eps_point)
                %ȡEps_point��һ����ŵ�����
                point = data(Eps_point(1),:);
                visited(Eps_point(1))=1;
                Eps_point(1)=[];
                dis = all_dis(point(1,1),:);
                eps_point = find(dis<=Eps);
                %�����������
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

%���������δ�����������
noise = find(class==0);
class(noise(:,1)) = -1;
types(noise(:,1)) = -1;

%�������յľ���ͼ
% s = scatter(data(:,2), data(:,3), 20, 'blue','filled');
% title('ԭʼͼ��');
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
%         scatter(data(i,2), data(i,3), 20, 'c','filled');%����ɫ
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
disp(['����׼ȷ��ACCΪ',num2str(ACC(1))]);
disp(['��׼����ϢNMIΪ',num2str(ACC(2))]);
disp(['���ി��PURΪ',num2str(ACC(3))]);