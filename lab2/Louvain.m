clear all;
clc;
% udata =load('cornell.mat');
% udata =load('texas.mat');
% udata =load('washington.mat');
udata =load('wisconsin.mat');
%dataF = udata.F;
%�ڽӾ���
dataA = udata.A; %ȫ�ֳ�ʼ�����ڽӱ��������������

n = size(dataA,1);% n:������

%�����ڽӾ���������
m = 0; %����
for i=1:n
    for j=i+1:n
        if dataA(i,j)==1
            m = m+1;
        end
    end
end

%��¼ÿ�����Ĵ�
cluster = zeros(n,1);
%��ʼ��ÿ�����Ϊһ����
for i=1:n
    cluster(i) = i;
end

%��������
count = 0;
%����Ƿ���������
update = 0;
%ͼ����Ȩ���򽫵�Ķ�����ΪȨ��~ki~�ߵ�Ȩ����Ϊ1
weight = sum(dataA~=0,2);
%��һ��ѭ����ÿһ��ѭ��rebuildһ��ͼ
while 1
    count = count + 1;
    update = 0;
    %�ҳ��ж��ٸ���difNum,�Լ�ÿ����difNum��С������row��ʼ
    [difNum,r] = unique(cluster,'rows');
    num = size(difNum,1);
    ki = zeros(num,1);
    in = zeros(num,1);
    %����ÿһ����� �����ki,in,tot
    for i=1:num
        %����ki
        row = find(cluster==difNum(i)); %�ҳ���Щ�㴦�ڸô�
        nr = size(row,1);
        for j=1:nr
            %������ͬһ���صĵ��ki���������Ǹôص�ki
            ki(i) = ki(i)+ weight(row(j)); 
            if nr > 1
                for k=j+1:nr
                    %�������յ㶼���ڸôصıߵ����
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
        %����ki_in
         for j=i+1:num
            r2 = find(cluster==difNum(j));
            for k=1:size(r1,1)
                for p=1:size(r2,1)
                    %�ҳ�r1�еĵ���r2�еĵ������ı�
                    if dataA(r1(k),r2(p))==1
                        ki_in(i,j) = ki_in(i,j)+1;
                    end
                end
            end
            ki_in(j,i) = ki_in(i,j);
         end
      
        for h =1:num
            if h~=i
                %�������
                cutQ(h) = ki_in(i) - tot(h)*ki(i)/m;
                %��������
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
    %�����������Ĵز��ٸı�
    if update==0
        break;
    end
end

%�ı�ص���ţ� ����С����1,2
dif = unique(cluster,'rows');
zn = 1;
for z=1:size(dif)
    rz = find(cluster==dif(z));
    cluster(rz) = zn;
    zn = zn+1;
end

clu = udata.label;
ACC = ClusteringMeasure(cluster,clu);
disp(['����׼ȷ��ACCΪ',num2str(ACC(1))]);
disp(['��׼����ϢNMIΪ',num2str(ACC(2))]);
disp(['���ി��PURΪ',num2str(ACC(3))]);
