function ncut = countNcut(n,m,cluster,W,num)
    if n~=0 && m~=0
        r1 = find(cluster==n);
        cluster(r1)=m;
    end
    ncut = 0;
    difNum = unique(cluster,'rows');
    nc = size(difNum,1);
    rowSum = sum(W,2);
    for i=1:nc
        row = find(cluster==difNum(i)); %找出哪些点处于该簇
        nr = size(row,1);
        cut = 0;
        assoc = 0;
        for j=1:nr
            for k=1:num
                lib = ismember(k,row);
                if lib==0
                    cut = cut+ rowSum(row(j));
                end
            end
            assoc = assoc+sum(W(row(j)),2);
        end
        ncut = ncut + cut/assoc;
    end
end