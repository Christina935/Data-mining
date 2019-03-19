function RMSE = calRMSE(train,test,user,item)
num=0;
err = 0;
for i=1:user
    r = find(test(i,:)~=-1);
    nr = size(r,2);
    num = num + nr;
    for j=1:nr
        err = err + (test(i,r(j))-train(i,r(j)))*(test(i,r(j))-train(i,r(j)));
    end
end
RMSE = sqrt(err/num);
end