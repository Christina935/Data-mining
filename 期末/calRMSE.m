function MSE = calRMSE(userItem,test)
num=size(test,1);
err = 0;
for i=1:num
%      err = err + (userItem(test(i,1),test(i,2))-test(i,3))*(userItem(test(i,1),test(i,2))-test(i,3));
   err = err + abs(userItem(test(i,1),test(i,2))-test(i,3));
end
if num==0
    MSE=222;
else
   MSE =err/num; 
end

end