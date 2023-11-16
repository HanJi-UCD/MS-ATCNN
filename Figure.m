% Fig1: effect of UTI
data = table2array(Fig1ThrvsUTI);
categories = {'1', '2', '3', '4', '5'};

V1 = data(data(:,1) == 100, :);
V2 = data(data(:,1) == 200, :);
V3 = data(data(:,1) == 300, :);
V4 = data(data(:,1) == 400, :);
V5 = data(data(:,1) == 500, :);
data_V1 = [mean(V1(:, 2)), mean(V1(:, 3)), mean(V1(:, 4)), mean(V1(:, 5)), mean(V1(:, 6))];
data_V2 = [mean(V2(:, 2)), mean(V2(:, 3)), mean(V2(:, 4)), mean(V2(:, 5)), mean(V2(:, 6))];
data_V3 = [mean(V3(:, 2)), mean(V3(:, 3)), mean(V3(:, 4)), mean(V3(:, 5)), mean(V3(:, 6))];
data_V4 = [mean(V4(:, 2)), mean(V4(:, 3)), mean(V4(:, 4)), mean(V4(:, 5)), mean(V4(:, 6))];
data_V5 = [mean(V5(:, 2)), mean(V5(:, 3)), mean(V5(:, 4)), mean(V5(:, 5)), mean(V5(:, 6))];
% 创建图形对象
figure;

% 绘制左侧柱状图
bar([data_V1; data_V2; data_V3; data_V4; data_V5]);
hold on;
set(gca, 'XTick', 1:numel(categories), 'XTickLabel', categories);
xlabel('Fixed veloity (m/s)');
ylabel('Achievable throughput (Mbps)');
set(gca, 'YColor');

% 添加图例
legend('ATCNN@10ms', 'MS-ATCNN', 'ATCNN@200ms', 'ATCNN@500ms', 'ATCNN@1000ms');
% 显示图形
grid on;

%% Fig2: effect of UE number
data = table2array(Fig2ThrvsUEnumv1);

UE1 = data(data(:,1) == 10, :)*10;
UE2 = data(data(:,1) == 20, :)*20;
UE3 = data(data(:,1) == 30, :)*30;
UE4 = data(data(:,1) == 40, :)*40;
UE5 = data(data(:,1) == 50, :)*50;
UE6 = data(data(:,1) == 60, :)*60;
UE7 = data(data(:,1) == 70, :)*70;
UE8 = data(data(:,1) == 80, :)*80;
UE9 = data(data(:,1) == 90, :)*90;
UE10 = data(data(:,1) == 100, :)*100;

data_UE1 = mean(UE1);
data_UE2 = mean(UE2);
data_UE3 = mean(UE3);
data_UE4 = mean(UE4);
data_UE5 = mean(UE5);
data_UE6 = mean(UE6);
data_UE7 = mean(UE7);
data_UE8 = mean(UE8);
data_UE9 = mean(UE9);
data_UE10 = mean(UE10);

final_data = [data_UE1; data_UE2; data_UE3; data_UE4; data_UE5; data_UE6; data_UE7; data_UE8; data_UE9; data_UE10];

figure
plot(10:10:100, final_data(:, 2));
hold on
plot(10:10:100, final_data(:, 3));
plot(10:10:100, final_data(:, 4));
plot(10:10:100, final_data(:, 5));
plot(10:10:100, final_data(:, 6));
%% Fig3: effect of velocity
data = table2array(Fig3ThrvsVelocity);
V1 = mean(data(data(:,1) == 1, :));
V2 = mean(data(data(:,1) == 2, :));
V3 = mean(data(data(:,1) == 3, :));
V4 = mean(data(data(:,1) == 4, :));
V5 = mean(data(data(:,1) == 5, :));

data_V1 = [V1; V2; V3; V4; V5]*50;

categories = {'1', '2', '3', '4', '5'};
figure;
% 绘制左侧柱状图
bar([data_V1(1,2:6); data_V1(2,2:6); data_V1(3,2:6); data_V1(4,2:6); data_V1(5,2:6)]);
hold on;
set(gca, 'XTick', 1:numel(categories), 'XTickLabel', categories);
xlabel('Average veloity (m/s)');
ylabel('Achievable throughput (Mbps)');
set(gca, 'YColor');
%% Fig4:
V1 = data(data(:,1) == 0.01, :);
V2 = data(data(:,1) == 0.02, :);
V3 = data(data(:,1) == 0.03, :);
V4 = data(data(:,1) == 0.04, :);
V5 = data(data(:,1) == 0.05, :);
data_V1 = mean(V1);
data_V2 = mean(V2);
data_V3 = mean(V3);
data_V4 = mean(V4);
data_V5 = mean(V5);
MS_ATCNN = [data_V1(2), data_V2(2), data_V3(2), data_V4(2), data_V5(2)];
ATCNN = [data_V1(3), data_V2(3), data_V3(3), data_V4(3), data_V5(3)];
GT = [data_V1(4), data_V2(4), data_V3(4), data_V4(4), data_V5(4)];
SSS = [data_V1(5), data_V2(5), data_V3(5), data_V4(5), data_V5(5)];

figure
plot(1:1:5, MS_ATCNN);
hold on
plot(1:1:5, ATCNN);
plot(1:1:5, GT);
plot(1:1:5, SSS);

%% 
data = table2array(Fig1ThrvsUTI);
UE1 = mean(data(data(:,1) == 10, :))*10;
UE2 = mean(data(data(:,1) == 20, :))*20;
UE3 = mean(data(data(:,1) == 30, :))*30;
UE4 = mean(data(data(:,1) == 40, :))*40;
UE5 = mean(data(data(:,1) == 50, :))*50;
UE6 = mean(data(data(:,1) == 60, :))*60;
UE7 = mean(data(data(:,1) == 70, :))*70;
UE8 = mean(data(data(:,1) == 80, :))*80;
UE9 = mean(data(data(:,1) == 90, :))*90;
UE10 = mean(data(data(:,1) == 100, :))*100;

data_V1 = [UE1; UE2; UE3; UE4; UE5; UE6; UE7; UE8; UE9; UE10];

UE_num = 10:10:100;

figure
plot(UE_num, data_V1(:,2));
% hold on
% plot(UE_num, data_V1(:,3));
% plot(UE_num, data_V1(:,4));
% plot(UE_num, data_V1(:,5));
%%



