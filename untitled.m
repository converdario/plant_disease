clear; clc; close all;

% Dati dei test
Test = { ...
    'iterate\_axioms', 'iterate\_primitives', 'axioms\_of\_type', 'axioms\_of\_types', ...
    'axiom\_count', 'primitives\_count', 'iterate\_import\_iris', 'has\_primitive', ...
    'has\_axiom', 'axiom\_count\_for\_type', 'axiom\_count\_for\_types', ...
    'class\_assert\_axiom', 'data\_prop\_assert\_axiom', 'obj\_prop\_assert\_axiom', ...
    'sub\_class\_axiom', 'extract\_subontology'};

Time_HOME = [1.93 1.96 3.35 2.31 1.84 2.07 1.87 2.54 1.93 1.81 1.98 2.64 1.98 1.50 1.93 2.71];
Time_TEST = [1.19 1.62 1.18 1.21 1.26 1.15 1.13 1.81 1.72 1.15 1.20 1.95 1.77 0.98 1.17 1.83];

RSS_HOME = [764 776 644 932 984 876 520 620 812 700 748 840 804 368 628 824];
RSS_TEST = [880 824 876 940 872 844 812 752 776 772 720 772 792 372 804 800];

%% 1️⃣ Grafico lineare: tempi HOME vs TEST
figure('Name','Tempi di esecuzione','Color','w');
plot(Time_HOME, '-o', 'LineWidth', 1.8); hold on;
plot(Time_TEST, '-s', 'LineWidth', 1.8);
set(gca, 'XTick', 1:length(Test), 'XTickLabel', Test, 'XTickLabelRotation', 60);
ylabel('Tempo (ms)');
title('Tempi di esecuzione dei test');
legend('HOME','TEST','Location','northwest');
grid on;

%% 2️⃣ Grafico lineare: crescita RSS HOME vs TEST
figure('Name','Crescita RSS','Color','w');
plot(RSS_HOME, '-o', 'LineWidth', 1.8); hold on;
plot(RSS_TEST, '-s', 'LineWidth', 1.8);
set(gca, 'XTick', 1:length(Test), 'XTickLabel', Test, 'XTickLabelRotation', 60);
ylabel('Crescita RSS (KiB)');
title('Crescita memoria RSS per test');
legend('HOME','TEST','Location','northwest');
grid on;

%% 3️⃣ Differenze percentuali
Delta_Tempo = ((Time_HOME - Time_TEST) ./ Time_HOME) * 100;
Delta_RSS   = ((RSS_HOME - RSS_TEST) ./ RSS_HOME) * 100;

%% 4️⃣ Grafico a barre: riduzione tempo
figure('Name','Differenza tempo','Color','w');
bar(Delta_Tempo, 'FaceColor', [0.3 0.6 0.9]);
set(gca, 'XTick', 1:length(Test), 'XTickLabel', Test, 'XTickLabelRotation', 60);
ylabel('Riduzione tempo (%)');
title('Differenza percentuale di tempo (HOME vs TEST)');
yline(0, '--k');
grid on;

%% 5️⃣ Grafico a barre: riduzione memoria RSS
figure('Name','Differenza RSS','Color','w');
bar(Delta_RSS, 'FaceColor', [0.5 0.8 0.5]);
set(gca, 'XTick', 1:length(Test), 'XTickLabel', Test, 'XTickLabelRotation', 60);
ylabel('Riduzione RSS (%)');
title('Differenza percentuale di memoria RSS (HOME vs TEST)');
yline(0, '--k');
grid on;
