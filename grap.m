close all
[signal, fs] = audioread("signal_noise101.wav");
[noise, ~] = audioread("noise101.wav");
[rst, ~] = audioread("rst.wav");
subplot(3,1,1)
plot(1:length(signal), signal);
title("source signal + noise")
subplot(3,1,2)
plot(1:length(noise), noise);
title("source noise")
subplot(3,1,3)
plot(1:length(rst), rst);
title("result signal")
figure
specgram(signal, 257, fs, hann(257), 128);
title("source signal + noise")
set(gca,'CLim',[-65 15])
figure
specgram(rst, 257, fs, hann(257), 128);
title("result signal")
set(gca,'CLim',[-65 15])