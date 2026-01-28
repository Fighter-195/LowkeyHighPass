clc; clear;

%% ===== CONFIG =====
librispeechRoot = "C:\Users\aweso\Documents\SPcup_final\LibriSpeech\train-clean-100";
outMale   = "C:\Users\aweso\Documents\SPcup_final\Male";
outFemale = "C:\Users\aweso\Documents\SPcup_final\Female";

MAX_MALE   = 500;
MAX_FEMALE = 500;

if ~exist(outMale,'dir'), mkdir(outMale); end
if ~exist(outFemale,'dir'), mkdir(outFemale); end

%% ===== READ METADATA =====
meta = readlines(fullfile(fileparts(librispeechRoot), "SPEAKERS.TXT"));

maleCount = 0;
femaleCount = 0;

h = waitbar(0, 'Splitting LibriSpeech (limited to 200 files)...');
tic;

%% ===== MAIN LOOP =====
for i = 1:length(meta)
    if maleCount >= MAX_MALE && femaleCount >= MAX_FEMALE
        break;
    end

    line = strtrim(meta(i));
    if startsWith(line,';') || line == ""
        continue;
    end

    parts  = split(line,'|');
    spkID  = strtrim(parts(1));
    gender = strtrim(parts(2));

    % Look ONLY inside the chosen subset
    spkPath = fullfile(librispeechRoot, spkID);
    if ~exist(spkPath,'dir')
        continue;
    end

    files = dir(fullfile(spkPath,"**","*.flac"));
    for f = 1:length(files)
        if gender == "M" && maleCount < MAX_MALE
            copyfile(fullfile(files(f).folder,files(f).name), outMale, 'f');
            maleCount = maleCount + 1;

        elseif gender ~= "M" && femaleCount < MAX_FEMALE
            copyfile(fullfile(files(f).folder,files(f).name), outFemale, 'f');
            femaleCount = femaleCount + 1;
        end

        if maleCount >= MAX_MALE && femaleCount >= MAX_FEMALE
            break;
        end
    end

    % Update progress bar
    elapsed = toc;
    totalDone = maleCount + femaleCount;
    waitbar(totalDone / (MAX_MALE + MAX_FEMALE), h, ...
        sprintf('Male: %d / %d | Female: %d / %d | Time: %.1f s', ...
        maleCount, MAX_MALE, femaleCount, MAX_FEMALE, elapsed));
end

close(h);

fprintf('DONE.\nMale files: %d\nFemale files: %d\n', maleCount, femaleCount); %[output:432b3099]


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":33.7}
%---
%[output:432b3099]
%   data: {"dataType":"text","outputData":{"text":"DONE.\nMale files: 500\nFemale files: 500\n","truncated":false}}
%---
