clc;
clear;

%% ===============================
% Task 2 – Reverberant (FINAL SUBMISSION)
% ===============================

% Always run from this script's folder
SCRIPT_DIR = fileparts(mfilename('fullpath'));
cd(SCRIPT_DIR);

%% Resolve project root
PROJECT_ROOT = fullfile(SCRIPT_DIR, '..', '..');

%% Python + model paths
pythonExe = "C:\Users\aweso\Documents\SPcup_final\venv311\Scripts\python.exe";
assert(exist(pythonExe,'file')==2, 'Python venv not found');


inferenceScript = fullfile(PROJECT_ROOT, ...
    'Resources', 'Model', 'inference_Conformer.py');

modelPath = fullfile(PROJECT_ROOT, ...
    'Resources', 'Model', 'reverb_Conformer.pth');

%% Required local files
assert(exist('mixture.wav','file')==2, 'mixture.wav missing');
assert(exist('target_signal.wav','file')==2, 'target_signal.wav missing');
assert(exist('interference_signal1.wav','file')==2, 'interference_signal1.wav missing');
assert(exist('rir.wav','file')==2, 'rir.wav missing');
assert(exist('meta.json','file')==2, 'meta.json missing');

%% Load metadata
meta = jsondecode(fileread('meta.json'));
targetAngle = meta.target_angle;

%% ===============================
% Run Python inference
% ===============================
outputWav = fullfile(SCRIPT_DIR, 'processed_signal.wav');

cmd = sprintf([ ...
    '"%s" "%s" ' ...
    '--input "%s" ' ...
    '--angle %.2f ' ...
    '--output "%s" ' ...
    '--model "%s" ' ...
    '--device cpu'], ...
    pythonExe, inferenceScript, ...
    fullfile(SCRIPT_DIR,'mixture.wav'), ...
    targetAngle, outputWav, modelPath);

disp('Running inference command:');
disp(cmd);

status = system(cmd);
assert(status == 0, 'Python inference failed');

%% ===============================
% Load audio signals
% ===============================
[target_signal, fs] = audioread('target_signal.wav');
[interference_signal, fs2] = audioread('interference_signal1.wav');
[mixture_signal, fs3] = audioread('mixture.wav');
[rir_data, fs4] = audioread('rir.wav');
[processed_signal, fs5] = audioread('processed_signal.wav');

assert(all([fs fs2 fs3 fs4 fs5] == fs), 'Sampling rate mismatch');

%% Trim all signals to same length
minLen = min([
    length(target_signal), ...
    length(interference_signal), ...
    length(mixture_signal), ...
    length(processed_signal)]);

target_signal = target_signal(1:minLen);
interference_signal = interference_signal(1:minLen);
mixture_signal = mixture_signal(1:minLen);
processed_signal = processed_signal(1:minLen);

%% ===============================
% Metrics (placeholders)
% ===============================
metrics.OSINR = NaN;
metrics.PESQ  = NaN;
metrics.STOI  = NaN;

%% ===============================
% Params
% ===============================
params.fs = fs;
params.SNR_dB = 5;
params.SIR_dB = 0;
params.environment = "reverberant";
params.RT60 = 0.5;
params.target_angle = targetAngle;

%% ===============================
% Save MAT file (FINAL OUTPUT)
% ===============================
save('Task2_Reverberant_5dB.mat', ...
    'target_signal', ...
    'interference_signal', ...
    'mixture_signal', ...
    'rir_data', ...
    'processed_signal', ...
    'metrics', ...
    'params');

disp('✅ Task2_Reverberant_5dB.mat created successfully');
