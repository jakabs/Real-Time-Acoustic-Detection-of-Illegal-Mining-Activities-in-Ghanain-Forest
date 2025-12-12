function [raw_id, raw_conf, raw_threat] = run_galamsey_inference(Audio_Frame)
    % 1. PERSISTENT MEMORY & CONFIGURATION
    persistent model last_alert_time
    
    % --- CONFIGURATION: SENSOR LOCATION (Kumasi, Ghana) ---
    SENSOR_LAT  = 6.6885; 
    SENSOR_LONG = -1.6242;
    % ------------------------------------------------------

    % Load Model (Only runs once)
    if isempty(model)
        temp = load('simulink_data_package.mat');
        model = temp.simulinkModel;
    end
    
    % Initialize Timer
    if isempty(last_alert_time)
        last_alert_time = datetime('now') - minutes(1);
    end

    % 2. INITIALIZE OUTPUTS (Fixes 'Unrecognized variable' error)
    raw_id = 1;         % Default: Background
    raw_conf = 0.0;
    raw_threat = 0;     % Default: Safe

    % 3. NOISE GATE (Skip silence)
    if rms(Audio_Frame) < 0.002
        return; 
    end

    % 4. FEATURE EXTRACTION
    fs = model.config.fs_target;

    % -- Primary (Mel) --
    mel = melSpectrogram(Audio_Frame, fs, ...
        'Window', hann(model.config.win_len_samples,'periodic'), ...
        'OverlapLength', model.config.win_len_samples - model.config.hop_len_samples, ...
        'FFTLength', model.config.fft_len, ...
        'NumBands', model.config.num_mel_bands, ...
        'FrequencyRange', model.config.freq_range, ...
        'SpectrumType', 'power');
    mel = log10(mel + 1e-6);
    mel = (mel - min(mel(:))) / (max(mel(:)) - min(mel(:)) + 1e-8);
    
    % Resize Mel
    mel_resized = zeros(64, 128);
    [r, c] = size(mel);
    mel_resized(1:min(r,64), 1:min(c,128)) = mel(1:min(r,64), 1:min(c,128));
    mel_final = (mel_resized - model.stats_prim.mu) / (model.stats_prim.sigma + eps);
    mel_final = reshape(mel_final, [64, 128, 1, 1]);

    % -- Secondary (MFCC) --
    [coeffs, ~] = mfcc(Audio_Frame, fs, ...
        'NumCoeffs', model.config.mfcc_coeffs, ...
        'FFTLength', model.config.fft_len, ...
        'Window', hann(model.config.win_len_samples));
    coeffs = coeffs'; 
    coeffs = (coeffs - min(coeffs(:))) / (max(coeffs(:)) - min(coeffs(:)) + 1e-8);
    
    % Resize MFCC
    mfcc_resized = zeros(13, 128);
    [r, c] = size(coeffs);
    mfcc_resized(1:min(r,13), 1:min(c,128)) = coeffs(1:min(r,13), 1:min(c,128));
    mfcc_final = (mfcc_resized - model.stats_sec.mu) / (model.stats_sec.sigma + eps);
    mfcc_final = reshape(mfcc_final, [13, 128, 1, 1]);

    % 5. INFERENCE (Ensemble)
    probs_prim = predict(model.net_prim, mel_final);
    probs_sec  = predict(model.net_sec, mfcc_final);
    w = model.weights;
    probs_final = (w(1) * probs_prim) + (w(2) * probs_sec);
    [conf, idx] = max(probs_final);

    % 6. PHYSICS LOGIC CHECK (Bass vs Treble)
    if idx > 1
        N = 1024; Y = fft(Audio_Frame, N); P = abs(Y(1:N/2+1)).^2;
        f = fs*(0:(N/2))/N;
        bass_pow = sum(P(f>=50 & f<=600)); 
        treble_pow = sum(P(f>=1500 & f<=8000));
        ratio = bass_pow / (treble_pow + eps);
        if idx == 2 && ratio > 1.5, idx = 3; elseif idx == 3 && ratio < 0.5, idx = 2; end
    end

    % 7. ASSIGN FINAL RESULTS
    raw_id = double(idx);
    raw_conf = double(conf);
    if idx > 1
        raw_threat = 1; 
    else
        raw_threat = 0; 
    end
    
    % 8. TELEGRAM ALERT LOGIC (Runs AFTER raw_threat is calculated)
    if raw_threat == 1
        % Check cool-down (30 seconds)
        time_since_last = seconds(datetime('now') - last_alert_time);
        
        if time_since_last > 30
            % Name the threat
            if raw_id == 2
                threat_name = 'CHAINSAW';
            else
                threat_name = 'ENGINE';
            end
            
            % Construct Message
            msg = sprintf('⚠️ THREAT DETECTED: %s\nConfidence: %.1f%%', ...
                          threat_name, raw_conf*100);
            
            % Send Alert with GPS Coordinates
            % Note: Make sure you saved the updated send_telegram_alert.m as well!
            send_telegram_alert(msg, SENSOR_LAT, SENSOR_LONG);
            
            % Reset Timer
            last_alert_time = datetime('now');
        end
    end
end