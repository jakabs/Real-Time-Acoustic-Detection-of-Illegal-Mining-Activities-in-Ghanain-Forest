
   function send_telegram_alert(message, lat, long)
    % SEND_TELEGRAM_ALERT - Sends text + GPS Location link
    
    % --- CONFIGURATION ---
  bot_token = '8222074924:AAEl11hMnb7Olz3UxKUlCMuhsD9Jvf21prQ';
    chat_id   = '1195638533'; 
    % ---------------------

    try
        % 1. Create Google Maps Link
        % Format: https://www.google.com/maps/search/?api=1&query=LAT,LONG
        map_link = sprintf('https://www.google.com/maps/search/?api=1&query=%.6f,%.6f', lat, long);
        
        % 2. Append Link to Message
        full_msg = sprintf('%s\n\n📍 LOCATION:\n%s', message, map_link);
        
        % 3. Send
        api_url = sprintf('https://api.telegram.org/bot%s/sendMessage', bot_token);
        options = weboptions('MediaType', 'application/json', 'Timeout', 5);
        data = struct('chat_id', chat_id, 'text', full_msg);
        
        webwrite(api_url, data, options);
        fprintf('[TELEGRAM] Alert sent with coordinates.\n');
        
    catch ME
        fprintf('[TELEGRAM ERROR] %s\n', ME.message);
    end
end