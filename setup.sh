sh setupworktrees.sh
cd live_trading
mkdir config
printf "APCA_API_KEY_ID = 'your-api-key'\nAPCA_API_SECRET_KEY = 'your-secret-key'\n\nAPCA_API_BASE_URL = 'https://paper-api.alpaca.markets'\nEMAIL_ADDRESS = 'your-email-address'\nEMAIL_PASSWORD = 'your-email-password" >> config/config.py
cd ..