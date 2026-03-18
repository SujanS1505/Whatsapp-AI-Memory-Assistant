// FILE: whatsapp-bot/src/config.js

require('dotenv').config();

const config = {
  whatsapp: {
    sessionName: process.env.WHATSAPP_SESSION || 'whatsapp-ai-session',
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
    puppeteerArgs: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--disable-software-rasterizer',
      '--disable-extensions',
      '--no-first-run',
      '--disable-accelerated-2d-canvas',
      '--disable-features=IsolateOrigins,site-per-process',
    ],
  },
  backend: {
    apiUrl: process.env.BACKEND_API_URL || 'http://localhost:8000',
    timeout: parseInt(process.env.API_TIMEOUT, 10) || 30000,
  },
  assistant: {
    triggerPrefix: '@assistant',
  },
};

module.exports = config;
