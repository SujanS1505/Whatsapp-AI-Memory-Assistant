// FILE: whatsapp-bot/src/whatsappClient.js

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const { handleGroupMessage } = require('./messageHandler');
const config = require('./config');

const puppeteerConfig = {
  args: config.whatsapp.puppeteerArgs,
  headless: true,
  protocolTimeout: 60000,
};
if (config.whatsapp.executablePath) {
  puppeteerConfig.executablePath = config.whatsapp.executablePath;
}

const client = new Client({
  authStrategy: new LocalAuth({
    clientId: config.whatsapp.sessionName,
  }),
  puppeteer: puppeteerConfig,
  // Cache the WhatsApp Web bundle locally — prevents version-mismatch navigations
  // that destroy the Puppeteer execution context on startup.
  webVersionCache: {
    type: 'local',
  },
});

// QR Code generation — display in terminal for scanning
client.on('qr', (qr) => {
  console.log('\n[WhatsApp] Scan the QR code below with your WhatsApp app:\n');
  qrcode.generate(qr, { small: true });
});

// Authentication success
client.on('authenticated', () => {
  console.log('[WhatsApp] Authenticated successfully.');
});

// Authentication failure
client.on('auth_failure', (msg) => {
  console.error(`[WhatsApp] Authentication failure: ${msg}`);
  process.exit(1);
});

// Client ready
client.on('ready', () => {
  reconnectAttempt = 0; // reset backoff counter on successful connection
  console.log('[WhatsApp] Client is ready and connected!');
  console.log(`[WhatsApp] Listening for messages with prefix "${config.assistant.triggerPrefix}"`);
});

// Handle incoming messages (message_create fires for ALL messages including
// ones you send from your own phone, which is needed when the bot runs on
// a personal account rather than a dedicated number)
client.on('message_create', async (msg) => {
  // Resolve the chat first. WhatsApp Channels (broadcast channels) and other
  // unsupported chat types cause whatsapp-web.js to throw inside Channel._patch.
  // Catching here lets us skip those silently without polluting the logs.
  let chat;
  try {
    chat = await msg.getChat();
  } catch {
    // Non-group or unsupported chat type (e.g. WhatsApp Channel) — skip.
    return;
  }

  try {
    // Only handle group messages
    if (!chat.isGroup) return;

    // Skip the bot's own AI replies to prevent infinite loops
    const body = (msg.body || '').trim();
    if (msg.fromMe && body.startsWith('*AI Assistant:*')) return;

    // Pass the already-resolved chat so handleGroupMessage doesn't need
    // to call getChat() a second time (avoids extra Puppeteer round-trips).
    await handleGroupMessage(msg, chat, client);
  } catch (err) {
    console.error(`[WhatsApp] Unhandled error in message event: ${err.message}`, err.stack);
  }
});

// Disconnected
client.on('disconnected', (reason) => {
  console.warn(`[WhatsApp] Client disconnected: ${reason}`);
  scheduleReconnect();
});

// Retry initialize with exponential backoff.
// "Execution context was destroyed" is a transient Puppeteer error that
// almost always resolves on the next attempt.
let reconnectAttempt = 0;
function scheduleReconnect() {
  const delayMs = Math.min(1000 * 2 ** reconnectAttempt, 30000); // cap at 30 s
  reconnectAttempt++;
  console.log(`[WhatsApp] Reconnecting in ${delayMs / 1000}s (attempt ${reconnectAttempt})...`);
  setTimeout(() => {
    client.initialize().catch((err) => {
      console.error(`[WhatsApp] Reconnect attempt ${reconnectAttempt} failed: ${err.message}`);
      scheduleReconnect();
    });
  }, delayMs);
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n[WhatsApp] Shutting down gracefully...');
  await client.destroy();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n[WhatsApp] Received SIGTERM. Shutting down...');
  await client.destroy();
  process.exit(0);
});

// Initialize (with retry on transient Puppeteer errors)
console.log('[WhatsApp] Initializing client...');
client.initialize().catch((err) => {
  console.error('[WhatsApp] Initialization error:', err.message);
  scheduleReconnect();
});

module.exports = client;
