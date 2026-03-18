// FILE: whatsapp-bot/src/apiClient.js

const axios = require('axios');
const config = require('./config');

const httpClient = axios.create({
  baseURL: config.backend.apiUrl,
  timeout: config.backend.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Retry logic for transient failures
async function withRetry(fn, retries = 3, delayMs = 1000) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const isLast = attempt === retries;
      if (isLast) throw err;
      console.warn(`[apiClient] Attempt ${attempt} failed: ${err.message}. Retrying in ${delayMs}ms...`);
      await new Promise((res) => setTimeout(res, delayMs * attempt));
    }
  }
}

/**
 * Send an ingested group message to the backend.
 * @param {Object} payload - { group_id, sender, message, timestamp }
 */
async function sendMessage(payload) {
  return withRetry(async () => {
    const response = await httpClient.post('/messages', payload);
    return response.data;
  });
}

/**
 * Send a query to the backend and receive an AI-generated answer.
 * @param {Object} payload - { group_id, question }
 */
async function sendQuery(payload) {
  return withRetry(async () => {
    const response = await httpClient.post('/query', payload);
    return response.data;
  });
}

module.exports = { sendMessage, sendQuery };
