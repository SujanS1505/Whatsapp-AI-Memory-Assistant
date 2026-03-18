// FILE: whatsapp-bot/src/messageHandler.js

const { sendMessage, sendQuery } = require('./apiClient');
const config = require('./config');

const TRIGGER = config.assistant.triggerPrefix.toLowerCase();

/**
 * Determines whether a raw message text is directed at the assistant.
 * @param {string} text
 * @returns {boolean}
 */
function isAssistantQuery(text) {
  return text.trim().toLowerCase().startsWith(TRIGGER);
}

/**
 * Extracts the question part after the trigger prefix.
 * @param {string} text
 * @returns {string}
 */
function extractQuestion(text) {
  return text.trim().slice(TRIGGER.length).trim();
}

/**
 * Main handler called for every incoming group message.
 * @param {import('whatsapp-web.js').Message} msg  - whatsapp-web.js message object
 * @param {import('whatsapp-web.js').GroupChat} chat - already-resolved chat (passed from caller)
 * @param {import('whatsapp-web.js').Client}  client - whatsapp-web.js client
 */
async function handleGroupMessage(msg, chat, client) {
  const body = msg.body || '';
  // chat is pre-fetched by the caller — no extra Puppeteer call needed
  
  // Safely fetch contact with fallback — whatsapp-web.js sometimes returns incomplete contact data
  let sender = msg.from; // default fallback
  try {
    const contact = await msg.getContact();
    if (contact && (contact.pushname || contact.number)) {
      sender = contact.pushname || contact.number;
    }
  } catch (contactErr) {
    // If getContact fails, use the from field (always available)
    console.warn(`[messageHandler] Could not fetch contact details: ${contactErr.message}`);
  }

  const groupId = chat.id._serialized;
  const timestamp = new Date(msg.timestamp * 1000).toISOString();

  // Always store the message (even if it's a query)
  try {
    await sendMessage({
      group_id: groupId,
      sender,
      message: body,
      timestamp,
    });
    console.log(`[messageHandler] Stored message from ${sender} in group ${groupId}`);
  } catch (err) {
    console.error(`[messageHandler] Failed to store message: ${err.message}`);
  }

  // If message is a query, retrieve AI response and reply
  if (isAssistantQuery(body)) {
    const question = extractQuestion(body);
    if (!question) {
      await chat.sendMessage(`Please provide a question after ${config.assistant.triggerPrefix}.`);
      return;
    }

    console.log(`[messageHandler] Query from ${sender}: "${question}"`);

    try {
      const result = await sendQuery({ group_id: groupId, question });
      const answer = result?.answer || 'I could not generate a response.';
      await chat.sendMessage(`*AI Assistant:*\n\n${answer}`);
    } catch (err) {
      console.error(`[messageHandler] Query failed: ${err.message}`);
      await chat.sendMessage('Sorry, I encountered an error while processing your query. Please try again.');
    }
  }
}

module.exports = { handleGroupMessage };
