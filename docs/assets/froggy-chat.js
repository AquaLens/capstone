/**
 * Froggy Chat Widget
 * Standalone chat functionality for AquaLens
 */

class FroggyChat {
    constructor() {
      this.USER_MESSAGE_CLASS = 'froggy-message froggy-user';
      this.AI_MESSAGE_CLASS = 'froggy-message froggy-froggy';
      this.API_ENDPOINT = 'https://aqualens-froggy-backend.hf.space/api/ask';
      this.RESET_ENDPOINT = 'https://aqualens-froggy-backend.hf.space/api/reset_conversation';
      
      // Initialize conversation tracking
      this.conversationContext = {
        sessionIds: [],
        messages: []
      };
      
      // Initialize session ID from localStorage if available
      this.sessionId = localStorage.getItem('froggy_session_id') || null;
      
      // If we have a session ID, add it to our tracking
      if (this.sessionId) {
        this.conversationContext.sessionIds.push(this.sessionId);
      }
      
      this.hasAgreedToTerms = localStorage.getItem('froggy_terms_agreed') === 'true';
      
      this.init();
    }
    
    init() {
      document.addEventListener('DOMContentLoaded', () => {
        this.setupElements();
        this.setupEventListeners();
        this.setupResetButton();
        
        // Check if conversation is empty and add welcome message if needed
        if (this.messages && this.messages.children.length === 0) {
          this.addMessageToConversation('Hello! I\'m Froggy, your guide to water quality initiatives and research. How can I help you today?', 'ai');
        }
      });
    }
    
    setupElements() {
      this.froggyBtn = document.getElementById('froggy-button');
      this.froggyPopup = document.getElementById('froggy-popup');
      this.froggyClose = document.getElementById('froggy-close');
      this.sendBtn = document.getElementById('froggy-send');
      this.textarea = document.getElementById('froggy-question');
      this.messages = document.getElementById('froggy-messages');
      this.disclaimerModal = document.getElementById('froggy-disclaimer-modal');
      this.agreeBtn = document.getElementById('froggy-agree');
      this.declineBtn = document.getElementById('froggy-decline');
    }
    
    setupEventListeners() {
      if (this.froggyBtn) {
        this.froggyBtn.addEventListener('click', () => {
          if (!this.hasAgreedToTerms) {
            this.showDisclaimer();
          } else {
            this.openChat();
          }
        });
      }
      
      if (this.sendBtn) {
        this.sendBtn.addEventListener('click', () => this.handleSendMessage());
      }
      
      if (this.textarea) {
        this.textarea.addEventListener('keypress', (e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.handleSendMessage();
          }
        });
      }
      
      // Disclaimer modal events
      if (this.agreeBtn) {
        this.agreeBtn.addEventListener('click', () => {
          this.agreeToTerms();
        });
      }
      
      if (this.declineBtn) {
        this.declineBtn.addEventListener('click', () => {
          this.declineTerms();
        });
      }
      
      // Close disclaimer on backdrop click
      if (this.disclaimerModal) {
        this.disclaimerModal.addEventListener('click', (e) => {
          if (e.target === this.disclaimerModal) {
            this.declineTerms();
          }
        });
      }
    }
    
    setupResetButton() {
      const header = document.querySelector('.froggy-chat-header');
      if (header && this.froggyClose) {
        // Create reset button
        const resetBtn = document.createElement('button');
        resetBtn.id = 'froggy-reset';
        resetBtn.className = 'froggy-reset-button';
        resetBtn.textContent = 'New Chat';
        resetBtn.addEventListener('click', () => this.resetConversation());
        
        // Create container for header buttons
        const headerButtons = document.createElement('div');
        headerButtons.className = 'froggy-header-buttons';
        
        // Clone the close button (we'll remove the original)
        const newCloseBtn = this.froggyClose.cloneNode(true);
        newCloseBtn.addEventListener('click', () => {
          this.froggyPopup.style.display = 'none';
        });
        
        // Add buttons to container
        headerButtons.appendChild(resetBtn);
        headerButtons.appendChild(newCloseBtn);
        
        // Remove the original close button if it exists
        if (this.froggyClose.parentNode) {
          this.froggyClose.parentNode.removeChild(this.froggyClose);
        }
        
        // Add container to header
        header.appendChild(headerButtons);
      }
    }
    
    showDisclaimer() {
      if (this.disclaimerModal) {
        this.disclaimerModal.style.display = 'flex';
      }
    }
    
    agreeToTerms() {
      this.hasAgreedToTerms = true;
      localStorage.setItem('froggy_terms_agreed', 'true');
      if (this.disclaimerModal) {
        this.disclaimerModal.style.display = 'none';
      }
      this.openChat();
    }
    
    declineTerms() {
      if (this.disclaimerModal) {
        this.disclaimerModal.style.display = 'none';
      }
    }
    
    openChat() {
      if (this.froggyPopup) {
        this.froggyPopup.style.display = 'flex';
      }
    }
    
    async handleSendMessage() {
      const text = this.textarea.value.trim();
      if (!text) return;
      
      // Clear input field
      this.textarea.value = '';
      
      // Add user message to conversation
      this.addMessageToConversation(text, 'user');
      
      // Store message in context
      this.conversationContext.messages.push({
        role: 'user',
        content: text,
        timestamp: new Date().toISOString()
      });
      
      // Add typing indicator
      const typingMsgId = 'typing-msg-' + Date.now();
      this.addTypingIndicator(typingMsgId);
      
      // Scroll to bottom of conversation
      this.messages.scrollTop = this.messages.scrollHeight;
      
      try {
        // Send request to backend with all known session IDs
        const response = await fetch(this.API_ENDPOINT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          mode: 'cors',
          credentials: 'omit',
          body: JSON.stringify({
            question: text,
            session_id: this.sessionId,
            previous_sessions: this.conversationContext.sessionIds,
            message_history: this.conversationContext.messages.slice(-5)
          })
        });
        
        // Remove typing indicator
        const typingMsg = document.getElementById(typingMsgId);
        if (typingMsg) {
          this.messages.removeChild(typingMsg);
        }
        
        if (!response.ok) {
          throw new Error(`API request failed with status ${response.status}`);
        }
        
        const data = await response.json();
        
        // Save session ID if provided
        if (data.session_id) {
          // Add to session ID history
          this.conversationContext.sessionIds.push(data.session_id);
          
          // Update current session ID
          this.sessionId = data.session_id;
          localStorage.setItem('froggy_session_id', data.session_id);
        } else if (!this.sessionId && data.id) {
          // Fallback if session_id is not provided but id is
          this.conversationContext.sessionIds.push(data.id);
          this.sessionId = data.id;
          localStorage.setItem('froggy_session_id', data.id);
        }
        
        const froggyResponse = data.answer || 'Sorry, I couldn\'t find an answer to that question.';
        
        // Add AI response to conversation
        this.addMessageToConversation(froggyResponse, 'ai');
        
        // Store AI response in context
        this.conversationContext.messages.push({
          role: 'assistant',
          content: froggyResponse,
          timestamp: new Date().toISOString()
        });
        
        // Scroll to bottom of conversation
        this.messages.scrollTop = this.messages.scrollHeight;
        
      } catch (error) {
        console.error('Error details:', error);
        
        // Remove typing indicator if it still exists
        const typingMsg = document.getElementById(typingMsgId);
        if (typingMsg) {
          this.messages.removeChild(typingMsg);
        }
        
        this.addMessageToConversation('Sorry, it seems that I might be having some connection issues. Please try again later.', 'ai');
        this.messages.scrollTop = this.messages.scrollHeight;
      }
    }
    
    addMessageToConversation(message, sender) {
      const messageElement = document.createElement('div');
      messageElement.className = sender === 'user' ? this.USER_MESSAGE_CLASS : this.AI_MESSAGE_CLASS;
      
      // Add message content with markdown parsing for AI responses
      if (sender === 'ai') {
        // Parse markdown (basic implementation)
        let formattedMessage = message
          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
          .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
          .replace(/\n\n/g, '<br><br>') // Paragraphs
          .replace(/\n/g, '<br>'); // Line breaks
        
        messageElement.innerHTML = formattedMessage;
      } else {
        messageElement.textContent = message;
      }
      
      // Add timestamp
      const timestampElement = document.createElement('div');
      timestampElement.className = 'froggy-timestamp';
      const now = new Date();
      timestampElement.textContent = now.toLocaleTimeString();
      messageElement.appendChild(timestampElement);
      
      // Add to container
      this.messages.appendChild(messageElement);
    }
    
    addTypingIndicator(id) {
      const msg = document.createElement('div');
      msg.id = id;
      msg.className = 'froggy-message froggy-froggy froggy-typing';
      msg.innerHTML = '<span>Thinking</span><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>';
      this.messages.appendChild(msg);
    }
    
    resetConversation() {
      // Clear conversation display
      if (this.messages) {
        this.messages.innerHTML = '';
      }
      
      // Reset conversation context
      this.conversationContext = {
        sessionIds: [],
        messages: []
      };
      
      // Clear session ID
      this.sessionId = null;
      localStorage.removeItem('froggy_session_id');
      
      // API call to reset conversation on backend
      fetch(this.RESET_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify({})
      })
      .then(response => response.json())
      .then(data => {
        console.log('Conversation reset:', data);
        
        // Add welcome message
        const welcomeMessage = 'Hello! I\'m Froggy, your guide to water quality initiatives and research. How can I help you today?';
        this.addMessageToConversation(welcomeMessage, 'ai');
        
        // Store welcome message in context
        this.conversationContext.messages.push({
          role: 'assistant',
          content: welcomeMessage,
          timestamp: new Date().toISOString()
        });
        
        // If reset gives us a new session ID, store it
        if (data && data.session_id) {
          this.sessionId = data.session_id;
          this.conversationContext.sessionIds.push(data.session_id);
          localStorage.setItem('froggy_session_id', data.session_id);
        }
      })
      .catch(error => {
        console.error('Error resetting conversation:', error);
        // Add welcome message anyway
        const welcomeMessage = 'Hello! I\'m Froggy, your guide to water quality initiatives and research. How can I help you today?';
        this.addMessageToConversation(welcomeMessage, 'ai');
        
        // Store welcome message in context
        this.conversationContext.messages.push({
          role: 'assistant',
          content: welcomeMessage,
          timestamp: new Date().toISOString()
        });
      });
    }
  }
  
  // Initialize Froggy Chat when the script loads
  new FroggyChat();