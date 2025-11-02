// Multi-Agent RAG System - Custom JavaScript

document.addEventListener('DOMContentLoaded', function() {
  
  // Agent Status Tracking
  const agents = {
    'Orchestrator': { active: false, lastActive: null, duration: 0 },
    'RAG Agent': { active: false, lastActive: null, duration: 0 },
    'Ingest Agent': { active: false, lastActive: null, duration: 0 },
    'Business Analysis Agent': { active: false, lastActive: null, duration: 0 },
    'Guardrail Agent': { active: false, lastActive: null, duration: 0 }
  };
  
  let agentStatusInterval = null;
  
  // Format time duration
  function formatDuration(ms) {
    if (!ms) return '00:00:00';
    
    const seconds = Math.floor(ms / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  
  // Update agent status
  function updateAgentStatus(agentName, isActive) {
    if (agents[agentName] === undefined) {
      agents[agentName] = { active: false, lastActive: null, duration: 0 };
    }
    
    const agent = agents[agentName];
    
    if (isActive && !agent.active) {
      // Agent just became active
      agent.active = true;
      agent.lastActive = Date.now();
      agent.duration = 0;
    } else if (!isActive && agent.active) {
      // Agent just became inactive
      if (agent.lastActive) {
        agent.duration += Date.now() - agent.lastActive;
      }
      agent.active = false;
      agent.lastActive = null;
    } else if (isActive && agent.active && agent.lastActive) {
      // Agent still active, update duration
      agent.duration += Date.now() - agent.lastActive;
      agent.lastActive = Date.now();
    }
    
    renderAgentStatusBar();
  }
  
  // Render agent status bar
  function renderAgentStatusBar() {
    let statusBar = document.querySelector('.cl-agent-status-bar');
    
    if (!statusBar) {
      statusBar = document.createElement('div');
      statusBar.className = 'cl-agent-status-bar';
      
      const header = document.querySelector('.cl-header');
      if (header && header.nextSibling) {
        header.parentNode.insertBefore(statusBar, header.nextSibling);
      } else if (header) {
        header.parentNode.appendChild(statusBar);
      } else {
        document.body.insertBefore(statusBar, document.body.firstChild);
      }
    }
    
    statusBar.innerHTML = '';
    
    Object.entries(agents).forEach(([name, agent]) => {
      const statusItem = document.createElement('div');
      statusItem.className = 'cl-agent-status-item';
      
      const indicator = document.createElement('span');
      indicator.className = `cl-agent-indicator ${agent.active ? 'active' : 'inactive'}`;
      
      const nameSpan = document.createElement('span');
      nameSpan.className = 'cl-agent-name';
      nameSpan.textContent = name;
      
      const timeSpan = document.createElement('span');
      timeSpan.className = 'cl-agent-time';
      
      if (agent.active) {
        const currentDuration = agent.duration + (Date.now() - (agent.lastActive || Date.now()));
        timeSpan.textContent = formatDuration(currentDuration);
      } else {
        timeSpan.textContent = agent.duration > 0 ? formatDuration(agent.duration) : '00:00:00';
      }
      
      statusItem.appendChild(indicator);
      statusItem.appendChild(nameSpan);
      statusItem.appendChild(timeSpan);
      statusBar.appendChild(statusItem);
    });
  }
  
  // Detect agent activity from messages
  function detectAgentActivity() {
    const messages = document.querySelectorAll('[data-author]');
    
    messages.forEach(message => {
      const author = message.getAttribute('data-author') || '';
      const messageTime = message.querySelector('.cl-message-time');
      
      // Check if message is recent (within last 5 seconds)
      const isRecent = messageTime && (Date.now() - new Date(messageTime.textContent).getTime()) < 5000;
      
      // Update agent status based on author
      Object.keys(agents).forEach(agentName => {
        if (author.includes(agentName) || author === agentName) {
          updateAgentStatus(agentName, isRecent);
        }
      });
    });
  }
  
  // Watch for new messages
  const messageObserver = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      mutation.addedNodes.forEach(function(node) {
        if (node.nodeType === 1) {
          // Check if it's a message element
          const message = node.classList && node.classList.contains('cl-message') 
            ? node 
            : node.querySelector && node.querySelector('.cl-message');
          
          if (message) {
            const author = message.querySelector('[data-author]')?.getAttribute('data-author') || 
                          message.textContent.match(/(?:Orchestrator|RAG Agent|Ingest Agent|Business Analysis Agent|Guardrail Agent)/)?.[0];
            
            if (author) {
              Object.keys(agents).forEach(agentName => {
                if (author.includes(agentName)) {
                  updateAgentStatus(agentName, true);
                  
                  // Mark as inactive after 10 seconds
                  setTimeout(() => {
                    updateAgentStatus(agentName, false);
                  }, 10000);
                }
              });
            }
            
            // Style messages
            styleMessages();
          }
        }
      });
    });
    
    detectAgentActivity();
  });
  
  // Style messages
  function styleMessages() {
    const messages = document.querySelectorAll('.cl-message');
    
    messages.forEach(message => {
      // Skip if already styled
      if (message.querySelector('.cl-message-avatar')) {
        return;
      }
      
      const author = message.querySelector('[data-author]')?.getAttribute('data-author') || '';
      const content = message.querySelector('.cl-message-content');
      
      if (!content) return;
      
      // Determine if user or assistant
      const isUser = author.toLowerCase().includes('user') || message.classList.contains('cl-message-user');
      
      if (isUser) {
        message.classList.add('cl-message-user');
      } else {
        message.classList.add('cl-message-assistant');
      }
      
      // Create avatar
      const avatar = document.createElement('div');
      avatar.className = 'cl-message-avatar';
      avatar.textContent = isUser ? '👤' : '🤖';
      
      // Wrap content
      const contentWrapper = document.createElement('div');
      contentWrapper.className = 'cl-message-content-wrapper';
      
      const meta = document.createElement('div');
      meta.className = 'cl-message-meta';
      
      if (author) {
        const authorSpan = document.createElement('span');
        authorSpan.className = 'cl-message-author';
        authorSpan.textContent = author;
        meta.appendChild(authorSpan);
      }
      
      const timeSpan = document.createElement('span');
      timeSpan.className = 'cl-message-time';
      timeSpan.textContent = new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      meta.appendChild(timeSpan);
      
      contentWrapper.appendChild(content.cloneNode(true));
      contentWrapper.appendChild(meta);
      
      content.remove();
      message.insertBefore(avatar, message.firstChild);
      message.appendChild(contentWrapper);
    });
  }
  
  // Update agent status every second
  function startAgentStatusTracking() {
    if (agentStatusInterval) {
      clearInterval(agentStatusInterval);
    }
    
    agentStatusInterval = setInterval(() => {
      renderAgentStatusBar();
    }, 1000);
  }
  
  // Initialize
  function init() {
    renderAgentStatusBar();
    styleMessages();
    detectAgentActivity();
    startAgentStatusTracking();
    
    // Observe DOM for new messages
    messageObserver.observe(document.body, {
      childList: true,
      subtree: true
    });
    
    // Watch for steps (tool calls)
    const stepObserver = new MutationObserver(() => {
      const steps = document.querySelectorAll('.cl-step');
      steps.forEach(step => {
        const stepName = step.querySelector('.cl-step-name')?.textContent || '';
        Object.keys(agents).forEach(agentName => {
          if (stepName.includes(agentName)) {
            updateAgentStatus(agentName, true);
            setTimeout(() => {
              updateAgentStatus(agentName, false);
            }, 5000);
          }
        });
      });
    });
    
    stepObserver.observe(document.body, {
      childList: true,
      subtree: true
    });
  }
  
  // Start initialization
  setTimeout(init, 500);
  
  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    if (agentStatusInterval) {
      clearInterval(agentStatusInterval);
    }
    messageObserver.disconnect();
  });
});
