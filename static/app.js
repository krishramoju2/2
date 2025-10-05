// app.js - client-side chat wiring for Flask API backend
(function(){
  const form = document.getElementById('chat-form');
  const input = document.getElementById('input');
  const messages = document.getElementById('messages');
  const sendBtn = document.getElementById('send');
  const typing = document.getElementById('typing');

  // Append message to chat window
  function appendMessage(text, isBot, meta){
    const wrap = document.createElement('div');
    wrap.className = 'message ' + (isBot ? 'bot' : 'user');

    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + (isBot ? 'bot' : 'user');
    bubble.textContent = text;

    wrap.appendChild(bubble);
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;

    // Optional: log intent & confidence for debugging
    if(meta){
      console.log('Intent:', meta.intent, 'Confidence:', meta.confidence);
    }
  }

  // Show or hide typing indicator
  function showTyping(show){
    typing.style.display = show ? 'block' : 'none';
    typing.setAttribute('aria-hidden', (!show).toString());
  }

  // Send user message to backend
  async function sendMessage(){
    const value = input.value.trim();
    if(!value) return;

    appendMessage(value, false); // show user message
    input.value = '';
    input.focus();
    showTyping(true);

    try{
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({message: value})
      });

      if(!resp.ok){
        appendMessage(`Server error (${resp.status})`, true, {intent:'error', confidence:0});
        return;
      }

      const data = await resp.json();
      const botText = data && data.response ? data.response : "I'm sorry, I couldn't process that.";
      appendMessage(botText, true, {intent: data.intent || 'unknown', confidence: data.confidence || 0});

    } catch(err){
      appendMessage('Network error: failed to reach the server.', true, {intent:'error', confidence:0});
      console.error('Chat send error:', err);
    } finally{
      showTyping(false);
    }
  }

  // Button click handler
  sendBtn.addEventListener('click', (e)=>{
    e.preventDefault();
    sendMessage();
  });

  // Enter key handler
  input.addEventListener('keydown', (e)=>{
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      sendMessage();
    }
  });

  // Welcome message on load
  document.addEventListener('DOMContentLoaded', ()=>{
    if(messages.children.length === 0){
      appendMessage(
        "Hi, I'm the Vibeathon bot! You can ask me about joining or registration, event dates and costs, collaborations, social media, GitHub, contact info, current projects, or even provide feedback.",
        true,
        {intent:'greeting', confidence:1}
      );
    }
  });

})();
