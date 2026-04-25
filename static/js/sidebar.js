/**
 * Sidebar Manager for AI Trading Bot
 * Handles sidebar state, chat history loading, and navigation
 * @module sidebar
 */

// State management
const SidebarState = {
    /** @type {boolean} */
    isOpen: true,
    /** @type {Array<Object>} */
    chatHistory: [],
    /** @type {number|null} */
    currentChatId: null,
    /** @type {boolean} */
    isLoading: false,
    /** @type {number} */
    page: 1,
    /** @type {number} */
    pageSize: 20,
    /** @type {boolean} */
    hasMore: true
};

/**
 * DOM element references
 * @type {Object}
 */
const elements = {
    sidebar: null,
    toggleBtn: null,
    chatList: null,
    newChatBtn: null,
    searchInput: null,
    loadingIndicator: null,
    emptyState: null,
    errorState: null
};

/**
 * Initialize the sidebar component
 * @param {Object} options - Configuration options
 * @param {string} options.sidebarId - Sidebar container ID
 * @param {string} options.toggleBtnId - Toggle button ID
 * @param {string} options.chatListId - Chat list container ID
 * @param {string} options.newChatBtnId - New chat button ID
 * @param {string} options.searchInputId - Search input ID
 * @param {string} options.loadingId - Loading indicator ID
 * @param {string} options.emptyStateId - Empty state element ID
 * @param {string} options.errorStateId - Error state element ID
 * @returns {Promise<boolean>} Success status
 */
export async function initializeSidebar(options = {}) {
    try {
        // Validate and cache DOM elements
        const defaults = {
            sidebarId: 'sidebar',
            toggleBtnId: 'sidebar-toggle',
            chatListId: 'chat-list',
            newChatBtnId: 'new-chat-btn',
            searchInputId: 'chat-search',
            loadingId: 'sidebar-loading',
            emptyStateId: 'sidebar-empty',
            errorStateId: 'sidebar-error'
        };

        const config = { ...defaults, ...options };

        // Cache DOM elements
        elements.sidebar = document.getElementById(config.sidebarId);
        elements.toggleBtn = document.getElementById(config.toggleBtnId);
        elements.chatList = document.getElementById(config.chatListId);
        elements.newChatBtn = document.getElementById(config.newChatBtnId);
        elements.searchInput = document.getElementById(config.searchInputId);
        elements.loadingIndicator = document.getElementById(config.loadingId);
        elements.emptyState = document.getElementById(config.emptyStateId);
        elements.errorState = document.getElementById(config.errorStateId);

        // Validate required elements
        if (!elements.sidebar || !elements.chatList) {
            throw new Error('Required sidebar elements not found');
        }

        // Restore sidebar state from localStorage
        restoreSidebarState();

        // Attach event listeners
        attachEventListeners();

        // Load initial chat history
        await loadChatHistory();

        // Set up infinite scroll
        setupInfiniteScroll();

        console.log('Sidebar initialized successfully');
        return true;
    } catch (error) {
        console.error('Failed to initialize sidebar:', error);
        showError('Failed to initialize sidebar');
        return false;
    }
}

/**
 * Attach event listeners to sidebar elements
 */
function attachEventListeners() {
    // Toggle sidebar visibility
    if (elements.toggleBtn) {
        elements.toggleBtn.addEventListener('click', toggleSidebar);
    }

    // New chat button
    if (elements.newChatBtn) {
        elements.newChatBtn.addEventListener('click', createNewChat);
    }

    // Search input with debounce
    if (elements.searchInput) {
        let searchTimeout;
        elements.searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                searchChats(e.target.value);
            }, 300);
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);

    // Window resize handler for responsive behavior
    window.addEventListener('resize', handleResponsiveResize);
}

/**
 * Toggle sidebar open/closed state
 */
function toggleSidebar() {
    SidebarState.isOpen = !SidebarState.isOpen;
    
    if (elements.sidebar) {
        elements.sidebar.classList.toggle('collapsed', !SidebarState.isOpen);
        elements.sidebar.setAttribute('aria-expanded', SidebarState.isOpen);
    }

    // Save state to localStorage
    localStorage.setItem('sidebar_open', SidebarState.isOpen);

    // Dispatch custom event for other components
    window.dispatchEvent(new CustomEvent('sidebar-toggle', {
        detail: { isOpen: SidebarState.isOpen }
    }));
}

/**
 * Restore sidebar state from localStorage
 */
function restoreSidebarState() {
    const savedState = localStorage.getItem('sidebar_open');
    if (savedState !== null) {
        SidebarState.isOpen = savedState === 'true';
        if (elements.sidebar) {
            elements.sidebar.classList.toggle('collapsed', !SidebarState.isOpen);
            elements.sidebar.setAttribute('aria-expanded', SidebarState.isOpen);
        }
    }
}

/**
 * Load chat history from API with pagination
 * @param {boolean} append - Whether to append to existing list or replace
 * @returns {Promise<void>}
 */
export async function loadChatHistory(append = false) {
    if (SidebarState.isLoading) return;

    try {
        SidebarState.isLoading = true;
        showLoading();

        const response = await fetch(`/api/chats?page=${SidebarState.page}&page_size=${SidebarState.pageSize}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to load chat history');
        }

        // Update state
        SidebarState.hasMore = data.has_more;
        
        if (append) {
            SidebarState.chatHistory = [...SidebarState.chatHistory, ...data.chats];
        } else {
            SidebarState.chatHistory = data.chats;
            SidebarState.page = 1;
        }

        renderChatList();
        
        if (SidebarState.chatHistory.length === 0) {
            showEmptyState();
        } else {
            hideEmptyState();
        }

    } catch (error) {
        console.error('Failed to load chat history:', error);
        showError('Failed to load chat history');
        
        if (!append && SidebarState.chatHistory.length === 0) {
            showEmptyState();
        }
    } finally {
        SidebarState.isLoading = false;
        hideLoading();
    }
}

/**
 * Render chat list items in the sidebar
 */
function renderChatList() {
    if (!elements.chatList) return;

    const fragment = document.createDocumentFragment();

    SidebarState.chatHistory.forEach(chat => {
        const chatItem = createChatItemElement(chat);
        fragment.appendChild(chatItem);
    });

    // Clear and append new items
    if (SidebarState.page === 1) {
        elements.chatList.innerHTML = '';
    }
    
    // Remove existing "load more" indicator if present
    const loadMoreIndicator = elements.chatList.querySelector('.load-more-indicator');
    if (loadMoreIndicator) {
        loadMoreIndicator.remove();
    }

    elements.chatList.appendChild(fragment);

    // Add load more indicator if there are more items
    if (SidebarState.hasMore && !SidebarState.isLoading) {
        const loadMoreEl = document.createElement('div');
        loadMoreEl.className = 'load-more-indicator';
        loadMoreEl.innerHTML = '<span class="loading-dots">Loading more...</span>';
        elements.chatList.appendChild(loadMoreEl);
    }
}

/**
 * Create a single chat item DOM element
 * @param {Object} chat - Chat object from API
 * @returns {HTMLElement}
 */
function createChatItemElement(chat) {
    const item = document.createElement('div');
    item.className = `chat-item${chat.id === SidebarState.currentChatId ? ' active' : ''}`;
    item.dataset.chatId = chat.id;

    // Sanitize title to prevent XSS
    const title = sanitizeHtml(chat.title || 'New Chat');
    
    item.innerHTML = `
        <div class="chat-item-content">
            <div class="chat-item-title" title="${escapeHtml(title)}">${truncateText(title, 50)}</div>
            <div class="chat-item-meta">
                <span class="chat-item-date">${formatDate(chat.updated_at || chat.created_at)}</span>
                <span class="chat-item-messages">${chat.message_count || 0} messages</span>
            </div>
            <div class="chat-item-preview">${escapeHtml(truncateText(chat.preview || '', 100))}</div>
        </div>
        <div class="chat-item-actions">
            <button class="btn-icon delete-chat-btn" title="Delete chat" aria-label="Delete chat">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                </svg>
            </button>
            <button class="btn-icon rename-chat-btn" title="Rename chat" aria-label="Rename chat">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
                    <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
                </svg>
            </button>
        </div>
    `;

    // Click handler for navigation
    item.addEventListener('click', (e) => {
        // Don't navigate if clicking action buttons
        if (e.target.closest('.chat-item-actions')) return;
        
        navigateToChat(chat.id);
        
        // Update active state
        document.querySelectorAll('.chat-item.active').forEach(el => el.classList.remove('active'));
        item.classList.add('active');
        
        // Close sidebar on mobile after selection
        if (window.innerWidth <= 768) {
            toggleSidebar();
        }
    });

    // Delete handler
    const deleteBtn = item.querySelector('.delete-chat-btn');
    if (deleteBtn) {
        deleteBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            await deleteChat(chat.id);
            item.remove();
            
            if (SidebarState.chatHistory.length === 0) {
                showEmptyState();
            }
            
            // If deleted chat was active, navigate to new chat
            if (chat.id === SidebarState.currentChatId) {
                await createNewChat();
            }
        });
    }

    // Rename handler
    const renameBtn = item.querySelector('.rename-chat-btn');
    if (renameBtn) {
        renameBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            await renameChat(chat.id, item);
        });
    }

    return item;
}

/**
 * Navigate to a specific chat
 * @param {number} chatId - Chat ID to navigate to
 */
function navigateToChat(chatId) {
    SidebarState.currentChatId = chatId;
    
    // Update URL without page reload
    const url = new URL(window.location);
    url.searchParams.set('chat_id', chatId);
    window.history.pushState({}, '', url);

    // Dispatch navigation event for main content area
    window.dispatchEvent(new CustomEvent('navigate-chat', {
        detail: { chatId }
    }));
}

/**
 * Create a new chat session
 * @returns {Promise<void>}
 */
async function createNewChat() {
    try {
        const response = await fetch('/api/chats/new', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({
                title: 'New Chat'
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to create new chat');
        }

        // Navigate to new chat
        navigateToChat(data.chat.id);

        // Reload chat list to include new chat at top
        await loadChatHistory();

    } catch (error) {
        console.error('Failed to create new chat:', error);
        
        // Show user-friendly error notification
        showNotification('Failed to create new chat', 'error');
        
        // Fallback: navigate to a blank state without API call
        navigateToChat(null);
        
        // Dispatch event for main content area to show blank state
        window.dispatchEvent(new CustomEvent('navigate-chat', {
            detail: { chatId: null }
        }));
        
        // Reload list anyway to maintain consistency
        await loadChatHistory();
        
       } finally {
           hideLoading();
       }
   }

   /**
     * Delete a chat session with confirmation and error handling.
     * @param {number} chatId - Chat ID to delete.
     */
   async function deleteChat(chatId) {
       try{
           // Confirm deletion with user.
           const confirmed=confirm("Are you sure you want to delete this chat? This action cannot be undone.");
           if(!confirmed){return;}
           
           const response=await fetch(`/api/chats/${chatId}`,{
               method:'DELETE',
               headers:{
                   'Content-Type':'application/json',
                   'X-Requested-With':'XMLHttpRequest'
               }
           });
           
           if(!response.ok){
               throw new Error(`HTTP error! status: ${response.status}`);
           }
           
           const data=await response.json();
           
           if(!data.success){
               throw new Error(data.error||'Failed to delete chat');
           }
           
           showNotification("Chat deleted successfully","success");
           
       }catch(error){
           console.error("Failed to delete chat:",error);
           showNotification("Failed to delete chat","error");
       }
   }

   /**
     * Rename a chat session.
     * @param {number} chatId - Chat ID.
     * @param {HTMLElement} item - Chat item element.
     */
   async function renameChat(chatId,item){
       try{
           const titleElement=item.querySelector('.chat-item-title');
           const currentTitle=titleElement.textContent||'';
           
           const newTitle=prompt("Enter new name for this chat:",currentTitle);
           
           if(!newTitle||newTitle.trim()===currentTitle){return;}
           
           const sanitizedTitle=sanitizeHtml(newTitle.trim());
           
           const response=await fetch(`/api/chats/${chatId}/rename`,{
               method:'PATCH',
               headers:{
                   'Content-Type':'application/json',
                   'X-Requested-With':'XMLHttpRequest'
               },
               body:JSON.stringify({title:sanitizedTitle})
           });
           
           if(!response.ok){
               throw new Error(`HTTP error! status: ${response.status}`);
           }
           
           const data=await response.json();
           
           if(!data.success){
               throw new Error(data.error||'Failed to rename chat');
           }
           
           titleElement.textContent=truncateText(sanitizedTitle,50);
           titleElement.title=sanitizedTitle;
           
           showNotification("Chat renamed successfully","success");
           
       }catch(error){
           console.error("Failed to rename chat:",error);
           showNotification("Failed to rename chat","error");
       }
   }

   /**
     * Search chats based on query string.
     * @param {string} query - Search query.
     */
   async function searchChats(query){
       try{
           const sanitizedQuery=sanitizeHtml(query.trim());
           
           if(!sanitizedQuery){
               await loadChatHistory(false);
               return;
           }
           
           showLoading();
           
           const response=await fetch(`/api/chats/search?q=${encodeURIComponent(sanitizedQuery)}`,{
               method:'GET',
               headers:{
                   'Content-Type':'application/json',
                   'X-Requested-With':'XMLHttpRequest'
               }
           });
           
           if(!response.ok){
               throw new Error(`HTTP error! status: ${response.status}`);
           }
           
           const data=await response.json();
           
           if(!data.success){
               throw new Error(data.error||'Search failed');
           }
           
           SidebarState.chatHistory=data.chats;
           renderChatList();
           
           if(SidebarState.chatHistory.length===0){
               showEmptyState("No chats found matching your search.");
           }else{
               hideEmptyState();
           }
           
       }catch(error){
           console.error("Search failed:",error);
       }finally{
           hideLoading();
       }
   }

   /**
     * Setup infinite scroll for loading more chats.
     */
   function setupInfiniteScroll(){
       let scrollTimeout;
       
       elements.chatList.addEventListener('scroll',()=>{
           clearTimeout(scrollTimeout);
           
           scrollTimeout=setTimeout(()=>{
               const{scrollTop,scrollHeight,clientHeight}=elements.chatList;
               
               if(scrollHeight-scrollTop-clientHeight<100&&!SidebarState.isLoading&&SidebarState.hasMore){
                   SidebarState.page++;
                   loadChatHistory(true);
               }
           },200);
       },{passive:true});
   }

   /**
     * Handle keyboard shortcuts for sidebar navigation.
     * @param {KeyboardEvent} event - Keyboard event.
     */
   function handleKeyboardShortcuts(event){
       // Ctrl+B or Cmd+B to toggle sidebar.
       if((event.ctrlKey||event.metaKey)&&event.key==='b'){
           event.preventDefault();
           toggleSidebar();
       }
       
       // Escape key closes search or sidebar on mobile.
       if(event.key==='Escape'){
           if(document.activeElement===elements.searchInput){
               elements.searchInput.value='';
               elements.searchInput.blur();
               searchChats('');
           }else if(window.innerWidth<=768&&SidebarState.isOpen){
               toggleSidebar();
           }
       }
   }

   /**
     * Handle responsive behavior on window resize.
     */
   function handleResponsiveResize(){
       if(window.innerWidth<=768&&SidebarState.isOpen){
           // Optionally auto-collapse on small screens.
       }else if(window.innerWidth>768&&!SidebarState.isOpen){
           // Optionally auto-expand on large screens.
       }
   }

   /**
     * Show loading indicator.
     */
   function showLoading(){
       if(elements.loadingIndicator){
           elements.loadingIndicator.style.display='block';
       }
       
       if(elements.chatList){
           elements.chatList.classList.add('loading');
       }
   }

   /**
     * Hide loading indicator.
     */
   function hideLoading(){
       if(elements.loadingIndicator){
           elements.loadingIndicator.style.display='none';
       }
       
       if(elements.chatList){
           elements.chatList.classList.remove('loading');
       }
   }

   /**
     * Show empty state message.
     * @param {string} [message] - Custom message.
     */
   function showEmptyState(message='No chats yet. Start a new conversation!'){
       if(elements.emptyState){
           elements.emptyState.style.display='block';
           
           const messageEl=elements.emptyState.querySelector('.empty-state-message');
           
           if(messageEl){
               messageEl.textContent=sanitizeHtml(message);
           }
       }
       
       hideError();
   }

   /**
     * Hide empty state.
     */
   function hideEmptyState(){
       if(elements.emptyState){
           elements.emptyState.style.display='none';
       }
   }

   /**
     * Show error message.
     * @param {string} message - Error message.
     */
   function showError(message='An error occurred'){
       console.error(message);
       
       if(elements.errorState){
           elements.errorState.style.display='block';
           
           const messageEl=elements.errorState.querySelector('.error-state-message');
           
           if(messageEl){
               messageEl.textContent=sanitizeHtml(message);
           }
           
           const retryBtn=elements.errorState.querySelector('.retry-btn');
           
           if(retryBtn&&!retryBtn.dataset.listenerAttached){
               retryBtn.addEventListener('click',()=>loadChatHistory(false));
               retryBtn.dataset.listenerAttached='true';
           }
       }
       
       hideEmptyState();
   }

   /**
     * Hide error state.
     */
   function hideError(){
       if(elements.errorState){
           elements.errorState.style.display='none';
       }
   }

   /**
     * Show notification toast.
     * @param {string} message - Notification message.
     * @param {'success'|'error'|'info'} type - Notification type.
     */
   function showNotification(message,type='info'){
       const notification=document.createElement('div');
       notification.className=`notification notification-${type}`;
       
       notification.innerHTML=`
         <span class="notification-icon">${getNotificationIcon(type)}</span>
         <span class="notification-message">${sanitizeHtml(message)}</span>
         <button class="notification-close" aria-label="Close notification">&times;</button>
       `;
       
       notification.querySelector('.notification-close').addEventListener('click',()=>{
         notification.remove();
       });
       
       document.body.appendChild(notification);
       
       setTimeout(()=>{
         notification.classList.add('show');
         
         setTimeout(()=>{
             notification.classList.remove('show');
             setTimeout(()=>notification.remove(),300);
         },4000);
         
       },100);
   }

   /**
     * Get icon SVG for notification type.
     * @param {'success'|'error'|'info'} type - Notification type.
     * @returns {string} SVG markup.
     */
   function getNotificationIcon(type){
       switch(type){
         case'success':
             return '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
         case'error':
             return '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
         default:
             return '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
       }
   }

   /**
     * Sanitize HTML string to prevent XSS attacks.
     * @param {string} str - Input string.
     * @returns {string} Sanitized string.
     */
   function sanitizeHtml(str){
       const div=document.createElement('div');
       div.textContent=str||'';
       return div.innerHTML;
   }

   /**
     * Escape HTML special characters.
     * @param {string} str - Input string.
     * @returns {string} Escaped string.
     */
   function escapeHtml(str){
       return str.replace(/&/g,'&amp;')
                .replace(/</g,'&lt;')
                .replace(/>/g,'&gt;')
                .replace(/"/g,'&quot;')
                .replace(/'/g,'&#039;');
   }

   /**
     * Truncate text to specified length with ellipsis.
     * @param {string} text - Text to truncate.
     * @param {number} maxLength - Maximum length before truncation.
     * @returns {string} Truncated text.
     */
   function truncateText(text,maxLength=50){
       if(!text||text.length<=maxLength){return text||'';}
       
       return text.substring(0,maxLength).trimEnd()+'...';
   }

   /**
     * Format date string for display.
     * @param {string|Date} dateInput - Date input.
     * @returns {string} Formatted date string.
     */
   function formatDate(dateInput){
       try{
         const date=new Date(dateInput);
         
         if(isNaN(date.getTime())){return'';}
         
         const now=new Date();
         const diffMs=now-date;
         const diffMins=Math.floor(diffMs/(1000*60));
         const diffHours=Math.floor(diffMs/(1000*60*60));
         const diffDays=Math.floor(diffMs/(1000*60*60*24));
         
         if(diffMins<1){return'Just now';}
         else if(diffMins<60){return`${diffMins}m ago`;}
         else if(diffHours<24){return`${diffHours}h ago`;}
         else if(diffDays<7){return`${diffDays}d ago`;}
         else{
             return date.toLocaleDateString('en-US',{
                 month:'short',
                 day:'numeric',
                 year:date.getFullYear()!==now.getFullYear()?'numeric':undefined
             });
         }
         
       }catch(error){
         console.error("Date formatting error:",error);
         return'';
       }
   }

// Export public API for external use.
export default{
   initializeSidebar,
   loadChatHistory,
   navigateToChat,
   createNewChat,
   
   get state(){return{...SidebarState};},
   
   get isOpen(){return SidebarState.isOpen;},
   
   get currentChatId(){return SidebarState.currentChatId;},
   
   set currentChatId(id){SidebarState.currentChatId=id;}
};