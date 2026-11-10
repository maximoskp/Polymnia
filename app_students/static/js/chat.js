(() => {
    const STORAGE_KEY = "polymnia_students_conversations_v1";

    const conversationList = document.getElementById("conversation-list");
    const messagesElement = document.getElementById("messages");
    const welcomeScreen = document.getElementById("welcome-screen");
    const input = document.getElementById("message-input");
    const form = document.getElementById("chat-form");
    const sendButton = document.getElementById("send-button");
    const newChatButton = document.getElementById("new-chat");
    const sidebar = document.getElementById("sidebar");
    const mobileMenu = document.getElementById("mobile-menu");
    const overlay = document.getElementById("sidebar-overlay");

    let conversations = loadConversations();
    let activeConversationId = conversations[0]?.id || null;
    let isSending = false;

    function makeId() {
        return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function loadConversations() {
        try {
            const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
            return Array.isArray(stored) ? stored : [];
        } catch {
            return [];
        }
    }

    function saveConversations() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
        } catch (error) {
            console.warn("Could not save chat history:", error);
        }
    }

    function activeConversation() {
        return conversations.find(item => item.id === activeConversationId);
    }

    function createConversation() {
        const conversation = {
            id: makeId(),
            title: "Νέα συνομιλία",
            updatedAt: Date.now(),
            messages: []
        };
        conversations.unshift(conversation);
        activeConversationId = conversation.id;
        saveConversations();
        render();
        closeMobileSidebar();
        input.focus();
    }

    function selectConversation(id) {
        activeConversationId = id;
        render();
        closeMobileSidebar();
    }

    function updateConversationTitle(conversation) {
        if (conversation.title !== "Νέα συνομιλία") return;
        const firstUserMessage = conversation.messages.find(m => m.role === "user");
        if (firstUserMessage) {
            conversation.title = firstUserMessage.content
                .replace(/\s+/g, " ")
                .slice(0, 42);
            if (firstUserMessage.content.length > 42) {
                conversation.title += "…";
            }
        }
    }

    function renderSidebar() {
        conversationList.replaceChildren();

        conversations
            .slice()
            .sort((a, b) => b.updatedAt - a.updatedAt)
            .forEach(conversation => {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "conversation-item"
                    + (conversation.id === activeConversationId ? " active" : "");

                const icon = document.createElement("span");
                icon.className = "conversation-icon";
                icon.textContent = "◷";

                const title = document.createElement("span");
                title.className = "conversation-title";
                title.textContent = conversation.title || "Νέα συνομιλία";

                button.append(icon, title);
                button.addEventListener("click", () => selectConversation(conversation.id));
                conversationList.append(button);
            });
    }

    function addMessageElement(message) {
        const row = document.createElement("article");
        row.className = `message-row ${message.role}`;

        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = message.role === "assistant" ? "P" : "Ε";

        const content = document.createElement("div");
        content.className = "message-content";

        const role = document.createElement("div");
        role.className = "message-role";
        role.textContent = message.role === "assistant" ? "Polymnia" : "Εσύ";

        const text = document.createElement("div");
        text.className = "message-text";
        // textContent is intentional: model output is displayed safely, not as HTML.
        text.textContent = message.content || "";

        content.append(role, text);

        if (message.role === "assistant" && Array.isArray(message.sources)
            && message.sources.length) {
            const sources = document.createElement("div");
            sources.className = "sources";

            const heading = document.createElement("div");
            heading.className = "sources-heading";
            heading.textContent = "Πηγές";
            sources.append(heading);

            message.sources.forEach((source, index) => {
                const item = document.createElement("div");
                item.className = "source-item";

                const title = document.createElement("strong");
                title.textContent = `${source.number || index + 1}. ${source.title || "Άγνωστο έγγραφο"}`;
                item.append(title);

                if (source.section) {
                    const section = document.createElement("span");
                    section.className = "source-section";
                    section.textContent = ` — ${source.section}`;
                    item.append(section);
                }
                sources.append(item);
            });

            content.append(sources);
        }

        row.append(avatar, content);
        messagesElement.append(row);
    }

    function renderMessages() {
        messagesElement.replaceChildren();
        const conversation = activeConversation();
        const messages = conversation?.messages || [];
        const hasMessages = messages.length > 0;

        welcomeScreen.hidden = hasMessages;
        messagesElement.classList.toggle("visible", hasMessages);

        messages.forEach(addMessageElement);

        requestAnimationFrame(() => {
            document.getElementById("chat-area").scrollTop =
                document.getElementById("chat-area").scrollHeight;
        });
    }

    function render() {
        renderSidebar();
        renderMessages();
    }

    function showTyping() {
        const row = document.createElement("article");
        row.className = "message-row assistant";
        row.id = "typing-row";

        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = "P";

        const content = document.createElement("div");
        content.className = "message-content";

        const role = document.createElement("div");
        role.className = "message-role";
        role.textContent = "Polymnia";

        const text = document.createElement("div");
        text.className = "typing-indicator";
        text.textContent = "Σκέφτομαι…";

        content.append(role, text);
        row.append(avatar, content);
        messagesElement.append(row);

        document.getElementById("chat-area").scrollTop =
            document.getElementById("chat-area").scrollHeight;
    }

    function setSending(value) {
        isSending = value;
        sendButton.disabled = value;
        input.disabled = value;
    }

    async function sendMessage(rawMessage) {
        const message = rawMessage.trim();
        if (!message || isSending) return;

        if (!activeConversation()) {
            createConversation();
        }

        const conversation = activeConversation();
        const previousHistory = conversation.messages
            .slice(-12)
            .map(item => ({ role: item.role, content: item.content }));

        conversation.messages.push({
            role: "user",
            content: message
        });
        conversation.updatedAt = Date.now();
        updateConversationTitle(conversation);
        saveConversations();
        render();

        input.value = "";
        resizeInput();
        setSending(true);
        showTyping();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message,
                    history: previousHistory
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Παρουσιάστηκε σφάλμα.");
            }

            const current = activeConversation();
            // The user may have switched chats while the request was running.
            const target = conversations.find(item => item.id === conversation.id);
            if (target) {
                target.messages.push({
                    role: "assistant",
                    content: data.answer || "Δεν δημιουργήθηκε απάντηση.",
                    sources: data.sources || []
                });
                target.updatedAt = Date.now();
                updateConversationTitle(target);
            }
            if (current) {
                document.getElementById("typing-row")?.remove();
            }

        } catch (error) {
            const target = conversations.find(item => item.id === conversation.id);
            if (target) {
                target.messages.push({
                    role: "assistant",
                    content: `Δεν ήταν δυνατή η ολοκλήρωση του αιτήματος.\n${error.message}`,
                    sources: []
                });
            }
        } finally {
            document.getElementById("typing-row")?.remove();
            saveConversations();
            setSending(false);
            render();
            input.focus();
        }
    }

    function resizeInput() {
        input.style.height = "auto";
        input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
    }

    function closeMobileSidebar() {
        sidebar.classList.remove("open");
        overlay.classList.remove("show");
    }

    form.addEventListener("submit", event => {
        event.preventDefault();
        sendMessage(input.value);
    });

    input.addEventListener("input", resizeInput);

    input.addEventListener("keydown", event => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    newChatButton.addEventListener("click", createConversation);

    document.querySelectorAll(".suggestion").forEach(button => {
        button.addEventListener("click", () => {
            sendMessage(button.dataset.question || "");
        });
    });

    mobileMenu.addEventListener("click", () => {
        sidebar.classList.add("open");
        overlay.classList.add("show");
    });
    overlay.addEventListener("click", closeMobileSidebar);

    if (!activeConversationId) {
        // Keep the welcome screen visible until the first message is sent.
        render();
    } else {
        render();
    }
})();
