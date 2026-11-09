
const launcher = document.getElementById("polymnia-launcher");
const chat = document.getElementById("polymnia-chat");
const closeButton = document.getElementById("polymnia-close");

const messagesContainer = document.getElementById("polymnia-messages");

const input = document.getElementById("polymnia-input");
const sendButton = document.getElementById("polymnia-send");

const status = document.getElementById("polymnia-status");


// --------------------------------------------------
// Conversation history
// --------------------------------------------------

let history = [];


// --------------------------------------------------
// Open / close
// --------------------------------------------------

launcher.addEventListener("click", () => {
    chat.classList.add("open");
    input.focus();
});

closeButton.addEventListener("click", () => {
    chat.classList.remove("open");
});


// --------------------------------------------------
// Display messages safely
// --------------------------------------------------

function addMessage(content, role) {

    const message = document.createElement("div");

    message.className = `chat-message ${role}`;

    message.textContent = content;

    messagesContainer.appendChild(message);

    messagesContainer.scrollTop =
        messagesContainer.scrollHeight;

    return message;
}


// --------------------------------------------------
// Display sources
// --------------------------------------------------

function addSources(container, sources) {

    if (!sources || sources.length === 0) {
        return;
    }

    const section = document.createElement("div");

    section.className = "chat-sources";

    const heading = document.createElement("strong");

    heading.textContent = "Πηγές:";

    section.appendChild(heading);

    const list = document.createElement("ul");

    sources.forEach(source => {

        const item = document.createElement("li");

        const link = document.createElement("a");

        const title = source.title || "Έγγραφο";

        const sectionName = source.section
            ? ` – ${source.section}`
            : "";

        link.textContent = `[${source.number}] ${title}${sectionName}`;

        link.href = source.url || source.original_url || "#";

        if (source.url) {
            link.target = "_blank";
            link.rel = "noopener noreferrer";
        }

        item.appendChild(link);

        list.appendChild(item);

    });

    section.appendChild(list);

    container.appendChild(section);

    messagesContainer.scrollTop =
        messagesContainer.scrollHeight;
}


// --------------------------------------------------
// Send message
// --------------------------------------------------

async function sendMessage() {

    const question = input.value.trim();

    if (!question) {
        return;
    }

    // Display user message.
    addMessage(question, "user");

    input.value = "";

    sendButton.disabled = true;

    status.textContent = "Η Polymnia επεξεργάζεται την ερώτησή σας...";

    // Preserve history before adding the current question.
    const previousHistory = history.slice(-12);

    history.push({
        role: "user",
        content: question
    });

    try {

        const response = await fetch("/api/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                message: question,

                history: previousHistory

            })

        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Chat request failed"
            );
        }

        const answer = data.answer;

        const message = addMessage(
            answer,
            "assistant"
        );

        addSources(message, data.sources);

        history.push({
            role: "assistant",
            content: answer
        });

        // Keep the last six exchanges.
        history = history.slice(-12);

        status.textContent = "";

    } catch (error) {

        addMessage(
            "Παρουσιάστηκε σφάλμα κατά την επικοινωνία με την Polymnia. Παρακαλώ δοκιμάστε ξανά.",
            "error"
        );

        // Remove the failed question from conversation history.
        history.pop();

        status.textContent = "";

        console.error(error);

    } finally {

        sendButton.disabled = false;

        input.focus();

    }

}


// --------------------------------------------------
// Event listeners
// --------------------------------------------------

sendButton.addEventListener("click", sendMessage);

input.addEventListener("keydown", event => {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        sendMessage();

    }

});
