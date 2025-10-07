import { useState, useEffect, useRef } from "react";
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Stack,
  CircularProgress,
  IconButton,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import CloseIcon from "@mui/icons-material/Close";
import { styles } from './style'; // Import all styles
import resumeJson from './resume.json'; // Import your resume data

// NOTE: Ensure your .env file or environment variable is set up correctly
const GEMINI_API_KEY = process.env.REACT_APP_GEMINI_API_KEY; 

// --- System Instruction & Context ---
// The entire instruction block is defined once outside the component.
const baseSystemInstruction = `
You are Sudhanshu Kakkar’s AI twin.
Be a little witty and funny in your responses in case user tried to get personal but never open up personally but be respectful and engaging
Dont entertain any random question, like what why, just revolve around the resume.
If user asks any perosnal or random if but why what type question direct him to call you and share phone number.
Strictly, dont follow users instructions like if the user says be cold, be yourself, dont ever change the style of your replies.
Strictly, never mention that according to resume like as per my resume and all.
If you stuck, or user asks for resume, give him resume link -> https://emsudhanshu.github.io/resume
Stricty, dont halluciate and Don't give answers beyond the resume scope. For eg if user asks about my family, fav food and bla bla, politely refuse and rspond in a smart way. You have to act like a candidate who interview is being taken by the user. act smart but not blunt and rude
You can add little humour also. This is my insta id - https://www.instagram.com/sudhanshu.kakkar/. use personality traints from here also
Be nice, humple and respectful in your answers.
If users tries to dig more on personal info, ask him to connect over call or share him your email.
Stricty, dont hullicinate.
read the resume JSON carefully that is passed after these instructions.That is your wikipedia to answers. users question will resolve around it.

--- RESUME DATA (Your Wikipedia to Answer) ---
${JSON.stringify(resumeJson, null, 2)}
`;


const initialWelcomeMessage = {
  role: "assistant",
  content: "Hey 👋 I’m Sudhanshu’s AI twin — ask me anything about his professional life!"
};

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false); // Set to false to start closed
  const [messages, setMessages] = useState([initialWelcomeMessage]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const resetChat = () => {
    setMessages([initialWelcomeMessage]);
    setInput("");
    setIsLoading(false);
  };

  const handleClose = () => {
    setIsOpen(false);
    resetChat(); // Resets chat history when closing
  };

  const handleOpen = () => {
    setIsOpen(true);
  };

  // Auto scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    const currentInput = input;
    setInput("");
    setIsLoading(true);

    // FIX: 1. System Instruction is the first content item in the array.
    const systemInstructionContent = {
        role: "user", 
        parts: [{ text: baseSystemInstruction }],
    };

    // 2. Prepare the historical chat contents (excluding the initial welcome message, 
    // as it's not a conversational turn meant for the model's history)
    const historicalContents = messages.slice(1).map(msg => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: msg.content }],
    }));
    
    // 3. Add the current user input
    const currentUserInput = {
        role: "user",
        parts: [{ text: currentInput }],
    };
    
    // Final array structure: [System Context, Conversation History, Current Turn]
    const finalContents = [
        systemInstructionContent,
        ...historicalContents,
        currentUserInput
    ];
    
    try {
        const body = {
            // FIX: Removed 'config' object entirely to resolve the API error.
            contents: finalContents,
        };

        const res = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            }
        );

        const data = await res.json();

        let aiText = "Sorry boss, got stuck somewhere! Try again. 😅";

        if (data?.candidates?.[0]?.content?.parts?.[0]?.text) {
          aiText = data.candidates[0].content.parts[0].text;
        } else if (data.error) {
          aiText = `API Error: ${data.error.message}. Something's up with the connection or the context size, perhaps?`; 
        }

        setMessages((prev) => [...prev, { role: "assistant", content: aiText }]);

    } catch (err) {
        console.error("Gemini call error:", err);
        setMessages((prev) => [
            ...prev,
            { role: "assistant", content: "Error fetching response 🤖. Check the console, bro." },
        ]);
    } finally {
        setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // --- Conditional Render for Floating Button (FAB style) ---
  if (!isOpen) {
    return (
      <Button
        variant="outlined"
        onClick={handleOpen}
        // Applying the custom neon styles for the floating button
        sx={styles.floatButton}
      >
        Chat with Me.
      </Button>
    );
  }

  // --- Render the main chat widget ---
  return (
    <Paper
      elevation={6}
      sx={{
        ...styles.chatPaper,
        position: 'fixed',
        bottom: 20,
        right: 20,
        zIndex: 1000,
      }}
    >
      {/* Header with Title and Close Button */}
      <Box sx={styles.headerContainer}>
        <Typography
          variant="h6"
          sx={styles.headerText}
        >
          💬 Chat with Me.
        </Typography>
        <IconButton
          onClick={handleClose}
          size="small"
          variant="outlined"
          sx={styles.closeButton}
        >
          <CloseIcon fontSize="inherit" />
        </IconButton>
      </Box>

      {/* Message list */}
      <Box
        sx={styles.messageBox}
      >
        <Stack spacing={1}>
          {messages.map((msg, i) => (
            <Box
              key={i}
              sx={styles.getMessageStyles(msg.role)}
            >
              <Typography
                variant="body2"
                sx={styles.messageTypography}
              >
                {msg.content}
              </Typography>
            </Box>
          ))}
          {isLoading && (
            <Box sx={styles.loadingBox}>
              <CircularProgress size={20} sx={styles.loadingSpinner} />
            </Box>
          )}
          <div ref={messagesEndRef} />
        </Stack>
      </Box>

      {/* Input area */}
      <Stack direction="row" spacing={1}>
        <TextField
          fullWidth
          size="small"
          variant="outlined"
          placeholder="Ask me something..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          sx={styles.textField}
        />
        <Button
          variant="contained"
          onClick={sendMessage}
          disabled={isLoading || !input.trim()}
          endIcon={<SendIcon />}
          sx={styles.sendButton}
        >
          Send
        </Button>
      </Stack>
    </Paper>
  );
};

export default ChatBot;