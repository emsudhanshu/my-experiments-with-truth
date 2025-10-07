// style.js

export const styles = {
    // --- Floating Button Style (FAB) ---
    floatButton: {
        textTransform: 'Capitalize',
        position: 'fixed',
        bottom: 20,
        right: 20,
        bgcolor: '#0d0d0d',
        color: '#39FF14',
        borderColor: '#39FF14',
        borderRadius: 5,
        p: 2,
        zIndex: 1000,
        boxShadow: '0 0 10px #39FF14',
        '&:hover': {
            color: '#fff',
            bgcolor: '#FF6D00',
            borderColor: '#ff6d00',
            boxShadow: '0 0 10px #ff6d00',
        }
    },

    // --- Chat Window Styles (Made Responsive) ---
    chatPaper: {
        // RESPONSIVE WIDTH: 90% of viewport width on small screens, max 420px on large screens
        width: { xs: '90vw', sm: 420 },
        // RESPONSIVE HEIGHT: 70% of viewport height on small screens, fixed 550px on large screens
        height: { xs: '70vh', sm: 550 },
        p: 2,
        display: "flex",
        flexDirection: "column",
        bgcolor: "#0d0d0d",
        borderRadius: 3,
        border: "1px solid #39FF14",
    },

    headerContainer: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        mb: 2,
    },

    headerText: {
        color: "#39FF14",
        fontWeight: 600,
        flexGrow: 1,
        textAlign: "center",
        // Removed fixed ml: 5 for true centering across all screen sizes
    },

    closeButton: {
        minWidth: 'auto',
        p: 0.5,
        color: '#39FF14',
        borderColor: '#39FF14',
        '&:hover': {
            borderColor: '#FF3914',
            color: '#FF3914',
        }
    },

    messageBox: {
        flexGrow: 1,
        overflowY: "auto",
        bgcolor: "#111",
        borderRadius: 2,
        p: 1,
        mb: 2,
        // Custom Scrollbar
        "&::-webkit-scrollbar": {
            width: "6px",
        },
        "&::-webkit-scrollbar-thumb": {
            backgroundColor: "#39FF14",
            borderRadius: "10px",
        },
        "&::-webkit-scrollbar-track": {
            backgroundColor: "#1e1e1e",
        },
    },

    loadingBox: {
        display: "flex",
        justifyContent: "flex-start",
        py: 1,
    },

    loadingSpinner: {
        color: "#39FF14",
    },

    textField: {
        input: {
            color: "#39FF14"
        },
        "& .MuiOutlinedInput-root": {
            "& fieldset": {
                borderColor: "#39FF14"
            },
            "&:hover fieldset": {
                borderColor: "#39FF14"
            },
            "&.Mui-focused fieldset": {
                borderColor: "#39FF14"
            },
        },
    },

    sendButton: {
        bgcolor: "#39FF14",
        color: "#000",
        fontWeight: "bold",
        "&:hover": {
            bgcolor: "#4AFF2E"
        },
    },

    getMessageStyles: (role) => ({
        alignSelf: role === "user" ? "flex-end" : "flex-start",
        bgcolor: role === "user" ? "#39FF14" : "#1e1e1e",
        color: role === "user" ? "#000" : "#39FF14",
        px: 1.5,
        py: 1,
        borderRadius: 2,
        maxWidth: "80%",
        wordBreak: 'break-word',
    }),

    messageTypography: {
        whiteSpace: "pre-wrap",
        fontFamily: "monospace",
    }
};