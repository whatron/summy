const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 8178;

// Enable CORS
app.use(cors());

// Configure multer for handling file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const dir = './uploads';
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    cb(null, dir);
  },
  filename: function (req, file, cb) {
    cb(null, `audio-${Date.now()}.raw`);
  }
});

const upload = multer({ storage: storage });

// Create uploads directory if it doesn't exist
if (!fs.existsSync('./uploads')) {
  fs.mkdirSync('./uploads', { recursive: true });
}

// Endpoint to receive audio chunks
app.post('/stream', upload.single('audio'), (req, res) => {
  console.log('Received audio chunk:', {
    filename: req.file.filename,
    size: req.file.size,
    mimetype: req.file.mimetype
  });

  // Send a dummy response
  res.json({
    segments: [
      {
        text: "This is a dummy transcription",
        t0: 0.0,
        t1: 1.0
      }
    ],
    buffer_size_ms: 1000
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
}); 